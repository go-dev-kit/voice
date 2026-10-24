package modelstore

import (
	"archive/tar"
	"compress/bzip2"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
)

const (
	defaultMaxDownload = 8 << 30
	defaultMaxExtract  = 8 << 30
	defaultMaxFiles    = 10000
)

// Result describes a completed Fetch.
type Result struct {
	// Path is the fetched file, or the model directory for archives.
	Path string
	// SHA256 is the digest of the downloaded bytes. For an unpinned entry
	// fetched with WithAllowUnpinned it is the value to pin in the manifest.
	SHA256 string
	// Pinned reports whether SHA256 was checked against the manifest.
	Pinned bool
}

type fetchConfig struct {
	allowUnpinned bool
	maxDownload   int64
	maxExtract    int64
	maxFiles      int
}

// Option customizes Fetch.
type Option func(*fetchConfig)

// WithAllowUnpinned lets Fetch proceed for an entry with an empty sha256. The
// computed digest is returned in Result.SHA256 so it can be pinned. Without
// this option an unpinned entry fails with ErrUnpinned before any request.
func WithAllowUnpinned() Option { return func(c *fetchConfig) { c.allowUnpinned = true } }

// WithMaxDownloadBytes bounds the downloaded size (default 8 GiB).
func WithMaxDownloadBytes(n int64) Option { return func(c *fetchConfig) { c.maxDownload = n } }

// WithMaxExtractBytes bounds the total bytes written when extracting an
// archive (default 8 GiB), which stops a compression bomb.
func WithMaxExtractBytes(n int64) Option { return func(c *fetchConfig) { c.maxExtract = n } }

// WithMaxFiles bounds the number of archive members (default 10000).
func WithMaxFiles(n int) Option { return func(c *fetchConfig) { c.maxFiles = n } }

// Fetch downloads entry into dir. The body is written to a temporary file in
// dir and its SHA-256 is verified before the content is used. Archives are
// extracted into a temporary directory in dir, checked against the entry's
// expected files, and renamed into place, so a failed or canceled fetch
// leaves no partial model behind. A client of nil means http.DefaultClient.
//
// The result is Entry.Target(dir). An existing target is replaced.
func Fetch(ctx context.Context, client *http.Client, entry Entry, dir string, opts ...Option) (Result, error) {
	cfg := fetchConfig{maxDownload: defaultMaxDownload, maxExtract: defaultMaxExtract, maxFiles: defaultMaxFiles}
	for _, o := range opts {
		o(&cfg)
	}
	if err := entry.Validate(); err != nil {
		return Result{}, err
	}
	pinned := entry.SHA256 != ""
	if !pinned && !cfg.allowUnpinned {
		return Result{}, fmt.Errorf("%w: %s", ErrUnpinned, entry.Name)
	}
	if client == nil {
		client = http.DefaultClient
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Result{}, err
	}

	tmp, err := os.CreateTemp(dir, ".modelstore-dl-*")
	if err != nil {
		return Result{}, err
	}
	defer os.Remove(tmp.Name()) // no-op once renamed
	got, err := download(ctx, client, entry.URL, tmp, cfg.maxDownload)
	closeErr := tmp.Close()
	if err != nil {
		return Result{}, err
	}
	if closeErr != nil {
		return Result{}, closeErr
	}
	if pinned && got != entry.SHA256 {
		return Result{}, &DigestError{What: entry.URL, Want: entry.SHA256, Got: got}
	}

	res := Result{Path: entry.Target(dir), SHA256: got, Pinned: pinned}
	if entry.Kind() == ArchiveNone {
		if err := checkSingle(&entry, tmp.Name()); err != nil {
			return Result{}, err
		}
		if err := os.Chmod(tmp.Name(), 0o644); err != nil {
			return Result{}, err
		}
		return res, replace(tmp.Name(), res.Path)
	}

	stage, err := os.MkdirTemp(dir, ".modelstore-x-*")
	if err != nil {
		return Result{}, err
	}
	defer os.RemoveAll(stage) // no-op once renamed
	if err := extract(ctx, tmp.Name(), entry.Kind(), stage, cfg); err != nil {
		return Result{}, err
	}
	src := stage
	// Model archives usually wrap everything in one top-level directory;
	// promote it so Files are relative to the model directory.
	if ents, err := os.ReadDir(stage); err == nil && len(ents) == 1 && ents[0].IsDir() {
		src = filepath.Join(stage, ents[0].Name())
	}
	if err := verifyStaged(&entry, src); err != nil {
		return Result{}, err
	}
	if err := os.Chmod(src, 0o755); err != nil {
		return Result{}, err
	}
	return res, replace(src, res.Path)
}

// checkSingle verifies the expected per-file digest of a single-file entry,
// when it differs from the entry digest.
func checkSingle(e *Entry, p string) error {
	if len(e.Files) == 1 && e.Files[0].SHA256 != "" {
		return VerifyFile(p, e.Files[0].SHA256)
	}
	return nil
}

func verifyStaged(e *Entry, root string) error {
	r, err := os.OpenRoot(root)
	if err != nil {
		return err
	}
	defer r.Close()
	for _, f := range e.Files {
		if err := verifyInRoot(r, f); err != nil {
			return err
		}
	}
	return nil
}

// replace moves src to dst, displacing an existing dst. The final rename is
// atomic; replacing an existing directory is two renames and can briefly leave
// no dst.
func replace(src, dst string) error {
	if _, err := os.Lstat(dst); err == nil {
		old := dst + ".old-" + filepath.Base(src)
		if err := os.Rename(dst, old); err != nil {
			return err
		}
		if err := os.Rename(src, dst); err != nil {
			_ = os.Rename(old, dst)
			return err
		}
		return os.RemoveAll(old)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return os.Rename(src, dst)
}

func download(ctx context.Context, client *http.Client, url string, w io.Writer, limit int64) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("modelstore: GET %s: %s", url, resp.Status)
	}
	if resp.ContentLength > limit {
		return "", fmt.Errorf("%w: download %d bytes > %d", ErrTooLarge, resp.ContentLength, limit)
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(w, h), &ctxReader{ctx: ctx, r: io.LimitReader(resp.Body, limit+1)})
	if err != nil {
		return "", err
	}
	if n > limit {
		return "", fmt.Errorf("%w: download exceeds %d bytes", ErrTooLarge, limit)
	}
	if resp.ContentLength >= 0 && n != resp.ContentLength {
		return "", fmt.Errorf("modelstore: GET %s: short body %d of %d bytes", url, n, resp.ContentLength)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ctxReader fails reads once ctx is done, so cancellation interrupts local
// work as well as the network.
type ctxReader struct {
	ctx context.Context
	r   io.Reader
}

func (c *ctxReader) Read(p []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	return c.r.Read(p)
}

func extract(ctx context.Context, archive string, kind ArchiveKind, dst string, cfg fetchConfig) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	var src io.Reader
	switch kind {
	case ArchiveTarGz:
		zr, err := gzip.NewReader(f)
		if err != nil {
			return fmt.Errorf("modelstore: gzip: %w", err)
		}
		defer zr.Close()
		src = zr
	case ArchiveTarBz2:
		src = bzip2.NewReader(f)
	default:
		return fmt.Errorf("modelstore: cannot extract archive kind %q", kind)
	}
	root, err := os.OpenRoot(dst)
	if err != nil {
		return err
	}
	defer root.Close()

	tr := tar.NewReader(&ctxReader{ctx: ctx, r: src})
	var written int64
	var members int
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("modelstore: read archive: %w", err)
		}
		members++
		if members > cfg.maxFiles {
			return fmt.Errorf("%w: more than %d archive members", ErrTooLarge, cfg.maxFiles)
		}
		name, err := safeName(hdr.Name)
		if err != nil {
			return err
		}
		if name == "." {
			continue
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := root.MkdirAll(name, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if hdr.Size > cfg.maxExtract-written {
				return fmt.Errorf("%w: extraction exceeds %d bytes", ErrTooLarge, cfg.maxExtract)
			}
			if err := writeMember(root, name, hdr, tr); err != nil {
				return err
			}
			written += hdr.Size
		case tar.TypeSymlink:
			if err := checkLink(root, name, hdr.Linkname); err != nil {
				return err
			}
			if err := mkParent(root, name); err != nil {
				return err
			}
			if err := root.Symlink(hdr.Linkname, name); err != nil {
				return err
			}
		default:
			return fmt.Errorf("%w: %q: unsupported member type %q", ErrUnsafeArchive, hdr.Name, hdr.Typeflag)
		}
	}
}

// safeName cleans an archive member name and rejects absolute paths and
// traversal. It returns a slash-free-of-dots OS path relative to the root.
func safeName(name string) (string, error) {
	if path.IsAbs(name) || filepath.IsAbs(name) || name == "" {
		return "", fmt.Errorf("%w: %q: absolute or empty path", ErrUnsafeArchive, name)
	}
	clean := path.Clean(name)
	if clean == "." {
		return ".", nil
	}
	if !filepath.IsLocal(filepath.FromSlash(clean)) {
		return "", fmt.Errorf("%w: %q: path escapes the target directory", ErrUnsafeArchive, name)
	}
	return filepath.FromSlash(clean), nil
}

// checkLink rejects symlinks whose target is absolute or resolves outside the
// extraction root. The target is resolved lexically, which is only sound when
// no component of the link's parent path or of the target itself is a symlink,
// so both cases are refused: a symlink in the parent path (a -> . then
// a/b -> ../x) would make the link physically live elsewhere, and a ".." that
// follows a normal target component could step out through a symlink.
// os.Root additionally blocks writes through such links.
func checkLink(root *os.Root, name, target string) error {
	if target == "" || path.IsAbs(target) || filepath.IsAbs(target) {
		return fmt.Errorf("%w: %q: symlink target %q is absolute or empty", ErrUnsafeArchive, name, target)
	}
	parts := strings.Split(filepath.ToSlash(target), "/")
	for len(parts) > 0 && (parts[0] == ".." || parts[0] == ".") {
		parts = parts[1:]
	}
	if slices.Contains(parts, "..") {
		return fmt.Errorf("%w: %q: symlink target %q has \"..\" after a path element", ErrUnsafeArchive, name, target)
	}
	resolved := path.Join(path.Dir(filepath.ToSlash(name)), target)
	if !filepath.IsLocal(filepath.FromSlash(resolved)) {
		return fmt.Errorf("%w: %q: symlink target %q escapes the target directory", ErrUnsafeArchive, name, target)
	}
	// Lstat every ancestor of the link; a missing one is created by mkParent
	// as a real directory.
	dir := filepath.Dir(name)
	for dir != "." {
		fi, err := root.Lstat(dir)
		if err == nil && fi.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("%w: %q: symlink inside symlinked directory %q", ErrUnsafeArchive, name, dir)
		}
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		dir = filepath.Dir(dir)
	}
	return nil
}

func mkParent(root *os.Root, name string) error {
	if dir := filepath.Dir(name); dir != "." {
		return root.MkdirAll(dir, 0o755)
	}
	return nil
}

func writeMember(root *os.Root, name string, hdr *tar.Header, r io.Reader) error {
	if err := mkParent(root, name); err != nil {
		return err
	}
	mode := fs.FileMode(0o644)
	if hdr.Mode&0o111 != 0 {
		mode = 0o755
	}
	f, err := root.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return err
	}
	n, err := io.Copy(f, io.LimitReader(r, hdr.Size))
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return err
	}
	if n != hdr.Size {
		return fmt.Errorf("modelstore: %q: short archive member", hdr.Name)
	}
	return nil
}
