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
	"sync"
	"time"
)

const (
	defaultMaxDownload = 8 << 30
	defaultMaxExtract  = 8 << 30
	defaultMaxFiles    = 10000
	// tempPrefix starts every temporary entry Fetch creates in dir. Fetch
	// names them uniquely and removes them itself; after a crash a leftover
	// is harmless and never blocks a later Fetch. [Sweep] removes stale ones,
	// safely only while no Fetch into dir is running in any process.
	tempPrefix = ".modelstore-"
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
// leaves no partial model behind.
//
// A client of nil means [defaultHTTPClient]: the same transport policy as
// [http.DefaultClient] (which it shares) plus an overall response-header
// timeout, a global redirect cap, and a refusal to follow a redirect to a
// non-https URL. Even so, model bytes are trusted only through the pinned
// digest, never through the transport.
//
// The result is Entry.Target(dir). An existing target is replaced. Symlinks in
// an archive are kept only when they resolve inside the final model directory.
//
// Install atomicity: the single-file install re-digests the installed bytes
// after the rename and retries on mismatch, so a writer that swaps the
// temporary file's name between digest check and install cannot plant
// unverified bytes — the fetch fails with a DigestError instead. An archive
// install renames a verified staging directory by path; the same re-check is
// not affordable there (it would mean re-extracting), so a writer holding
// dir could in principle swap that path between verification and rename,
// which the in-process lock and cross-process retry make unlikely but cannot
// rule out. Fetching into a directory other writers create files in is not
// supported.
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
		client = defaultHTTPClient()
	}
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return Result{}, err
	}
	// MkdirAll does not tighten a directory that already exists with looser
	// permissions; enforce the mode the store contract promises.
	if fi, err := os.Stat(dir); err == nil && fi.Mode().Perm()&0o077 != 0 {
		if err := os.Chmod(dir, 0o750); err != nil { // #nosec G302 -- a directory needs the search bit; 0750 is the store contract
			return Result{}, err
		}
	}

	tmp, err := os.CreateTemp(dir, tempPrefix+"dl-*")
	if err != nil {
		return Result{}, err
	}
	defer os.Remove(tmp.Name()) // no-op once renamed
	// The handle stays open until the content has been used: extraction and
	// every digest re-check read through it, never through the name. The
	// single-file install renames this handle, so its installed bytes are
	// exactly the bytes hashed here; an archive install renames a staging
	// directory by path (see the Fetch doc for the residual caveat).
	defer tmp.Close()
	got, err := download(ctx, client, entry.URL, tmp, cfg.maxDownload)
	if err != nil {
		return Result{}, err
	}
	if pinned && got != entry.SHA256 {
		return Result{}, &DigestError{What: entry.URL, Want: entry.SHA256, Got: got}
	}

	res := Result{Path: entry.Target(dir), SHA256: got, Pinned: pinned}
	if entry.Kind() == ArchiveNone {
		if err := checkSingle(&entry, tmp); err != nil {
			return Result{}, err
		}
		if err := tmp.Chmod(0o644); err != nil {
			return Result{}, err
		}
		if err := tmp.Close(); err != nil {
			return Result{}, err
		}
		return res, installVerified(dir, tmp.Name(), res.Path, res.SHA256)
	}

	stage, err := os.MkdirTemp(dir, tempPrefix+"x-*")
	if err != nil {
		return Result{}, err
	}
	defer os.RemoveAll(stage) // no-op once renamed
	if _, err := tmp.Seek(0, io.SeekStart); err != nil {
		return Result{}, err
	}
	if err := extract(ctx, tmp, entry.Kind(), stage, cfg); err != nil {
		return Result{}, err
	}
	src := stage
	// Model archives usually wrap everything in one top-level directory;
	// promote it so Files are relative to the model directory.
	if ents, err := os.ReadDir(stage); err == nil && len(ents) == 1 && ents[0].IsDir() {
		src = filepath.Join(stage, ents[0].Name())
	}
	// Links are checked against the final model directory, not the stage:
	// promoting the top-level directory shortens every link's path by one
	// element, so a link that stayed inside the stage can resolve outside src.
	if err := checkTreeLinks(src); err != nil {
		return Result{}, err
	}
	if err := verifyStaged(&entry, src); err != nil {
		return Result{}, err
	}
	// Model files are not secret and must be readable by the service account;
	// directories need the search bit.
	if err := os.Chmod(src, 0o755); err != nil { // #nosec G302
		return Result{}, err
	}
	return res, replace(dir, src, res.Path)
}

// checkSingle verifies the expected per-file digest of a single-file entry,
// when it differs from the entry digest, reading through the open handle.
func checkSingle(e *Entry, f *os.File) error {
	if len(e.Files) == 1 && e.Files[0].SHA256 != "" {
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return err
		}
		got, err := digestReader(f)
		if err != nil {
			return err
		}
		if got != e.Files[0].SHA256 {
			return &DigestError{What: e.Files[0].Path, Want: e.Files[0].SHA256, Got: got}
		}
	}
	return nil
}

// checkTreeLinks rejects any symlink under root that is absolute or whose
// lexical target leaves root. checkLink already refuses links reached through
// other links and ".." after a normal element, so lexical resolution is sound.
func checkTreeLinks(root string) error {
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type()&fs.ModeSymlink == 0 {
			return nil
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		target, err := os.Readlink(p)
		if err != nil {
			return err
		}
		resolved := path.Join(path.Dir(filepath.ToSlash(rel)), filepath.ToSlash(target))
		if path.IsAbs(target) || !filepath.IsLocal(filepath.FromSlash(resolved)) {
			return fmt.Errorf("%w: %q: symlink target %q escapes the model directory", ErrUnsafeArchive, filepath.ToSlash(rel), target)
		}
		return nil
	})
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

// replaceMu serializes installs within a process; the retry loops cover
// other processes.
var replaceMu sync.Mutex

// installDigest digests an installed file for installVerified's re-check.
var installDigest = DigestFile

// replace moves src to dst, displacing an existing dst. dst must be strictly
// inside dir. See replaceLocked for the mechanics.
func replace(dir, src, dst string) error {
	rel, err := filepath.Rel(dir, dst)
	if err != nil || rel == "." || !filepath.IsLocal(rel) {
		return fmt.Errorf("modelstore: refusing to replace %q: not inside %q", dst, dir)
	}
	replaceMu.Lock()
	defer replaceMu.Unlock()
	return replaceLocked(dir, src, dst)
}

// replaceLocked is replace's rename loop with the store lock already held.
// The final rename is atomic; replacing an existing target is two renames and
// can briefly leave no dst. The displaced target is parked in a uniquely
// named ".modelstore-old-*" directory, so leftovers from a crash never block
// a later Fetch. Deleting the parked copy after a successful swap is
// best-effort: the model is installed either way. Concurrent Fetches of the
// same target race benignly (one install wins); the loop retries when
// another goroutine or process changes dst underneath it.
func replaceLocked(dir, src, dst string) error {
	var last error
	for range 8 {
		if _, err := os.Lstat(dst); errors.Is(err, fs.ErrNotExist) {
			if last = os.Rename(src, dst); last == nil {
				return nil
			}
			continue
		} else if err != nil {
			return err
		}
		holder, err := os.MkdirTemp(dir, tempPrefix+"old-*")
		if err != nil {
			return err
		}
		old := filepath.Join(holder, "old")
		if last = os.Rename(dst, old); last != nil {
			_ = os.Remove(holder)
			if errors.Is(last, fs.ErrNotExist) {
				continue // another Fetch moved dst first
			}
			return last
		}
		if last = os.Rename(src, dst); last != nil {
			// Put the previous model back unless another Fetch already
			// installed one.
			_ = os.Rename(old, dst)
			_ = os.RemoveAll(holder)
			if _, err := os.Lstat(dst); err == nil {
				continue
			}
			return last
		}
		_ = os.RemoveAll(holder)
		return nil
	}
	return last
}

// installVerified installs a downloaded single file and then re-digests the
// installed bytes. The re-check closes the gap a path rename cannot close:
// a rename installs whatever its source name points at when it fires, not
// necessarily the inode whose bytes were hashed. A mismatch means the file
// was swapped under us; because the temporary source is gone after the
// rename there is nothing to retry, so the fetch fails loudly with a
// DigestError instead of leaving unverified bytes installed.
func installVerified(dir, src, dst, want string) error {
	rel, err := filepath.Rel(dir, dst)
	if err != nil || rel == "." || !filepath.IsLocal(rel) {
		return fmt.Errorf("modelstore: refusing to replace %q: not inside %q", dst, dir)
	}
	replaceMu.Lock()
	defer replaceMu.Unlock()
	if err := replaceLocked(dir, src, dst); err != nil {
		return err
	}
	got, err := installDigest(dst)
	if err != nil {
		return err
	}
	if got != want {
		return &DigestError{What: dst, Want: want, Got: got}
	}
	return nil
}

// defaultHTTPClient is the client used when Fetch's client argument is nil:
// the http.DefaultTransport policies plus the timeouts and redirect bounds a
// download of untrusted bytes needs to terminate on its own. It shares
// http.DefaultClient's transport, so proxy and TLS environment settings keep
// applying; only the client-level timeout and redirect policy are added.
func defaultHTTPClient() *http.Client {
	return &http.Client{
		Transport: http.DefaultTransport,
		// Overall budget including body read; Fetch cancels earlier through
		// the context, this is the backstop for a stalled server.
		Timeout: 30 * time.Minute,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("%w: more than %d redirects", ErrTooLarge, len(via))
			}
			if req.URL.Scheme != "https" {
				return fmt.Errorf("modelstore: redirect to %s: https required", req.URL)
			}
			return nil
		},
	}
}

// Sweep removes temporary entries a crashed Fetch left in dir: download
// files, staging directories, and parked old targets carrying tempPrefix.
// Sweep takes the same lock as an install, but a Fetch running concurrently
// in another process is invisible to it, so call Sweep only while no Fetch
// into dir is running anywhere. Crash leftovers are harmless to Fetch —
// Sweep exists to reclaim the disk space of a store that crashes often.
func Sweep(dir string) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	var firstErr error
	for _, e := range ents {
		if !strings.HasPrefix(e.Name(), tempPrefix) {
			continue
		}
		if err := os.RemoveAll(filepath.Join(dir, e.Name())); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
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

// extract unpacks the archive read from f, which must be positioned at its
// start, into dst.
func extract(ctx context.Context, f io.Reader, kind ArchiveKind, dst string, cfg fetchConfig) error {
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
