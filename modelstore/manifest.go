// Package modelstore describes, downloads, and verifies VoiceKit model
// assets. VoiceKit ships no model weights; a deployment lists the models it
// needs in a JSON manifest and uses this package to fetch them with pinned
// SHA-256 digests and safe archive extraction. The package depends only on the
// standard library.
package modelstore

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// ArchiveKind is how an entry's download is packaged.
type ArchiveKind string

const (
	// ArchiveNone is a single file stored as-is.
	ArchiveNone ArchiveKind = "none"
	// ArchiveTarGz is a gzip-compressed tar archive.
	ArchiveTarGz ArchiveKind = "tar.gz"
	// ArchiveTarBz2 is a bzip2-compressed tar archive.
	ArchiveTarBz2 ArchiveKind = "tar.bz2"
)

// Manifest version understood by this package.
const ManifestVersion = 1

// Sentinel errors, matchable with errors.Is.
var (
	// ErrInvalidManifest wraps every manifest parse or validation failure.
	ErrInvalidManifest = errors.New("modelstore: invalid manifest")
	// ErrUnpinned is returned when an entry has no sha256 and the caller did
	// not opt in with WithAllowUnpinned.
	ErrUnpinned = errors.New("modelstore: entry has no pinned sha256")
	// ErrUnsafeArchive wraps every rejected archive member.
	ErrUnsafeArchive = errors.New("modelstore: unsafe archive")
	// ErrTooLarge is returned when a download or extraction exceeds its limit.
	ErrTooLarge = errors.New("modelstore: size limit exceeded")
	// ErrNotFound is returned by Manifest.Lookup for an unknown name.
	ErrNotFound = errors.New("modelstore: model not found")
)

// DigestError reports a SHA-256 mismatch.
type DigestError struct {
	What string // file or URL that was hashed
	Want string
	Got  string
}

func (e *DigestError) Error() string {
	return fmt.Sprintf("modelstore: sha256 mismatch for %s: want %s got %s", e.What, e.Want, e.Got)
}

// File is one file expected inside a model. SHA256 is optional; when set it
// is checked by Entry.Verify.
type File struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256,omitempty"`
}

// Entry describes one downloadable model.
type Entry struct {
	Name string `json:"name"`
	// Family is the model family (for example "transducer" or "vad").
	Family string `json:"family,omitempty"`
	URL    string `json:"url"`
	// SHA256 is the lowercase hex digest of the downloaded bytes (the
	// archive for archive kinds, the file itself for ArchiveNone).
	SHA256  string      `json:"sha256,omitempty"`
	Archive ArchiveKind `json:"archive,omitempty"`
	// Files lists files expected under the model directory (relative,
	// slash-separated). For ArchiveNone it may hold the single file name;
	// when empty the base name of the URL is used.
	Files []File `json:"files,omitempty"`
}

// Manifest is the JSON document listing models.
type Manifest struct {
	Version int     `json:"version"`
	Models  []Entry `json:"models"`
}

// ParseManifest decodes and validates a JSON manifest. Unknown fields are
// rejected so typos do not silently disable a digest.
func ParseManifest(r io.Reader) (*Manifest, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var m Manifest
	if err := dec.Decode(&m); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidManifest, err)
	}
	if err := m.Validate(); err != nil {
		return nil, err
	}
	return &m, nil
}

// LoadManifest reads and validates the manifest file at path.
func LoadManifest(path string) (*Manifest, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseManifest(f)
}

// Validate checks the manifest structure. An empty sha256 is valid here: it is
// refused at Fetch time unless the caller opts in.
func (m *Manifest) Validate() error {
	if m.Version != ManifestVersion {
		return fmt.Errorf("%w: unsupported version %d (want %d)", ErrInvalidManifest, m.Version, ManifestVersion)
	}
	seen := make(map[string]struct{}, len(m.Models))
	for i := range m.Models {
		e := &m.Models[i]
		if err := e.Validate(); err != nil {
			return err
		}
		if _, dup := seen[e.Name]; dup {
			return fmt.Errorf("%w: duplicate model name %q", ErrInvalidManifest, e.Name)
		}
		seen[e.Name] = struct{}{}
	}
	return nil
}

// Lookup returns the entry with the given name.
func (m *Manifest) Lookup(name string) (Entry, error) {
	for _, e := range m.Models {
		if e.Name == name {
			return e, nil
		}
	}
	return Entry{}, fmt.Errorf("%w: %q", ErrNotFound, name)
}

// Validate checks a single entry.
func (e *Entry) Validate() error {
	if e.Name == "" || !filepath.IsLocal(e.Name) || strings.ContainsAny(e.Name, `/\`) {
		return fmt.Errorf("%w: invalid model name %q", ErrInvalidManifest, e.Name)
	}
	u, err := url.Parse(e.URL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return fmt.Errorf("%w: model %q: url must be absolute http(s)", ErrInvalidManifest, e.Name)
	}
	switch e.Kind() {
	case ArchiveNone, ArchiveTarGz, ArchiveTarBz2:
	default:
		return fmt.Errorf("%w: model %q: unknown archive kind %q", ErrInvalidManifest, e.Name, e.Archive)
	}
	if e.SHA256 != "" && !validDigest(e.SHA256) {
		return fmt.Errorf("%w: model %q: sha256 must be 64 lowercase hex characters", ErrInvalidManifest, e.Name)
	}
	for _, f := range e.Files {
		if f.Path == "" || !filepath.IsLocal(filepath.FromSlash(f.Path)) {
			return fmt.Errorf("%w: model %q: invalid file path %q", ErrInvalidManifest, e.Name, f.Path)
		}
		if f.SHA256 != "" && !validDigest(f.SHA256) {
			return fmt.Errorf("%w: model %q: file %q: sha256 must be 64 lowercase hex characters", ErrInvalidManifest, e.Name, f.Path)
		}
	}
	if e.Kind() == ArchiveNone && len(e.Files) > 1 {
		return fmt.Errorf("%w: model %q: a single-file entry lists %d files", ErrInvalidManifest, e.Name, len(e.Files))
	}
	return nil
}

// Kind returns the archive kind, treating the empty value as ArchiveNone.
func (e *Entry) Kind() ArchiveKind {
	if e.Archive == "" {
		return ArchiveNone
	}
	return e.Archive
}

// singleFileName is the file name of an ArchiveNone entry.
func (e *Entry) singleFileName() string {
	if len(e.Files) == 1 {
		return e.Files[0].Path
	}
	if u, err := url.Parse(e.URL); err == nil {
		if b := path.Base(u.Path); b != "" && b != "/" && b != "." {
			return b
		}
	}
	return e.Name
}

// Target is where Fetch places the entry inside dir: the file itself for
// ArchiveNone, otherwise a directory named after the entry.
func (e *Entry) Target(dir string) string {
	if e.Kind() == ArchiveNone {
		return filepath.Join(dir, filepath.FromSlash(e.singleFileName()))
	}
	return filepath.Join(dir, e.Name)
}

// Verify checks an already fetched entry under dir (the same dir given to
// Fetch): every expected file must exist as a regular file and match its
// digest when one is set. For ArchiveNone the file is also checked against the
// entry's own sha256 when pinned. Paths are resolved through os.Root, so a
// symlink cannot lead the check outside the model directory.
func (e *Entry) Verify(dir string) error {
	if e.Kind() == ArchiveNone {
		return e.verifySingle(dir)
	}
	root, err := os.OpenRoot(e.Target(dir))
	if err != nil {
		return err
	}
	defer root.Close()
	for _, f := range e.Files {
		if err := verifyInRoot(root, f); err != nil {
			return err
		}
	}
	return nil
}

func (e *Entry) verifySingle(dir string) error {
	name := e.singleFileName()
	root, err := os.OpenRoot(dir)
	if err != nil {
		return err
	}
	defer root.Close()
	f := File{Path: name, SHA256: e.SHA256}
	if len(e.Files) == 1 && e.Files[0].SHA256 != "" {
		f.SHA256 = e.Files[0].SHA256
	}
	return verifyInRoot(root, f)
}

func verifyInRoot(root *os.Root, f File) error {
	fh, err := root.Open(filepath.FromSlash(f.Path))
	if err != nil {
		return fmt.Errorf("modelstore: expected file %q: %w", f.Path, err)
	}
	defer fh.Close()
	st, err := fh.Stat()
	if err != nil {
		return err
	}
	if !st.Mode().IsRegular() {
		return fmt.Errorf("modelstore: expected file %q is not a regular file", f.Path)
	}
	if f.SHA256 == "" {
		return nil
	}
	got, err := digestReader(fh)
	if err != nil {
		return err
	}
	if got != f.SHA256 {
		return &DigestError{What: f.Path, Want: f.SHA256, Got: got}
	}
	return nil
}

// VerifyFile checks that the file at path has the given SHA-256 digest.
func VerifyFile(path, want string) error {
	if !validDigest(want) {
		return fmt.Errorf("modelstore: invalid sha256 %q", want)
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	got, err := digestReader(f)
	if err != nil {
		return err
	}
	if got != want {
		return &DigestError{What: path, Want: want, Got: got}
	}
	return nil
}

// DigestFile returns the lowercase hex SHA-256 of the file at path, for
// pinning an entry.
func DigestFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	return digestReader(f)
}

func digestReader(r io.Reader) (string, error) {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func validDigest(s string) bool {
	if len(s) != sha256.Size*2 {
		return false
	}
	for _, c := range s {
		if (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}
