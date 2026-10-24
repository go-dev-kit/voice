package modelstore

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// bz2Fixture is a tar.bz2 of m/a.txt ("hello model") and m/sub/b.txt
// ("nested"); the standard library has a bzip2 reader but no writer.
const bz2Fixture = "QlpoOTFBWSZTWZHulUMAAOd7kNOAQIBAAf+AAjF3b59ABAAQCDAA+AMaNGgGTIaMRpoGmBjRo0AyZDRiNNA0wCqmpNk0jSDQaaY1PUaYTanU4HRg17EazP8xsk0gT9vetQ6EyhJBKB4FCC5K5asPNtpa6ZZRzRDWvwCVw+CDHHLK0oIKlTItJb4aJIWnlQ2H0rFuRVUtSg9Fl6yrQVlZoMaJzKkys18Gkb+BvILOzlQ3Tp+9fs4HfscTAwcuVGdT5/jmTu0yUJnU1Gr1Zs5HdC8zNptbiX+LuSKcKEhI90qhgA=="

type member struct {
	name, body, link string
	typ              byte
}

func sum(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func tarGz(t *testing.T, members ...member) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(zw)
	for _, m := range members {
		typ := m.typ
		if typ == 0 {
			typ = tar.TypeReg
		}
		h := &tar.Header{Name: m.name, Typeflag: typ, Mode: 0o644, Linkname: m.link}
		if typ == tar.TypeReg {
			h.Size = int64(len(m.body))
		}
		if typ == tar.TypeDir {
			h.Mode = 0o755
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if typ == tar.TypeReg {
			if _, err := tw.Write([]byte(m.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func serve(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func noTemp(t *testing.T, dir string) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range ents {
		if strings.HasPrefix(e.Name(), ".modelstore-") {
			t.Errorf("leftover temp entry %q", e.Name())
		}
	}
}

func TestFetchSingleFile(t *testing.T) {
	body := []byte("onnx bytes")
	srv := serve(t, body)
	dir := t.TempDir()
	e := Entry{Name: "vad", URL: srv.URL + "/silero_vad.onnx", SHA256: sum(body)}
	res, err := Fetch(context.Background(), srv.Client(), e, dir)
	if err != nil {
		t.Fatal(err)
	}
	if !res.Pinned || res.SHA256 != sum(body) || res.Path != filepath.Join(dir, "silero_vad.onnx") {
		t.Fatalf("result %+v", res)
	}
	got, _ := os.ReadFile(res.Path)
	if !bytes.Equal(got, body) {
		t.Fatal("content mismatch")
	}
	if err := e.Verify(dir); err != nil {
		t.Fatal(err)
	}
	noTemp(t, dir)
}

func TestFetchDigestMismatch(t *testing.T) {
	srv := serve(t, []byte("actual"))
	dir := t.TempDir()
	e := Entry{Name: "vad", URL: srv.URL + "/m.onnx", SHA256: sum([]byte("other"))}
	_, err := Fetch(context.Background(), srv.Client(), e, dir)
	var de *DigestError
	if !errors.As(err, &de) || de.Got != sum([]byte("actual")) {
		t.Fatalf("want DigestError, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "m.onnx")); !os.IsNotExist(err) {
		t.Fatalf("mismatched file was published: %v", err)
	}
	noTemp(t, dir)
}

func TestFetchUnpinned(t *testing.T) {
	var hits atomic.Int32
	body := []byte("data")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.bin"}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); !errors.Is(err, ErrUnpinned) {
		t.Fatalf("want ErrUnpinned, got %v", err)
	}
	if hits.Load() != 0 {
		t.Fatal("unpinned entry hit the network without opt-in")
	}
	res, err := Fetch(context.Background(), srv.Client(), e, dir, WithAllowUnpinned())
	if err != nil {
		t.Fatal(err)
	}
	if res.Pinned || res.SHA256 != sum(body) {
		t.Fatalf("result %+v", res)
	}
}

func TestFetchTarGz(t *testing.T) {
	arc := tarGz(t,
		member{name: "root/", typ: tar.TypeDir},
		member{name: "root/model.onnx", body: "weights"},
		member{name: "root/tokens.txt", body: "a b"},
		member{name: "root/link.txt", typ: tar.TypeSymlink, link: "tokens.txt"},
	)
	srv := serve(t, arc)
	dir := t.TempDir()
	e := Entry{
		Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz,
		Files: []File{{Path: "model.onnx", SHA256: sum([]byte("weights"))}, {Path: "tokens.txt"}},
	}
	res, err := Fetch(context.Background(), srv.Client(), e, dir)
	if err != nil {
		t.Fatal(err)
	}
	if res.Path != filepath.Join(dir, "m") {
		t.Fatalf("path %q", res.Path)
	}
	if b, _ := os.ReadFile(filepath.Join(res.Path, "model.onnx")); string(b) != "weights" {
		t.Fatal("stripped root not promoted")
	}
	if err := e.Verify(dir); err != nil {
		t.Fatal(err)
	}
	// Refetch replaces the existing directory.
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err != nil {
		t.Fatal(err)
	}
	noTemp(t, dir)
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("dir has %d entries after refetch", len(entries))
	}
}

func TestFetchTarBz2(t *testing.T) {
	arc, err := base64.StdEncoding.DecodeString(bz2Fixture)
	if err != nil {
		t.Fatal(err)
	}
	srv := serve(t, arc)
	dir := t.TempDir()
	e := Entry{
		Name: "m", URL: srv.URL + "/m.tar.bz2", SHA256: sum(arc), Archive: ArchiveTarBz2,
		Files: []File{{Path: "a.txt", SHA256: sum([]byte("hello model"))}, {Path: "sub/b.txt", SHA256: sum([]byte("nested"))}},
	}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err != nil {
		t.Fatal(err)
	}
	if err := e.Verify(dir); err != nil {
		t.Fatal(err)
	}
}

func TestFetchMissingExpectedFile(t *testing.T) {
	arc := tarGz(t, member{name: "root/a.txt", body: "a"})
	srv := serve(t, arc)
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz, Files: []File{{Path: "b.txt"}}}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err == nil {
		t.Fatal("expected missing-file error")
	}
	if _, err := os.Stat(filepath.Join(dir, "m")); !os.IsNotExist(err) {
		t.Fatalf("incomplete model published: %v", err)
	}
	noTemp(t, dir)
}

func TestFetchUnsafeArchive(t *testing.T) {
	cases := map[string][]member{
		"parent traversal": {{name: "../evil", body: "x"}},
		"nested traversal": {{name: "a/../../evil", body: "x"}},
		"absolute":         {{name: "/etc/evil", body: "x"}},
		"symlink absolute": {{name: "l", typ: tar.TypeSymlink, link: "/etc/passwd"}},
		"symlink escape":   {{name: "d/l", typ: tar.TypeSymlink, link: "../../x"}},
		"hardlink":         {{name: "h", typ: tar.TypeLink, link: "a"}},
		"symlink in symlinked parent": {
			{name: "a", typ: tar.TypeSymlink, link: "."},
			{name: "a/b", typ: tar.TypeSymlink, link: "../x"},
		},
		"symlink dotdot after element": {
			{name: "a", typ: tar.TypeSymlink, link: "."},
			{name: "l", typ: tar.TypeSymlink, link: "a/../x"},
		},
		"promoted link escapes final root": {
			{name: "top/real.txt", body: "r"},
			{name: "top/sub/l", typ: tar.TypeSymlink, link: "../../secret"},
		},
		"regular member over link to existing dir": {
			{name: "sub/", typ: tar.TypeDir},
			{name: "l", typ: tar.TypeSymlink, link: "sub"},
			{name: "l", body: "x"},
		},
	}
	for name, members := range cases {
		t.Run(name, func(t *testing.T) {
			arc := tarGz(t, members...)
			srv := serve(t, arc)
			parent := t.TempDir()
			dir := filepath.Join(parent, "store")
			e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz}
			if _, err := Fetch(context.Background(), srv.Client(), e, dir); err == nil {
				t.Fatal("expected error")
			} else if name != "regular member over link to existing dir" && !errors.Is(err, ErrUnsafeArchive) {
				t.Fatalf("want ErrUnsafeArchive, got %v", err)
			}
			if _, err := os.Stat(filepath.Join(parent, "evil")); !os.IsNotExist(err) {
				t.Fatal("file escaped the target directory")
			}
			noTemp(t, dir)
			if _, err := os.Stat(filepath.Join(dir, "m")); !os.IsNotExist(err) {
				t.Fatal("unsafe archive was published")
			}
		})
	}
}

func TestFetchSizeLimits(t *testing.T) {
	big := tarGz(t, member{name: "r/big", body: strings.Repeat("A", 4096)})
	srv := serve(t, big)
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(big), Archive: ArchiveTarGz}

	if _, err := Fetch(context.Background(), srv.Client(), e, t.TempDir(), WithMaxExtractBytes(1024)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("extract limit: %v", err)
	}
	dir := t.TempDir()
	if _, err := Fetch(context.Background(), srv.Client(), e, dir, WithMaxDownloadBytes(10)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("download limit: %v", err)
	}
	noTemp(t, dir)

	many := tarGz(t, member{name: "r/1", body: "a"}, member{name: "r/2", body: "a"}, member{name: "r/3", body: "a"})
	srv2 := serve(t, many)
	e2 := Entry{Name: "m", URL: srv2.URL + "/m.tar.gz", SHA256: sum(many), Archive: ArchiveTarGz}
	if _, err := Fetch(context.Background(), srv2.Client(), e2, t.TempDir(), WithMaxFiles(2)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("member limit: %v", err)
	}
}

func TestFetchChunkedOverLimit(t *testing.T) {
	// No Content-Length: the limit must hold on the streamed body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fl := w.(http.Flusher)
		for range 8 {
			_, _ = w.Write(bytes.Repeat([]byte("x"), 1024))
			fl.Flush()
		}
	}))
	defer srv.Close()
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.bin", SHA256: sum(nil)}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir, WithMaxDownloadBytes(2048)); !errors.Is(err, ErrTooLarge) {
		t.Fatalf("got %v", err)
	}
	noTemp(t, dir)
}

func TestFetchCancelledMidDownload(t *testing.T) {
	started := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000000")
		_, _ = w.Write(bytes.Repeat([]byte("x"), 1024))
		w.(http.Flusher).Flush()
		close(started)
		<-r.Context().Done()
	}))
	defer srv.Close()
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-started
		cancel()
	}()
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.bin", SHA256: sum([]byte("whatever"))}
	done := make(chan error, 1)
	go func() {
		_, err := Fetch(ctx, srv.Client(), e, dir)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("want context.Canceled, got %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Fetch did not honor cancellation")
	}
	noTemp(t, dir)
	if _, err := os.Stat(filepath.Join(dir, "m.bin")); !os.IsNotExist(err) {
		t.Fatal("partial download published")
	}
}

func TestFetchTruncatedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Length", "100")
		_, _ = w.Write([]byte("short"))
	}))
	defer srv.Close()
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.bin", SHA256: sum([]byte("short"))}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err == nil {
		t.Fatal("expected truncated-body error")
	}
	noTemp(t, dir)
	if _, err := os.Stat(filepath.Join(dir, "m.bin")); !os.IsNotExist(err) {
		t.Fatal("truncated download published")
	}
}

func TestFetchHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler())
	defer srv.Close()
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.bin", SHA256: sum(nil)}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err == nil {
		t.Fatal("expected status error")
	}
	noTemp(t, dir)
}

func TestVerifyFileAndDigest(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f")
	if err := os.WriteFile(p, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	d, err := DigestFile(p)
	if err != nil || d != sum([]byte("abc")) {
		t.Fatalf("digest %q %v", d, err)
	}
	if err := VerifyFile(p, d); err != nil {
		t.Fatal(err)
	}
	var de *DigestError
	if err := VerifyFile(p, sum([]byte("x"))); !errors.As(err, &de) {
		t.Fatalf("want DigestError, got %v", err)
	}
	if err := VerifyFile(p, "nothex"); err == nil {
		t.Fatal("invalid digest accepted")
	}
}

func TestVerifyDirDetectsTamper(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "m")
	if err := os.MkdirAll(model, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(model, "w.onnx"), []byte("good"), 0o644); err != nil {
		t.Fatal(err)
	}
	e := Entry{Name: "m", URL: "https://example.test/m.tar.gz", Archive: ArchiveTarGz, Files: []File{{Path: "w.onnx", SHA256: sum([]byte("good"))}}}
	if err := e.Verify(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(model, "w.onnx"), []byte("bad"), 0o644); err != nil {
		t.Fatal(err)
	}
	var de *DigestError
	if err := e.Verify(dir); !errors.As(err, &de) {
		t.Fatalf("want DigestError, got %v", err)
	}
	// A symlink pointing outside the model directory must not be followed.
	outside := filepath.Join(dir, "outside")
	if err := os.WriteFile(outside, []byte("good"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(model, "w.onnx")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(model, "w.onnx")); err != nil {
		t.Fatal(err)
	}
	if err := e.Verify(dir); err == nil {
		t.Fatal("verify followed a symlink out of the model directory")
	}
}

func TestParseManifest(t *testing.T) {
	good := `{"version":1,"models":[{"name":"silero","family":"vad","url":"https://example.test/silero_vad.onnx","sha256":"` + sum(nil) + `"},
		{"name":"par","url":"https://example.test/p.tar.bz2","archive":"tar.bz2","files":[{"path":"tokens.txt"}]}]}`
	m, err := ParseManifest(strings.NewReader(good))
	if err != nil {
		t.Fatal(err)
	}
	if e, err := m.Lookup("par"); err != nil || e.Kind() != ArchiveTarBz2 {
		t.Fatalf("lookup %+v %v", e, err)
	}
	if _, err := m.Lookup("nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("got %v", err)
	}
	bad := map[string]string{
		"version":       `{"version":2,"models":[]}`,
		"unknown field": `{"version":1,"models":[{"name":"a","url":"https://x.test/a","sha":"x"}]}`,
		"bad url":       `{"version":1,"models":[{"name":"a","url":"file:///etc/passwd"}]}`,
		"bad name":      `{"version":1,"models":[{"name":"../a","url":"https://x.test/a"}]}`,
		"bad digest":    `{"version":1,"models":[{"name":"a","url":"https://x.test/a","sha256":"ABC"}]}`,
		"bad archive":   `{"version":1,"models":[{"name":"a","url":"https://x.test/a","archive":"zip"}]}`,
		"bad file path": `{"version":1,"models":[{"name":"a","url":"https://x.test/a","archive":"tar.gz","files":[{"path":"../x"}]}]}`,
		"duplicate":     `{"version":1,"models":[{"name":"a","url":"https://x.test/a"},{"name":"a","url":"https://x.test/b"}]}`,
	}
	for name, doc := range bad {
		if _, err := ParseManifest(strings.NewReader(doc)); !errors.Is(err, ErrInvalidManifest) {
			t.Errorf("%s: got %v", name, err)
		}
	}
}

func TestFetchPromotedLinkStaysInsideModel(t *testing.T) {
	// Exactly the review archive: valid against the stage root, but after the
	// single top-level directory is promoted, sub/l would read <dir>/secret.
	arc := tarGz(t,
		member{name: "top/real.txt", body: "r"},
		member{name: "top/sub/l", typ: tar.TypeSymlink, link: "../../secret"},
	)
	srv := serve(t, arc)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "secret"), []byte("s"), 0o644); err != nil {
		t.Fatal(err)
	}
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz}
	_, err := Fetch(context.Background(), srv.Client(), e, dir)
	if !errors.Is(err, ErrUnsafeArchive) {
		t.Fatalf("want ErrUnsafeArchive, got %v", err)
	}
	if _, err := os.Lstat(filepath.Join(dir, "m")); !os.IsNotExist(err) {
		t.Fatalf("model with escaping link was installed: %v", err)
	}
	noTemp(t, dir)

	// A link that stays inside the promoted directory is kept.
	ok := tarGz(t,
		member{name: "top/real.txt", body: "r"},
		member{name: "top/sub/l", typ: tar.TypeSymlink, link: "../real.txt"},
	)
	srv2 := serve(t, ok)
	e2 := Entry{Name: "m", URL: srv2.URL + "/m.tar.gz", SHA256: sum(ok), Archive: ArchiveTarGz}
	if _, err := Fetch(context.Background(), srv2.Client(), e2, dir); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(filepath.Join(dir, "m", "sub", "l")); err != nil || string(b) != "r" {
		t.Fatalf("inside link: %q %v", b, err)
	}
}

func TestEntryValidateNames(t *testing.T) {
	const u = "https://x.test/"
	bad := map[string]Entry{
		"dot name":        {Name: ".", URL: u + "a", Archive: ArchiveTarGz},
		"dotdot name":     {Name: "..", URL: u + "a"},
		"unclean name":    {Name: "a.", URL: u + "a"}, // path.Clean keeps this, so it is a control below
		"reserved prefix": {Name: ".modelstore-x", URL: u + "a", Archive: ArchiveTarGz},
		"dotdot url base": {Name: "a", URL: u + "..", SHA256: sum(nil)},
		"encoded dotdot":  {Name: "a", URL: u + "%2e%2e"},
		"subdir file":     {Name: "a", URL: u + "a", Files: []File{{Path: "sub/x.onnx"}}},
		"reserved file":   {Name: "a", URL: u + ".modelstore-dl-1"},
		"dot file":        {Name: "a", URL: u + "x", Files: []File{{Path: "./x"}}},
	}
	delete(bad, "unclean name")
	for name, e := range bad {
		if err := e.Validate(); !errors.Is(err, ErrInvalidManifest) {
			t.Errorf("%s: got %v", name, err)
		}
	}
	// A bare host path falls back to the entry name.
	e := Entry{Name: "a", URL: u}
	if err := e.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := e.Target("d"); got != filepath.Join("d", "a") {
		t.Fatalf("target %q", got)
	}
}

func TestManifestDuplicateTarget(t *testing.T) {
	m := Manifest{Version: ManifestVersion, Models: []Entry{
		{Name: "a", URL: "https://x.test/f.onnx", Files: []File{{Path: "same.onnx"}}},
		{Name: "b", URL: "https://x.test/g.onnx", Files: []File{{Path: "same.onnx"}}},
	}}
	if err := m.Validate(); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("same single-file target: %v", err)
	}
	m = Manifest{Version: ManifestVersion, Models: []Entry{
		{Name: "a", URL: "https://x.test/a.tar.gz", Archive: ArchiveTarGz},
		{Name: "x", URL: "https://x.test/a", Files: []File{{Path: "a"}}},
	}}
	if err := m.Validate(); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("archive vs file target: %v", err)
	}
}

func TestFetchRefusesEscapingTarget(t *testing.T) {
	// Bypass nothing: Fetch validates, so a "%2e%2e" base never reaches the
	// network and the parent of dir is untouched.
	parent := t.TempDir()
	dir := filepath.Join(parent, "store")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		_, _ = w.Write([]byte("x"))
	}))
	defer srv.Close()
	e := Entry{Name: "a", URL: srv.URL + "/%2e%2e", SHA256: sum([]byte("x"))}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("got %v", err)
	}
	if hits.Load() != 0 {
		t.Fatal("request made for invalid entry")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("store dir disturbed: %v", err)
	}
}

func TestReplaceStaysInsideDir(t *testing.T) {
	parent := t.TempDir()
	dir := filepath.Join(parent, "store")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	src := filepath.Join(dir, ".modelstore-dl-1")
	if err := os.WriteFile(src, []byte("n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, dst := range []string{dir, parent, filepath.Join(dir, ".."), filepath.Join(parent, "other")} {
		if err := replace(dir, src, dst); err == nil {
			t.Errorf("replace onto %q succeeded", dst)
		}
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("dir disturbed: %v", err)
	}
	if _, err := os.Stat(src); err != nil {
		t.Fatalf("src moved: %v", err)
	}
}

func TestFetchReplaceRobustness(t *testing.T) {
	arc := tarGz(t, member{name: "r/a.txt", body: "new"})
	srv := serve(t, arc)
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz, Files: []File{{Path: "a.txt"}}}

	// Crash leftovers, including a non-empty one at the old deterministic
	// name, must not break later fetches.
	stale := filepath.Join(dir, "m.old-x")
	if err := os.MkdirAll(filepath.Join(stale, "deep"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(stale, "deep", "f"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	staleHolder := filepath.Join(dir, ".modelstore-old-leftover")
	if err := os.MkdirAll(filepath.Join(staleHolder, "old", "f"), 0o755); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		if _, err := Fetch(context.Background(), srv.Client(), e, dir); err != nil {
			t.Fatal(err)
		}
	}
	// Concurrent fetches into one dir all succeed and leave a valid model.
	var wg sync.WaitGroup
	errs := make(chan error, 6)
	for range 6 {
		wg.Go(func() {
			_, err := Fetch(context.Background(), srv.Client(), e, dir)
			errs <- err
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent fetch: %v", err)
		}
	}
	if err := e.Verify(dir); err != nil {
		t.Fatal(err)
	}
}

func TestFetchCleanupFailureAfterSwapIsNotAnError(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	arc := tarGz(t, member{name: "r/a.txt", body: "new"})
	srv := serve(t, arc)
	dir := t.TempDir()
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err != nil {
		t.Fatal(err)
	}
	// A read-only subdirectory with content makes RemoveAll of the displaced
	// copy fail after the swap succeeded.
	m := filepath.Join(dir, "m")
	locked := filepath.Join(m, "locked")
	if err := os.MkdirAll(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(locked, "f"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(locked, 0o555); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = filepath.WalkDir(dir, func(p string, d os.DirEntry, _ error) error {
			if d != nil && d.IsDir() {
				_ = os.Chmod(p, 0o755)
			}
			return nil
		})
	})
	res, err := Fetch(context.Background(), srv.Client(), e, dir)
	if err != nil {
		t.Fatalf("installed model reported as failure: %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(res.Path, "a.txt")); string(b) != "new" {
		t.Fatal("new model not installed")
	}
}

// swapTransport replaces the temporary download file with evil right before
// the body is exhausted, i.e. between the digest being computed and any
// reopening of the file by name.
type swapTransport struct {
	base http.RoundTripper
	dir  string
	evil []byte
}

type swapBody struct {
	io.ReadCloser
	once sync.Once
	swap func()
}

func (b *swapBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil {
		b.once.Do(b.swap)
	}
	return n, err
}

func (s swapTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	resp, err := s.base.RoundTrip(r)
	if err != nil {
		return nil, err
	}
	resp.Body = &swapBody{ReadCloser: resp.Body, swap: func() {
		m, _ := filepath.Glob(filepath.Join(s.dir, ".modelstore-dl-*"))
		if len(m) != 1 {
			return
		}
		repl := filepath.Join(s.dir, "evil-swap")
		if os.WriteFile(repl, s.evil, 0o644) == nil {
			_ = os.Rename(repl, m[0])
		}
	}}
	return resp, nil
}

func TestFetchExtractsVerifiedBytes(t *testing.T) {
	good := tarGz(t, member{name: "r/a.txt", body: "good"})
	evil := tarGz(t, member{name: "r/a.txt", body: "evil"})
	srv := serve(t, good)
	dir := t.TempDir()
	client := &http.Client{Transport: swapTransport{base: srv.Client().Transport, dir: dir, evil: evil}}
	if client.Transport.(swapTransport).base == nil {
		client.Transport = swapTransport{base: http.DefaultTransport, dir: dir, evil: evil}
	}
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(good), Archive: ArchiveTarGz, Files: []File{{Path: "a.txt"}}}
	if _, err := Fetch(context.Background(), client, e, dir); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "m", "a.txt")); string(b) != "good" {
		t.Fatalf("extracted %q, not the verified archive", b)
	}
}

func TestFetchWriteViaLinkStaysInsideRoot(t *testing.T) {
	// l is a link to the existing directory sub; the member l/f is written
	// through it. An escaping link is refused up front (see
	// TestFetchUnsafeArchive), so the only reachable outcome is a write that
	// lands inside the model, never beside it.
	arc := tarGz(t,
		member{name: "top/sub/", typ: tar.TypeDir},
		member{name: "top/l", typ: tar.TypeSymlink, link: "sub"},
		member{name: "top/l/f", body: "x"},
	)
	srv := serve(t, arc)
	parent := t.TempDir()
	dir := filepath.Join(parent, "store")
	e := Entry{Name: "m", URL: srv.URL + "/m.tar.gz", SHA256: sum(arc), Archive: ArchiveTarGz}
	if _, err := Fetch(context.Background(), srv.Client(), e, dir); err != nil {
		t.Fatal(err)
	}
	if b, err := os.ReadFile(filepath.Join(dir, "m", "sub", "f")); err != nil || string(b) != "x" {
		t.Fatalf("sub/f: %q %v", b, err)
	}
	for _, p := range []string{filepath.Join(dir, "f"), filepath.Join(parent, "f")} {
		if _, err := os.Lstat(p); !os.IsNotExist(err) {
			t.Fatalf("%s written outside the model: %v", p, err)
		}
	}
}
