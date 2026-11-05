package modelstore

import (
	"crypto/tls"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// hardenClient is the client tests use against httptest TLS servers: the
// production redirect policy on top of a transport that trusts the test
// certificate.
func hardenClient() *http.Client {
	c := defaultHTTPClient()
	c.Transport = &http.Transport{TLSClientConfig: &tls.Config{InsecureSkipVerify: true}} // #nosec G402 -- test fixture
	return c
}

func TestFetchSingleFileRollsBackFailedRecheck(t *testing.T) {
	verified := []byte("verified bytes\n")
	rejected := []byte("swapped bytes\n")
	want := sum(verified)
	srv := serve(t, verified)
	defer srv.Close()
	entry := Entry{Name: "plain", URL: srv.URL, SHA256: want}
	client := func(dir string) *http.Client {
		base := srv.Client().Transport
		if base == nil {
			base = http.DefaultTransport
		}
		return &http.Client{Transport: swapTransport{base: base, dir: dir, evil: rejected}}
	}

	t.Run("restores previous install", func(t *testing.T) {
		dir := t.TempDir()
		previous := []byte("previous verified model\n")
		target := entry.Target(dir)
		if err := os.WriteFile(target, previous, 0o644); err != nil {
			t.Fatal(err)
		}

		_, err := Fetch(t.Context(), client(dir), entry, dir)
		var digestErr *DigestError
		if !errors.As(err, &digestErr) {
			t.Fatalf("swapped install: err = %v, want DigestError", err)
		}
		got, err := os.ReadFile(target)
		if err != nil {
			t.Fatalf("read restored target: %v", err)
		}
		if string(got) != string(previous) {
			t.Fatalf("target = %q, want previous bytes %q", got, previous)
		}
		noTemp(t, dir)
	})

	t.Run("removes rejected first install", func(t *testing.T) {
		dir := t.TempDir()
		target := entry.Target(dir)
		_, err := Fetch(t.Context(), client(dir), entry, dir)
		var digestErr *DigestError
		if !errors.As(err, &digestErr) {
			t.Fatalf("swapped install: err = %v, want DigestError", err)
		}
		if _, err := os.Stat(target); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("target stat = %v, want rejected first install removed", err)
		}
		noTemp(t, dir)
	})

	t.Run("restores previous install after read error", func(t *testing.T) {
		dir := t.TempDir()
		previous := []byte("previous verified model\n")
		target := entry.Target(dir)
		if err := os.WriteFile(target, previous, 0o644); err != nil {
			t.Fatal(err)
		}
		readErr := errors.New("recheck failed")
		installDigest = func(string) (string, error) { return "", readErr }
		defer func() { installDigest = DigestFile }()

		_, err := Fetch(t.Context(), srv.Client(), entry, dir)
		if !errors.Is(err, readErr) {
			t.Fatalf("Fetch error = %v, want %v", err, readErr)
		}
		got, err := os.ReadFile(target)
		if err != nil {
			t.Fatalf("read restored target: %v", err)
		}
		if string(got) != string(previous) {
			t.Fatalf("target = %q, want previous bytes %q", got, previous)
		}
		noTemp(t, dir)
	})
}

func TestFetchFollowsHTTPSRedirectsOnly(t *testing.T) {
	body := []byte("redirected\n")
	want := sum(body)
	inner := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer inner.Close()

	t.Run("https to https succeeds", func(t *testing.T) {
		outer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, inner.URL+"/file", http.StatusFound)
		}))
		defer outer.Close()
		res, err := Fetch(t.Context(), hardenClient(), Entry{Name: "m", URL: outer.URL, SHA256: want}, t.TempDir())
		if err != nil {
			t.Fatalf("Fetch: %v", err)
		}
		if res.SHA256 != want {
			t.Fatalf("digest = %s", res.SHA256)
		}
	})

	t.Run("redirect to http is refused", func(t *testing.T) {
		plain := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write(body)
		}))
		defer plain.Close()
		outer := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, plain.URL+"/file", http.StatusFound)
		}))
		defer outer.Close()
		_, err := Fetch(t.Context(), hardenClient(), Entry{Name: "m", URL: outer.URL, SHA256: want}, t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "https required") {
			t.Fatalf("err = %v, want https-required refusal", err)
		}
	})

	t.Run("redirect loop is bounded", func(t *testing.T) {
		var outer *httptest.Server
		outer = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Redirect(w, r, outer.URL+"/next", http.StatusFound)
		}))
		defer outer.Close()
		_, err := Fetch(t.Context(), hardenClient(), Entry{Name: "m", URL: outer.URL, SHA256: want}, t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "redirects") {
			t.Fatalf("err = %v, want redirect-bound failure", err)
		}
	})
}

func TestParseManifestRejectsTrailingDocuments(t *testing.T) {
	good := `{"version":1,"models":[{"name":"a","url":"https://x/a"}]}`
	if _, err := ParseManifest(strings.NewReader(good)); err != nil {
		t.Fatalf("single document: %v", err)
	}
	trailing := good + "\n" + `{"version":1,"models":[]}`
	if _, err := ParseManifest(strings.NewReader(trailing)); err == nil || !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest for trailing document", err)
	}
	// Whitespace after the document is not a second document.
	if _, err := ParseManifest(strings.NewReader(good + "\n  \n")); err != nil {
		t.Fatalf("trailing whitespace: %v", err)
	}
}

func TestFetchArchiveDigestMismatch(t *testing.T) {
	archive := tarGz(t, member{name: "m/a.txt", body: "x"})
	wrong := []byte(sum(archive))
	wrong[0] = '0'
	srv := serve(t, archive)
	defer srv.Close()
	dir := t.TempDir()
	_, err := Fetch(t.Context(), nil, Entry{Name: "m", URL: srv.URL, SHA256: string(wrong), Archive: ArchiveTarGz}, dir)
	var de *DigestError
	if !errors.As(err, &de) {
		t.Fatalf("err = %v, want DigestError", err)
	}
	if de.What != srv.URL {
		t.Fatalf("DigestError.What = %q, want the archive URL", de.What)
	}
	// Nothing was installed.
	if _, err := os.Stat(filepath.Join(dir, "m")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("target stat = %v, want not-exist", err)
	}
	noTemp(t, dir)
}

func TestSweepRemovesOnlyTempEntries(t *testing.T) {
	dir := t.TempDir()
	keep := filepath.Join(dir, "modeldir")
	if err := os.MkdirAll(keep, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(keep, "weights.bin"), []byte("k"), 0o644); err != nil {
		t.Fatal(err)
	}
	stale := []string{
		filepath.Join(dir, tempPrefix+"dl-123"),
		filepath.Join(dir, tempPrefix+"x-456", "inner.bin"),
		filepath.Join(dir, tempPrefix+"old-789", "old"),
	}
	for _, p := range stale {
		if err := os.MkdirAll(filepath.Dir(p), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("stale"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := Sweep(dir); err != nil {
		t.Fatalf("Sweep: %v", err)
	}
	for _, p := range stale {
		if _, err := os.Stat(p); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("%s stat = %v, want removed", p, err)
		}
	}
	if _, err := os.Stat(filepath.Join(keep, "weights.bin")); err != nil {
		t.Fatalf("Sweep removed a non-temp model: %v", err)
	}
	if err := Sweep(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("Sweep of a missing dir should fail")
	}
}

// TestDefaultHTTPClientPolicy pins the client Fetch installs for a nil
// argument: shares the default transport (proxy and TLS environment keep
// applying), bounds redirects, and refuses non-https redirect targets.
func TestDefaultHTTPClientPolicy(t *testing.T) {
	c := defaultHTTPClient()
	if c.Transport != http.DefaultTransport {
		t.Fatal("default client must keep the shared transport")
	}
	if err := c.CheckRedirect(httptest.NewRequest(http.MethodGet, "https://example.com/f", nil), make([]*http.Request, 5)); err == nil {
		t.Fatal("a sixth redirect must be refused")
	}
	toHTTP := httptest.NewRequest(http.MethodGet, "http://example.com/f", nil)
	if err := c.CheckRedirect(toHTTP, make([]*http.Request, 1)); err == nil || !strings.Contains(err.Error(), "https required") {
		t.Fatalf("http redirect: err = %v, want https required", err)
	}
	toHTTPS := httptest.NewRequest(http.MethodGet, "https://example.com/f", nil)
	if err := c.CheckRedirect(toHTTPS, make([]*http.Request, 1)); err != nil {
		t.Fatalf("https redirect: %v", err)
	}
}
