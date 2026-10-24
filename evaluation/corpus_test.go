package evaluation

import (
	"bufio"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const fleursFixture = "1855\ta.wav\tLöwen, Tiere.\tlöwen tiere\tl ö w e n | t i e r e |\t413760\tMALE\n" +
	"\n" +
	"1856\tb.wav\tZwei\tzwei\tz w e i |\t100\tFEMALE\r\n"

func TestLoadFLEURSTSV(t *testing.T) {
	rows, err := LoadFLEURSTSV(strings.NewReader(fleursFixture), "/data/wav")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("rows = %d", len(rows))
	}
	r := rows[0]
	if r.ID != "1855" || r.File != "a.wav" || r.AudioPath != filepath.Join("/data/wav", "a.wav") ||
		r.Transcription != "Löwen, Tiere." || r.Normalized != "löwen tiere" || r.NumSamples != 413760 || r.Gender != "MALE" {
		t.Fatalf("row = %+v", r)
	}
	if rows[1].Gender != "FEMALE" {
		t.Fatalf("CRLF not trimmed: %+v", rows[1])
	}
}

func TestLoadFLEURSTSVErrors(t *testing.T) {
	tests := []struct {
		name, in string
		line     int
	}{
		{"columns", "a\tb\tc\n", 1},
		{"samples", "ok\tf.wav\tt\tn\tc\t10\tM\nx\tf.wav\tt\tn\tc\tNaN\tM\n", 2},
		{"negative samples", "x\tf.wav\tt\tn\tc\t-1\tM\n", 1},
		{"empty id", "\tf.wav\tt\tn\tc\t1\tM\n", 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadFLEURSTSV(strings.NewReader(tt.in), "")
			var ce *CorpusError
			if !errors.As(err, &ce) || ce.Line != tt.line {
				t.Fatalf("err = %v, want CorpusError at line %d", err, tt.line)
			}
		})
	}
}

func TestLoadManifestJSONL(t *testing.T) {
	in := `{"id":"a","audio":"x/a.wav","reference":"hallo welt","duration_s":1.5,"language":"de","speaker":"s1"}` + "\n\n" +
		`{"id":"b","audio":"/abs/b.wav","reference":"zwei"}`
	rows, err := LoadManifestJSONL(strings.NewReader(in), "/m")
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].AudioPath != filepath.Join("/m", "x/a.wav") || rows[0].DurationS != 1.5 ||
		rows[0].Language != "de" || rows[0].Speaker != "s1" || rows[1].AudioPath != "/abs/b.wav" {
		t.Fatalf("rows = %+v", rows)
	}
}

func TestLoadManifestJSONLErrors(t *testing.T) {
	tests := []struct {
		name, in string
		line     int
	}{
		{"bad json", "{", 1},
		{"unknown field", `{"id":"a","audio":"a","reference":"r","extra":1}`, 1},
		{"missing id", `{"audio":"a","reference":"r"}`, 1},
		{"missing audio", `{"id":"a","reference":"r"}`, 1},
		{"missing reference", `{"id":"a","audio":"a"}`, 1},
		{"negative duration", `{"id":"a","audio":"a","reference":"r","duration_s":-1}`, 1},
		{"trailing data", `{"id":"a","audio":"a","reference":"r"} {}`, 1},
		{"stray closing brace", `{"id":"a","audio":"a","reference":"r"} }`, 1},
		{"trailing scalar", `{"id":"a","audio":"a","reference":"r"} 5`, 1},
		{"duplicate", `{"id":"a","audio":"a","reference":"r"}` + "\n" + `{"id":"a","audio":"b","reference":"r"}`, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadManifestJSONL(strings.NewReader(tt.in), "")
			var ce *CorpusError
			if !errors.As(err, &ce) || ce.Line != tt.line {
				t.Fatalf("err = %v, want CorpusError at line %d", err, tt.line)
			}
		})
	}
}

func TestCorpusLineBound(t *testing.T) {
	long := strings.Repeat("a", MaxCorpusLineBytes+1)
	_, err := LoadManifestJSONL(strings.NewReader(`{"id":"a","audio":"a","reference":"r"}`+"\n"+long+"\n"), "")
	var ce *CorpusError
	if !errors.As(err, &ce) || ce.Line != 2 || !errors.Is(err, ErrCorpusLineTooLong) {
		t.Fatalf("err = %v", err)
	}
	// exactly at the bound is accepted by the line reader (and rejected only by the parser)
	atBound := strings.Repeat("a", MaxCorpusLineBytes)
	got, rerr := readBoundedLine(bufioReader(atBound + "\n"))
	if rerr != nil || len(got) != MaxCorpusLineBytes {
		t.Fatalf("at-bound read: len=%d err=%v", len(got), rerr)
	}
}

func bufioReader(s string) *bufio.Reader { return bufio.NewReaderSize(strings.NewReader(s), 4096) }

func TestLoadFileHelpers(t *testing.T) {
	dir := t.TempDir()
	if err := os.Mkdir(filepath.Join(dir, "wav"), 0o755); err != nil {
		t.Fatal(err)
	}
	tsv := filepath.Join(dir, "test.tsv")
	if err := os.WriteFile(tsv, []byte(fleursFixture), 0o600); err != nil {
		t.Fatal(err)
	}
	rows, err := LoadFLEURSTSVFile(tsv)
	if err != nil || rows[0].AudioPath != filepath.Join(dir, "wav", "a.wav") {
		t.Fatalf("rows=%+v err=%v", rows, err)
	}
	man := filepath.Join(dir, "m.jsonl")
	if err := os.WriteFile(man, []byte(`{"id":"a","audio":"a.wav","reference":"r"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	mrows, err := LoadManifestJSONLFile(man)
	if err != nil || mrows[0].AudioPath != filepath.Join(dir, "a.wav") {
		t.Fatalf("rows=%+v err=%v", mrows, err)
	}
	if _, err := LoadManifestJSONLFile(filepath.Join(dir, "missing")); err == nil {
		t.Fatal("want error for missing file")
	}
}

func TestLoadLocalFLEURSDE(t *testing.T) {
	path := os.Getenv("VOICEKIT_FLEURS_DE_TSV")
	if path == "" {
		t.Skip("VOICEKIT_FLEURS_DE_TSV not set")
	}
	rows, err := LoadFLEURSTSVFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) == 0 {
		t.Fatal("no rows")
	}
	n := GermanNormalizer()
	found := 0
	for _, r := range rows[:min(len(rows), 50)] {
		if _, err := os.Stat(r.AudioPath); err == nil {
			found++
		}
		// The dataset's own normalized text should score close to our German profile.
		if rep := ScoreText(n, r.Normalized, r.Transcription); rep.WordErrorRate > 0.35 {
			t.Fatalf("row %s WER %.2f unexpectedly high", r.ID, rep.WordErrorRate)
		}
	}
	// The local cache holds only a subset of the audio; require that paths resolve for some.
	if found == 0 {
		t.Fatalf("no audio path resolved to an existing file under %s", filepath.Dir(rows[0].AudioPath))
	}
}

func TestRTFx(t *testing.T) {
	got, err := RTFx(500*time.Millisecond, 2*time.Second)
	if err != nil || got != 4 {
		t.Fatalf("RTFx = %v, %v", got, err)
	}
	for _, c := range [][2]time.Duration{{0, time.Second}, {-1, time.Second}, {time.Second, 0}, {time.Second, -1}} {
		if _, err := RTFx(c[0], c[1]); err != ErrInvalidDuration {
			t.Fatalf("RTFx(%v,%v) err = %v", c[0], c[1], err)
		}
	}
	if f, err := RealTimeFactor(0, time.Second); err != nil || f != 0 {
		t.Fatalf("RealTimeFactor zero processing = %v, %v", f, err)
	}
}

func TestLoadManifestJSONLTrailingWhitespace(t *testing.T) {
	in := `{"id":"a","audio":"a.wav","reference":"r"} 	` + "\r\n"
	rows, err := LoadManifestJSONL(strings.NewReader(in), "")
	if err != nil || len(rows) != 1 {
		t.Fatalf("rows = %v, err = %v", rows, err)
	}
}

func TestCorpusLoadersStripBOM(t *testing.T) {
	const bom = "\ufeff"
	rows, err := LoadManifestJSONL(strings.NewReader(bom+`{"id":"a","audio":"a.wav","reference":"r"}`+"\n"), "")
	if err != nil || len(rows) != 1 || rows[0].ID != "a" {
		t.Fatalf("manifest rows = %+v, err = %v", rows, err)
	}
	fr, err := LoadFLEURSTSV(strings.NewReader(bom+fleursFixture), "/d")
	if err != nil || len(fr) != 2 || fr[0].ID != "1855" {
		t.Fatalf("fleurs rows = %+v, err = %v", fr, err)
	}
	// Only the first line's BOM is dropped; one mid-file is data and fails.
	_, err = LoadManifestJSONL(strings.NewReader(`{"id":"a","audio":"a","reference":"r"}`+"\n"+bom+`{"id":"b","audio":"b","reference":"r"}`), "")
	var ce *CorpusError
	if !errors.As(err, &ce) || ce.Line != 2 {
		t.Fatalf("err = %v, want CorpusError at line 2", err)
	}
}
