package meeting

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"errors"
	"io"
	"strings"
	"testing"
	"time"
)

func readDOCX(t *testing.T, data []byte) map[string]string {
	t.Helper()
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("not a zip: %v", err)
	}
	parts := make(map[string]string, len(zr.File))
	for _, f := range zr.File {
		rc, err := f.Open()
		if err != nil {
			t.Fatalf("open %s: %v", f.Name, err)
		}
		content, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			t.Fatalf("read %s: %v", f.Name, err)
		}
		parts[f.Name] = string(content)
	}
	return parts
}

// docxText parses WordprocessingML and returns the text of each paragraph.
func docxText(t *testing.T, document string) []string {
	t.Helper()
	dec := xml.NewDecoder(strings.NewReader(document))
	var paragraphs []string
	var cur strings.Builder
	inText := false
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("document.xml is not well-formed: %v", err)
		}
		switch el := tok.(type) {
		case xml.StartElement:
			switch el.Name.Local {
			case "p":
				cur.Reset()
			case "t":
				inText = true
			}
		case xml.CharData:
			if inText {
				cur.Write(el)
			}
		case xml.EndElement:
			switch el.Name.Local {
			case "t":
				inText = false
			case "p":
				paragraphs = append(paragraphs, cur.String())
			}
		}
	}
	return paragraphs
}

func TestExportDOCXPackageStructure(t *testing.T) {
	m := sampleMeeting()
	data, err := m.ExportDOCX()
	if err != nil {
		t.Fatalf("ExportDOCX: %v", err)
	}
	parts := readDOCX(t, data)
	for _, name := range []string{"[Content_Types].xml", "_rels/.rels", "word/document.xml"} {
		content, ok := parts[name]
		if !ok {
			t.Fatalf("missing part %s", name)
		}
		dec := xml.NewDecoder(strings.NewReader(content))
		for {
			if _, err := dec.Token(); err == io.EOF {
				break
			} else if err != nil {
				t.Fatalf("%s is not well-formed: %v", name, err)
			}
		}
	}
	if len(parts) != 3 {
		t.Fatalf("expected 3 parts, got %d", len(parts))
	}
	if !strings.Contains(parts["[Content_Types].xml"], "wordprocessingml.document.main+xml") {
		t.Fatal("content types must declare the main document part")
	}
	if !strings.Contains(parts["_rels/.rels"], `Target="word/document.xml"`) {
		t.Fatal("rels must point at word/document.xml")
	}

	text := docxText(t, parts["word/document.xml"])
	joined := strings.Join(text, "\n")
	for _, want := range []string{
		"Product sync",
		"ID: meeting-1",
		"Language: en",
		"Source: meeting_platform / google_meet",
		"Summary",
		"Finalize the export layer.",
		"Action Items",
		"Transcript",
		"00:00 Ada: Ready to go",
		"00:01 Grace: Ship it now",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("document missing %q in:\n%s", want, joined)
		}
	}
}

func TestExportDOCXEscaping(t *testing.T) {
	m := sampleMeeting()
	m.Title = `A <b> & "q" 'a'`
	m.Transcript[0].Text = "x<y & z>w \"quoted\" 'single' \x01\x08 bell\x00end"
	m.Summary.Overview = "</w:t><w:p>injected</w:p>"

	data, err := m.ExportDOCX()
	if err != nil {
		t.Fatalf("ExportDOCX: %v", err)
	}
	doc := readDOCX(t, data)["word/document.xml"]
	for _, raw := range []string{"<b>", "injected</w:p>", "\x01", "\x08", "\x00"} {
		if strings.Contains(doc, raw) {
			t.Errorf("document contains unescaped %q", raw)
		}
	}
	text := strings.Join(docxText(t, doc), "\n")
	for _, want := range []string{
		`A <b> & "q" 'a'`,
		`x<y & z>w "quoted" 'single'`,
		"</w:t><w:p>injected</w:p>",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("round-tripped text missing %q in:\n%s", want, text)
		}
	}
	if !strings.Contains(text, "�") {
		t.Error("control characters should be replaced with U+FFFD")
	}
}

func TestExportDOCXEmptyMeeting(t *testing.T) {
	m := Meeting{ID: "empty"}
	data, err := m.ExportDOCX()
	if err != nil {
		t.Fatalf("ExportDOCX: %v", err)
	}
	text := docxText(t, readDOCX(t, data)["word/document.xml"])
	if len(text) < 2 || text[0] != "empty" || text[1] != "ID: empty" {
		t.Fatalf("unexpected empty-meeting paragraphs: %q", text)
	}
	for _, p := range text {
		if p == "Summary" || p == "Transcript" {
			t.Fatalf("empty meeting must not emit %q section", p)
		}
	}
}

func TestExportDOCXDeterministic(t *testing.T) {
	m := consentedMeeting()
	first, err := m.ExportDOCX()
	if err != nil {
		t.Fatal(err)
	}
	second, err := m.ExportDOCX()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("DOCX output must be byte-identical across runs")
	}
	zr, err := zip.NewReader(bytes.NewReader(first), int64(len(first)))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range zr.File {
		if !f.Modified.IsZero() && f.Modified.Year() > 1980 {
			t.Errorf("%s carries a timestamp: %v", f.Name, f.Modified)
		}
	}
}

func TestExportDOCXOptions(t *testing.T) {
	m := sampleMeeting()
	if _, err := m.ExportDOCXWithOptions(ExportOptions{RequireRedaction: true}); !errors.Is(err, ErrRedactionRequired) {
		t.Fatalf("expected ErrRedactionRequired, got %v", err)
	}
	m.Redacted = true
	if _, err := m.ExportDOCXWithOptions(ExportOptions{RequireRedaction: true}); err != nil {
		t.Fatalf("redacted export failed: %v", err)
	}
	if _, err := m.ExportDOCXWithOptions(ExportOptions{RequireConsent: true}); !errors.Is(err, ErrConsentRequired) {
		t.Fatalf("expected ErrConsentRequired, got %v", err)
	}
	if _, err := (Meeting{}).ExportDOCX(); !errors.Is(err, ErrMeetingIDRequired) {
		t.Fatalf("expected validation error, got %v", err)
	}
}

func TestExportDOCXConsentRetentionMetadata(t *testing.T) {
	m := consentedMeeting()
	m.Retention = &Retention{RetainUntil: time.Date(2030, 1, 2, 3, 4, 5, 0, time.UTC), LegalHold: true}
	// Legal hold suppresses the due date but is itself reported.
	data, err := m.ExportDOCX()
	if err != nil {
		t.Fatal(err)
	}
	text := strings.Join(docxText(t, readDOCX(t, data)["word/document.xml"]), "\n")
	if !strings.Contains(text, "Consent: obtained") || !strings.Contains(text, "Legal hold: yes") {
		t.Fatalf("missing consent/legal hold metadata:\n%s", text)
	}
	if strings.Contains(text, "Retain until") {
		t.Fatalf("legal hold must suppress retain-until:\n%s", text)
	}
	m.Retention.LegalHold = false
	data, _ = m.ExportDOCX()
	text = strings.Join(docxText(t, readDOCX(t, data)["word/document.xml"]), "\n")
	if !strings.Contains(text, "Retain until: 2030-01-02T03:04:05Z") {
		t.Fatalf("missing retain-until:\n%s", text)
	}
}
