package meeting

import (
	"archive/zip"
	"bytes"
	"encoding/xml"
	"fmt"
	"strings"
	"time"
)

const (
	docxContentTypes = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` + "\n" +
		`<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">` +
		`<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>` +
		`<Default Extension="xml" ContentType="application/xml"/>` +
		`<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>` +
		`</Types>`
	docxRels = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` + "\n" +
		`<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
		`<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>` +
		`</Relationships>`
	docxDocumentHead = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` + "\n" +
		`<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>`
	docxDocumentTail = `</w:body></w:document>`
)

// ExportDOCX renders meeting notes as a minimal WordprocessingML package.
func (m Meeting) ExportDOCX() ([]byte, error) {
	return m.ExportDOCXWithOptions(ExportOptions{})
}

// ExportDOCXWithOptions renders meeting notes as a DOCX package with export
// safety options. Output is deterministic: zip entries carry no timestamps.
func (m Meeting) ExportDOCXWithOptions(options ExportOptions) ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := m.validateExportOptions(options); err != nil {
		return nil, err
	}

	var body strings.Builder
	body.WriteString(docxDocumentHead)
	writeDOCXBody(&body, m)
	body.WriteString(docxDocumentTail)

	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	parts := []struct{ name, content string }{
		{"[Content_Types].xml", docxContentTypes},
		{"_rels/.rels", docxRels},
		{"word/document.xml", body.String()},
	}
	for _, part := range parts {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: part.name, Method: zip.Deflate})
		if err != nil {
			return nil, fmt.Errorf("create docx part %s: %w", part.name, err)
		}
		if _, err := w.Write([]byte(part.content)); err != nil {
			return nil, fmt.Errorf("write docx part %s: %w", part.name, err)
		}
	}
	if err := zw.Close(); err != nil {
		return nil, fmt.Errorf("close docx package: %w", err)
	}
	return buf.Bytes(), nil
}

func writeDOCXBody(b *strings.Builder, m Meeting) {
	title := strings.TrimSpace(m.Title)
	if title == "" {
		title = m.ID
	}
	writeDOCXParagraph(b, title, 40, true)
	writeDOCXMetadata(b, m)
	writeDOCXSummary(b, m.Summary)
	writeDOCXTranscript(b, m.Transcript)
}

func writeDOCXMetadata(b *strings.Builder, m Meeting) {
	writeDOCXParagraph(b, "ID: "+m.ID, 0, false)
	if m.Language != "" {
		writeDOCXParagraph(b, "Language: "+m.Language, 0, false)
	}
	if duration := m.Duration(); duration > 0 {
		writeDOCXParagraph(b, "Duration: "+duration.Round(time.Millisecond).String(), 0, false)
	}
	if !m.StartedAt.IsZero() {
		writeDOCXParagraph(b, "Started: "+m.StartedAt.Format(time.RFC3339), 0, false)
	}
	if m.Source.Kind != "" && m.Source.Kind != SourceUnknown {
		source := string(m.Source.Kind)
		if m.Source.Provider != "" {
			source += " / " + m.Source.Provider
		}
		writeDOCXParagraph(b, "Source: "+source, 0, false)
	}
	if m.Consent != nil {
		status := m.Consent.Status
		if status == "" {
			status = ConsentUnknown
		}
		writeDOCXParagraph(b, "Consent: "+string(status), 0, false)
	}
	if due, ok := m.DeletionDue(); ok {
		writeDOCXParagraph(b, "Retain until: "+due.Format(time.RFC3339), 0, false)
	}
	if m.Retention != nil && m.Retention.LegalHold {
		writeDOCXParagraph(b, "Legal hold: yes", 0, false)
	}
}

func writeDOCXSummary(b *strings.Builder, summary MeetingSummary) {
	if summary.isZero() {
		return
	}
	writeDOCXParagraph(b, "Summary", 32, true)
	if strings.TrimSpace(summary.Overview) != "" {
		writeDOCXParagraph(b, summary.Overview, 0, false)
	}

	var lines []string
	section := func(heading string) {
		if len(lines) > 0 {
			writeDOCXParagraph(b, heading, 28, true)
			for _, line := range lines {
				writeDOCXParagraph(b, "• "+line, 0, false)
			}
		}
		lines = lines[:0]
	}
	for _, topic := range summary.Topics {
		if strings.TrimSpace(topic.Title) == "" {
			continue
		}
		line := topic.Title
		if strings.TrimSpace(topic.Summary) != "" {
			line += ": " + topic.Summary
		}
		lines = append(lines, line)
	}
	section("Topics")
	for _, decision := range summary.Decisions {
		if strings.TrimSpace(decision.Text) != "" {
			lines = append(lines, decision.Text)
		}
	}
	section("Decisions")
	for _, item := range summary.ActionItems {
		if strings.TrimSpace(item.Text) == "" {
			continue
		}
		line := item.Text
		if item.OwnerName != "" {
			line += " - " + item.OwnerName
		}
		if !item.DueAt.IsZero() {
			line += " (due " + item.DueAt.Format("2006-01-02") + ")"
		}
		lines = append(lines, line)
	}
	section("Action Items")
	for _, followUp := range summary.FollowUps {
		if strings.TrimSpace(followUp.Text) == "" {
			continue
		}
		line := followUp.Text
		if followUp.Channel != "" {
			line += " via " + followUp.Channel
		}
		lines = append(lines, line)
	}
	section("Follow Ups")
	for _, question := range summary.OpenQuestions {
		if strings.TrimSpace(question.Text) != "" {
			lines = append(lines, question.Text)
		}
	}
	section("Open Questions")
	for _, risk := range summary.Risks {
		if strings.TrimSpace(risk.Text) == "" {
			continue
		}
		line := risk.Text
		if risk.Severity != "" {
			line = "[" + risk.Severity + "] " + line
		}
		lines = append(lines, line)
	}
	section("Risks")
}

func writeDOCXTranscript(b *strings.Builder, turns []TranscriptTurn) {
	if len(turns) == 0 {
		return
	}
	writeDOCXParagraph(b, "Transcript", 32, true)
	for _, turn := range turns {
		text := normalizeExportText(turn.Text)
		if text == "" {
			continue
		}
		writeDOCXParagraph(b, formatPlainTimestamp(turn.StartTime)+" "+normalizeExportText(turn.DisplayLabel())+": "+text, 0, false)
	}
}

// writeDOCXParagraph writes one paragraph. halfPoints sets the font size in
// half-points when positive; bold toggles bold run formatting.
func writeDOCXParagraph(b *strings.Builder, text string, halfPoints int, bold bool) {
	b.WriteString("<w:p><w:r>")
	if bold || halfPoints > 0 {
		b.WriteString("<w:rPr>")
		if bold {
			b.WriteString("<w:b/>")
		}
		if halfPoints > 0 {
			fmt.Fprintf(b, `<w:sz w:val="%d"/>`, halfPoints)
		}
		b.WriteString("</w:rPr>")
	}
	b.WriteString(`<w:t xml:space="preserve">`)
	_ = xml.EscapeText(stringWriter{b}, []byte(normalizeExportText(text)))
	b.WriteString("</w:t></w:r></w:p>")
}

type stringWriter struct{ b *strings.Builder }

func (w stringWriter) Write(p []byte) (int, error) { return w.b.Write(p) }
