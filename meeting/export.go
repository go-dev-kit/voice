package meeting

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

var ErrCaptionTimingRequired = errors.New("caption export requires positive turn timing")
var ErrRedactionRequired = errors.New("meeting export requires redaction")

// ExportOptions configures meeting artifact export behavior.
type ExportOptions struct {
	RequireRedaction bool
	// RequireConsent refuses export unless consent is obtained for the meeting
	// and, when per-participant records exist, for every participant.
	RequireConsent bool
}

// ExportJSON returns a stable product-facing JSON representation.
func (m Meeting) ExportJSON() ([]byte, error) {
	return m.ExportJSONWithOptions(ExportOptions{})
}

// ExportJSONWithOptions returns stable product-facing JSON with export safety options.
func (m Meeting) ExportJSONWithOptions(options ExportOptions) ([]byte, error) {
	if err := m.Validate(); err != nil {
		return nil, err
	}
	if err := m.validateExportOptions(options); err != nil {
		return nil, err
	}
	return json.MarshalIndent(exportMeetingFromMeeting(m), "", "  ")
}

// ExportMarkdown renders meeting notes as Markdown.
func (m Meeting) ExportMarkdown() (string, error) {
	return m.ExportMarkdownWithOptions(ExportOptions{})
}

// ExportMarkdownWithOptions renders meeting notes as Markdown with export safety options.
func (m Meeting) ExportMarkdownWithOptions(options ExportOptions) (string, error) {
	if err := m.Validate(); err != nil {
		return "", err
	}
	if err := m.validateExportOptions(options); err != nil {
		return "", err
	}

	var b strings.Builder
	title := strings.TrimSpace(m.Title)
	if title == "" {
		title = m.ID
	}

	writeMarkdownHeading(&b, 1, title)
	writeMarkdownMetadata(&b, m)
	writeMarkdownSummary(&b, m.Summary)
	writeMarkdownTranscript(&b, m.Transcript)
	return strings.TrimRight(b.String(), "\n") + "\n", nil
}

// ExportSRT renders transcript turns as SubRip subtitles.
func (m Meeting) ExportSRT() (string, error) {
	return m.ExportSRTWithOptions(ExportOptions{})
}

// ExportSRTWithOptions renders transcript turns as SubRip subtitles with export safety options.
func (m Meeting) ExportSRTWithOptions(options ExportOptions) (string, error) {
	if err := m.Validate(); err != nil {
		return "", err
	}
	if err := m.validateExportOptions(options); err != nil {
		return "", err
	}

	var b strings.Builder
	cueNumber := 1
	for _, turn := range m.Transcript {
		text := normalizeExportText(turn.Text)
		if text == "" {
			continue
		}
		if turn.EndTime <= turn.StartTime {
			return "", fmt.Errorf("%w: turn %q", ErrCaptionTimingRequired, turn.ID)
		}

		fmt.Fprintf(&b, "%d\n", cueNumber)
		fmt.Fprintf(&b, "%s --> %s\n", formatSRTTimestamp(turn.StartTime), formatSRTTimestamp(turn.EndTime))
		fmt.Fprintf(&b, "%s: %s\n\n", turn.DisplayLabel(), text)
		cueNumber++
	}
	return b.String(), nil
}

// ExportWebVTT renders transcript turns as WebVTT captions.
func (m Meeting) ExportWebVTT() (string, error) {
	return m.ExportWebVTTWithOptions(ExportOptions{})
}

// ExportWebVTTWithOptions renders transcript turns as WebVTT captions with export safety options.
func (m Meeting) ExportWebVTTWithOptions(options ExportOptions) (string, error) {
	if err := m.Validate(); err != nil {
		return "", err
	}
	if err := m.validateExportOptions(options); err != nil {
		return "", err
	}

	var b strings.Builder
	b.WriteString("WEBVTT\n\n")
	for _, turn := range m.Transcript {
		text := normalizeExportText(turn.Text)
		if text == "" {
			continue
		}
		if turn.EndTime <= turn.StartTime {
			return "", fmt.Errorf("%w: turn %q", ErrCaptionTimingRequired, turn.ID)
		}

		if turn.ID != "" {
			b.WriteString(sanitizeVTTIdentifier(turn.ID))
			b.WriteString("\n")
		}
		fmt.Fprintf(&b, "%s --> %s\n", formatVTTTimestamp(turn.StartTime), formatVTTTimestamp(turn.EndTime))
		fmt.Fprintf(&b, "%s: %s\n\n", turn.DisplayLabel(), text)
	}
	return b.String(), nil
}

// DisplayLabel returns the preferred human-readable label for a transcript turn.
func (t TranscriptTurn) DisplayLabel() string {
	if strings.TrimSpace(t.SpeakerLabel) != "" {
		return strings.TrimSpace(t.SpeakerLabel)
	}
	if strings.TrimSpace(t.ParticipantID) != "" {
		return strings.TrimSpace(t.ParticipantID)
	}
	return strings.TrimSpace(t.SpeakerID)
}

func (m Meeting) validateExportOptions(options ExportOptions) error {
	if options.RequireRedaction && !m.Redacted {
		return ErrRedactionRequired
	}
	if options.RequireConsent && (m.Consent == nil || !m.Consent.Obtained(m.Participants)) {
		return ErrConsentRequired
	}
	return nil
}

func writeMarkdownMetadata(b *strings.Builder, m Meeting) {
	b.WriteString("\n")
	b.WriteString("- ID: ")
	b.WriteString(escapeMarkdownInline(m.ID))
	b.WriteString("\n")
	if m.Language != "" {
		b.WriteString("- Language: ")
		b.WriteString(escapeMarkdownInline(m.Language))
		b.WriteString("\n")
	}
	if duration := m.Duration(); duration > 0 {
		b.WriteString("- Duration: ")
		b.WriteString(duration.Round(time.Millisecond).String())
		b.WriteString("\n")
	}
	if !m.StartedAt.IsZero() {
		b.WriteString("- Started: ")
		b.WriteString(m.StartedAt.Format(time.RFC3339))
		b.WriteString("\n")
	}
	if m.Source.Kind != "" && m.Source.Kind != SourceUnknown {
		b.WriteString("- Source: ")
		b.WriteString(escapeMarkdownInline(string(m.Source.Kind)))
		if m.Source.Provider != "" {
			b.WriteString(" / ")
			b.WriteString(escapeMarkdownInline(m.Source.Provider))
		}
		b.WriteString("\n")
	}
}

func writeMarkdownSummary(b *strings.Builder, summary MeetingSummary) {
	if summary.isZero() {
		return
	}

	b.WriteString("\n")
	writeMarkdownHeading(b, 2, "Summary")
	if strings.TrimSpace(summary.Overview) != "" {
		b.WriteString(strings.TrimSpace(summary.Overview))
		b.WriteString("\n\n")
	}
	writeMarkdownTopics(b, summary.Topics)
	writeMarkdownDecisions(b, summary.Decisions)
	writeMarkdownActionItems(b, summary.ActionItems)
	writeMarkdownFollowUps(b, summary.FollowUps)
	writeMarkdownQuestions(b, summary.OpenQuestions)
	writeMarkdownRisks(b, summary.Risks)
}

func writeMarkdownTopics(b *strings.Builder, topics []Topic) {
	if len(topics) == 0 {
		return
	}
	writeMarkdownHeading(b, 3, "Topics")
	for _, topic := range topics {
		if strings.TrimSpace(topic.Title) == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(escapeMarkdownInline(topic.Title))
		if strings.TrimSpace(topic.Summary) != "" {
			b.WriteString(": ")
			b.WriteString(escapeMarkdownInline(topic.Summary))
		}
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

func writeMarkdownDecisions(b *strings.Builder, decisions []Decision) {
	if len(decisions) == 0 {
		return
	}
	writeMarkdownHeading(b, 3, "Decisions")
	for _, decision := range decisions {
		if strings.TrimSpace(decision.Text) == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(escapeMarkdownInline(decision.Text))
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

func writeMarkdownActionItems(b *strings.Builder, actionItems []ActionItem) {
	if len(actionItems) == 0 {
		return
	}
	writeMarkdownHeading(b, 3, "Action Items")
	for _, item := range actionItems {
		if strings.TrimSpace(item.Text) == "" {
			continue
		}
		b.WriteString("- [ ] ")
		b.WriteString(escapeMarkdownInline(item.Text))
		if item.OwnerName != "" {
			b.WriteString(" - ")
			b.WriteString(escapeMarkdownInline(item.OwnerName))
		}
		if !item.DueAt.IsZero() {
			b.WriteString(" (due ")
			b.WriteString(item.DueAt.Format("2006-01-02"))
			b.WriteString(")")
		}
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

func writeMarkdownFollowUps(b *strings.Builder, followUps []FollowUp) {
	if len(followUps) == 0 {
		return
	}
	writeMarkdownHeading(b, 3, "Follow Ups")
	for _, followUp := range followUps {
		if strings.TrimSpace(followUp.Text) == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(escapeMarkdownInline(followUp.Text))
		if followUp.Channel != "" {
			b.WriteString(" via ")
			b.WriteString(escapeMarkdownInline(followUp.Channel))
		}
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

func writeMarkdownQuestions(b *strings.Builder, questions []Question) {
	if len(questions) == 0 {
		return
	}
	writeMarkdownHeading(b, 3, "Open Questions")
	for _, question := range questions {
		if strings.TrimSpace(question.Text) == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(escapeMarkdownInline(question.Text))
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

func writeMarkdownRisks(b *strings.Builder, risks []Risk) {
	if len(risks) == 0 {
		return
	}
	writeMarkdownHeading(b, 3, "Risks")
	for _, risk := range risks {
		if strings.TrimSpace(risk.Text) == "" {
			continue
		}
		b.WriteString("- ")
		if risk.Severity != "" {
			b.WriteString("[")
			b.WriteString(escapeMarkdownInline(risk.Severity))
			b.WriteString("] ")
		}
		b.WriteString(escapeMarkdownInline(risk.Text))
		b.WriteString("\n")
	}
	b.WriteString("\n")
}

func writeMarkdownTranscript(b *strings.Builder, turns []TranscriptTurn) {
	if len(turns) == 0 {
		return
	}

	b.WriteString("\n")
	writeMarkdownHeading(b, 2, "Transcript")
	for _, turn := range turns {
		text := normalizeExportText(turn.Text)
		if text == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString("`")
		b.WriteString(formatPlainTimestamp(turn.StartTime))
		b.WriteString("` ")
		b.WriteString("**")
		b.WriteString(escapeMarkdownInline(turn.DisplayLabel()))
		b.WriteString(":** ")
		b.WriteString(escapeMarkdownInline(text))
		b.WriteString("\n")
	}
}

func writeMarkdownHeading(b *strings.Builder, level int, text string) {
	b.WriteString(strings.Repeat("#", level))
	b.WriteString(" ")
	b.WriteString(escapeMarkdownInline(text))
	b.WriteString("\n")
}

func (s MeetingSummary) isZero() bool {
	return strings.TrimSpace(s.Overview) == "" &&
		len(s.Topics) == 0 &&
		len(s.Decisions) == 0 &&
		len(s.ActionItems) == 0 &&
		len(s.FollowUps) == 0 &&
		len(s.OpenQuestions) == 0 &&
		len(s.Risks) == 0
}

func normalizeExportText(text string) string {
	return strings.Join(strings.Fields(text), " ")
}

func escapeMarkdownInline(text string) string {
	replacer := strings.NewReplacer("\\", "\\\\", "`", "\\`", "*", "\\*", "_", "\\_", "[", "\\[", "]", "\\]")
	return replacer.Replace(normalizeExportText(text))
}

func sanitizeVTTIdentifier(id string) string {
	id = strings.TrimSpace(id)
	id = strings.ReplaceAll(id, "\n", " ")
	id = strings.ReplaceAll(id, "\r", " ")
	id = strings.ReplaceAll(id, "-->", "->")
	return id
}

func formatPlainTimestamp(duration time.Duration) string {
	if duration < 0 {
		duration = 0
	}
	totalSeconds := int64(duration / time.Second)
	hours := totalSeconds / 3600
	minutes := (totalSeconds % 3600) / 60
	seconds := totalSeconds % 60
	if hours > 0 {
		return fmt.Sprintf("%02d:%02d:%02d", hours, minutes, seconds)
	}
	return fmt.Sprintf("%02d:%02d", minutes, seconds)
}

func formatSRTTimestamp(duration time.Duration) string {
	hours, minutes, seconds, millis := splitDuration(duration)
	return fmt.Sprintf("%02d:%02d:%02d,%03d", hours, minutes, seconds, millis)
}

func formatVTTTimestamp(duration time.Duration) string {
	hours, minutes, seconds, millis := splitDuration(duration)
	return fmt.Sprintf("%02d:%02d:%02d.%03d", hours, minutes, seconds, millis)
}

func splitDuration(duration time.Duration) (hours int64, minutes int64, seconds int64, millis int64) {
	if duration < 0 {
		duration = 0
	}
	totalMillis := duration.Milliseconds()
	millis = totalMillis % 1000
	totalSeconds := totalMillis / 1000
	seconds = totalSeconds % 60
	totalMinutes := totalSeconds / 60
	minutes = totalMinutes % 60
	hours = totalMinutes / 60
	return hours, minutes, seconds, millis
}

type exportMeeting struct {
	ID           string             `json:"id"`
	Title        string             `json:"title,omitempty"`
	Language     string             `json:"language,omitempty"`
	Source       Source             `json:"source"`
	Participants []Participant      `json:"participants,omitempty"`
	Transcript   []exportTurn       `json:"transcript,omitempty"`
	Summary      *exportSummary     `json:"summary,omitempty"`
	Redacted     bool               `json:"redacted,omitempty"`
	Consent      *Consent           `json:"consent,omitempty"`
	Retention    *Retention         `json:"retention,omitempty"`
	StartedAt    time.Time          `json:"started_at,omitempty"`
	EndedAt      time.Time          `json:"ended_at,omitempty"`
	CreatedAt    time.Time          `json:"created_at"`
	ProcessedAt  time.Time          `json:"processed_at,omitempty"`
	DurationSec  float64            `json:"duration_sec,omitempty"`
	TalkTimeSec  map[string]float64 `json:"talk_time_sec,omitempty"`
}

type exportTurn struct {
	ID            string       `json:"id,omitempty"`
	SpeakerID     string       `json:"speaker_id"`
	ParticipantID string       `json:"participant_id,omitempty"`
	SpeakerLabel  string       `json:"speaker_label,omitempty"`
	Text          string       `json:"text"`
	StartSec      float64      `json:"start_sec"`
	EndSec        float64      `json:"end_sec"`
	DurationSec   float64      `json:"duration_sec"`
	Confidence    float32      `json:"confidence"`
	Words         []exportWord `json:"words,omitempty"`
}

type exportWord struct {
	Text       string  `json:"text"`
	StartSec   float64 `json:"start_sec"`
	EndSec     float64 `json:"end_sec"`
	Confidence float32 `json:"confidence"`
}

type exportSummary struct {
	Overview      string           `json:"overview,omitempty"`
	Topics        []exportTopic    `json:"topics,omitempty"`
	Decisions     []exportDecision `json:"decisions,omitempty"`
	ActionItems   []ActionItem     `json:"action_items,omitempty"`
	FollowUps     []FollowUp       `json:"follow_ups,omitempty"`
	OpenQuestions []Question       `json:"open_questions,omitempty"`
	Risks         []Risk           `json:"risks,omitempty"`
}

type exportTopic struct {
	ID       string  `json:"id,omitempty"`
	Title    string  `json:"title"`
	Summary  string  `json:"summary,omitempty"`
	StartSec float64 `json:"start_sec,omitempty"`
	EndSec   float64 `json:"end_sec,omitempty"`
}

type exportDecision struct {
	ID          string   `json:"id,omitempty"`
	Text        string   `json:"text"`
	OwnerID     string   `json:"owner_id,omitempty"`
	StartSec    float64  `json:"start_sec,omitempty"`
	EvidenceIDs []string `json:"evidence_ids,omitempty"`
}

func exportMeetingFromMeeting(m Meeting) exportMeeting {
	transcript := make([]exportTurn, 0, len(m.Transcript))
	for _, turn := range m.Transcript {
		transcript = append(transcript, exportTurnFromTurn(turn))
	}

	talkTime := m.SpeakerTalkTime()
	talkTimeSec := make(map[string]float64, len(talkTime))
	for speakerID, duration := range talkTime {
		talkTimeSec[speakerID] = duration.Seconds()
	}
	if len(talkTimeSec) == 0 {
		talkTimeSec = nil
	}

	return exportMeeting{
		ID:           m.ID,
		Title:        m.Title,
		Language:     m.Language,
		Source:       m.Source,
		Participants: m.Participants,
		Transcript:   transcript,
		Summary:      exportSummaryFromSummary(m.Summary),
		Redacted:     m.Redacted,
		Consent:      m.Consent,
		Retention:    m.Retention,
		StartedAt:    m.StartedAt,
		EndedAt:      m.EndedAt,
		CreatedAt:    m.CreatedAt,
		ProcessedAt:  m.ProcessedAt,
		DurationSec:  m.Duration().Seconds(),
		TalkTimeSec:  talkTimeSec,
	}
}

func exportTurnFromTurn(turn TranscriptTurn) exportTurn {
	words := make([]exportWord, 0, len(turn.Words))
	for _, word := range turn.Words {
		words = append(words, exportWord{
			Text:       word.Text,
			StartSec:   word.StartTime.Seconds(),
			EndSec:     word.EndTime.Seconds(),
			Confidence: word.Confidence,
		})
	}

	return exportTurn{
		ID:            turn.ID,
		SpeakerID:     turn.SpeakerID,
		ParticipantID: turn.ParticipantID,
		SpeakerLabel:  turn.DisplayLabel(),
		Text:          turn.Text,
		StartSec:      turn.StartTime.Seconds(),
		EndSec:        turn.EndTime.Seconds(),
		DurationSec:   turn.Duration().Seconds(),
		Confidence:    turn.Confidence,
		Words:         words,
	}
}

func exportSummaryFromSummary(summary MeetingSummary) *exportSummary {
	if summary.isZero() {
		return nil
	}

	topics := make([]exportTopic, 0, len(summary.Topics))
	for _, topic := range summary.Topics {
		topics = append(topics, exportTopic{
			ID:       topic.ID,
			Title:    topic.Title,
			Summary:  topic.Summary,
			StartSec: topic.StartTime.Seconds(),
			EndSec:   topic.EndTime.Seconds(),
		})
	}

	decisions := make([]exportDecision, 0, len(summary.Decisions))
	for _, decision := range summary.Decisions {
		decisions = append(decisions, exportDecision{
			ID:          decision.ID,
			Text:        decision.Text,
			OwnerID:     decision.OwnerID,
			StartSec:    decision.StartTime.Seconds(),
			EvidenceIDs: decision.EvidenceIDs,
		})
	}

	return &exportSummary{
		Overview:      summary.Overview,
		Topics:        topics,
		Decisions:     decisions,
		ActionItems:   summary.ActionItems,
		FollowUps:     summary.FollowUps,
		OpenQuestions: summary.OpenQuestions,
		Risks:         summary.Risks,
	}
}
