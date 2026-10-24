package meeting

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestMeetingRedactCoversTranscriptSummaryParticipantsAndSource(t *testing.T) {
	meeting := sensitiveMeeting()
	originalTranscript := meeting.Transcript[0].Text
	originalAction := meeting.Summary.ActionItems[0].Text
	policy := DefaultRedactionPolicy()
	policy.Terms = []string{"Confidential Roadmap", "Ada Private"}
	redactor := NewPatternRedactor(policy)

	redacted, report, err := meeting.Redact(redactor)
	if err != nil {
		t.Fatalf("Redact returned error: %v", err)
	}
	if !redacted.Redacted || !report.Redacted {
		t.Fatalf("expected redacted flags, meeting=%v report=%v", redacted.Redacted, report.Redacted)
	}
	if len(report.Matches) == 0 {
		t.Fatal("expected redaction matches")
	}
	if meeting.Redacted {
		t.Fatal("redaction should not mutate the original meeting redacted flag")
	}
	if meeting.Transcript[0].Text != originalTranscript || meeting.Summary.ActionItems[0].Text != originalAction {
		t.Fatal("redaction should not mutate the original meeting slices")
	}

	assertNoSensitiveText(t, redacted.Title)
	assertNoSensitiveText(t, redacted.Source.Reference)
	assertNoSensitiveText(t, redacted.Participants[0].DisplayName)
	assertNoSensitiveText(t, redacted.Transcript[0].SpeakerLabel)
	assertNoSensitiveText(t, redacted.Transcript[0].Text)
	assertNoSensitiveText(t, redacted.Transcript[0].Words[0].Text)
	assertNoSensitiveText(t, redacted.Summary.Overview)
	assertNoSensitiveText(t, redacted.Summary.Topics[0].Summary)
	assertNoSensitiveText(t, redacted.Summary.Decisions[0].Text)
	assertNoSensitiveText(t, redacted.Summary.ActionItems[0].Text)
	assertNoSensitiveText(t, redacted.Summary.ActionItems[0].OwnerName)
	assertNoSensitiveText(t, redacted.Summary.FollowUps[0].Text)
	assertNoSensitiveText(t, redacted.Summary.OpenQuestions[0].Text)
	assertNoSensitiveText(t, redacted.Summary.Risks[0].Text)

	for _, match := range report.Matches {
		if match.Field == "" || match.Kind == "" || match.Replacement == "" {
			t.Fatalf("expected non-sensitive match metadata, got %+v", match)
		}
	}
}

func TestMeetingRedactRequiresRedactor(t *testing.T) {
	_, _, err := sensitiveMeeting().Redact(nil)
	if !errors.Is(err, ErrRedactorRequired) {
		t.Fatalf("expected ErrRedactorRequired, got %v", err)
	}
}

func TestExportOptionsRequireRedaction(t *testing.T) {
	meeting := sensitiveMeeting()
	if _, err := meeting.ExportMarkdownWithOptions(ExportOptions{RequireRedaction: true}); !errors.Is(err, ErrRedactionRequired) {
		t.Fatalf("expected ErrRedactionRequired from markdown export, got %v", err)
	}
	if _, err := meeting.ExportJSONWithOptions(ExportOptions{RequireRedaction: true}); !errors.Is(err, ErrRedactionRequired) {
		t.Fatalf("expected ErrRedactionRequired from JSON export, got %v", err)
	}
	if _, err := meeting.ExportSRTWithOptions(ExportOptions{RequireRedaction: true}); !errors.Is(err, ErrRedactionRequired) {
		t.Fatalf("expected ErrRedactionRequired from SRT export, got %v", err)
	}
	if _, err := meeting.ExportWebVTTWithOptions(ExportOptions{RequireRedaction: true}); !errors.Is(err, ErrRedactionRequired) {
		t.Fatalf("expected ErrRedactionRequired from WebVTT export, got %v", err)
	}
}

func TestRedactedExportsPassRequiredRedaction(t *testing.T) {
	policy := DefaultRedactionPolicy()
	policy.Terms = []string{"Confidential Roadmap", "Ada Private"}
	redacted, _, err := sensitiveMeeting().Redact(NewPatternRedactor(policy))
	if err != nil {
		t.Fatalf("Redact returned error: %v", err)
	}

	markdown, err := redacted.ExportMarkdownWithOptions(ExportOptions{RequireRedaction: true})
	if err != nil {
		t.Fatalf("ExportMarkdownWithOptions returned error: %v", err)
	}
	assertNoSensitiveText(t, markdown)

	jsonData, err := redacted.ExportJSONWithOptions(ExportOptions{RequireRedaction: true})
	if err != nil {
		t.Fatalf("ExportJSONWithOptions returned error: %v", err)
	}
	assertNoSensitiveText(t, string(jsonData))
	var exported struct {
		Redacted bool `json:"redacted"`
	}
	if err := json.Unmarshal(jsonData, &exported); err != nil {
		t.Fatalf("JSON export failed to unmarshal: %v", err)
	}
	if !exported.Redacted {
		t.Fatal("expected JSON export to include redacted flag")
	}

	srt, err := redacted.ExportSRTWithOptions(ExportOptions{RequireRedaction: true})
	if err != nil {
		t.Fatalf("ExportSRTWithOptions returned error: %v", err)
	}
	assertNoSensitiveText(t, srt)

	vtt, err := redacted.ExportWebVTTWithOptions(ExportOptions{RequireRedaction: true})
	if err != nil {
		t.Fatalf("ExportWebVTTWithOptions returned error: %v", err)
	}
	assertNoSensitiveText(t, vtt)
}

func sensitiveMeeting() Meeting {
	return Meeting{
		ID:        "meeting-sensitive",
		Title:     "Confidential Roadmap sync",
		Language:  "en",
		Source:    Source{Kind: SourceMeetingPlatform, Provider: "google_meet", Reference: "https://meet.example.com/private-room"},
		CreatedAt: time.Date(2026, 7, 4, 12, 0, 0, 0, time.UTC),
		Participants: []Participant{
			{ID: "p1", DisplayName: "Ada Private", SpeakerID: "speaker-1"},
		},
		Transcript: []TranscriptTurn{
			{
				ID:           "turn-1",
				SpeakerID:    "speaker-1",
				SpeakerLabel: "Ada Private",
				Text:         "Email ada@example.com and call +1 415-555-1212 about the Confidential Roadmap.",
				StartTime:    0,
				EndTime:      2 * time.Second,
				Words: []Word{
					{Text: "ada@example.com", StartTime: 0, EndTime: 500 * time.Millisecond},
				},
			},
		},
		Summary: MeetingSummary{
			Overview: "Ada Private shared ada@example.com.",
			Topics: []Topic{
				{ID: "topic-1", Title: "Privacy", Summary: "Review https://internal.example.com/privacy"},
			},
			Decisions: []Decision{
				{ID: "decision-1", Text: "Call +1 415-555-1212 before sharing."},
			},
			ActionItems: []ActionItem{
				{ID: "action-1", Text: "Email ada@example.com", OwnerName: "Ada Private"},
			},
			FollowUps: []FollowUp{
				{ID: "follow-up-1", Text: "Send details to ada@example.com"},
			},
			OpenQuestions: []Question{
				{ID: "question-1", Text: "Can https://internal.example.com be shared?"},
			},
			Risks: []Risk{
				{ID: "risk-1", Text: "Phone +1 415-555-1212 may leak."},
			},
		},
	}
}

func assertNoSensitiveText(t *testing.T, text string) {
	t.Helper()
	for _, sensitive := range []string{
		"ada@example.com",
		"+1 415-555-1212",
		"+[REDACTED]",
		"+\\[REDACTED\\]",
		"https://meet.example.com",
		"https://internal.example.com",
		"Confidential Roadmap",
		"Ada Private",
	} {
		if strings.Contains(text, sensitive) {
			t.Fatalf("expected sensitive value %q to be redacted from %q", sensitive, text)
		}
	}
}

func consentBasisMeeting() Meeting {
	m := consentedMeeting()
	m.Consent.Basis = "verbal, confirmed by ada@example.com"
	m.Consent.Participants[0].Basis = "call +1 415 555 0100 to verify"
	m.Retention = &Retention{Policy: "std", RetainFor: time.Hour}
	return m
}

func TestMeetingRedactDoesNotShareConsentOrRetention(t *testing.T) {
	original := consentBasisMeeting()
	redacted, _, err := original.Redact(NewPatternRedactor(DefaultRedactionPolicy()))
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if redacted.Consent == original.Consent || redacted.Retention == original.Retention {
		t.Fatal("Redact result shares Consent or Retention pointers with the original")
	}
	redacted.Consent.Status = ConsentWithdrawn
	redacted.Consent.Participants[1].Status = ConsentDeclined
	redacted.Retention.LegalHold = true
	redacted.Retention.RetainFor = 0
	if original.Consent.Status != ConsentObtained || original.Consent.Participants[1].Status != ConsentObtained {
		t.Fatalf("mutating redacted consent changed the original: %+v", original.Consent)
	}
	if original.Consent.Basis != "verbal, confirmed by ada@example.com" || original.Consent.Participants[0].Basis != "call +1 415 555 0100 to verify" {
		t.Fatalf("redaction altered the original basis: %+v", original.Consent)
	}
	if original.Retention.LegalHold || original.Retention.RetainFor != time.Hour {
		t.Fatalf("mutating redacted retention changed the original: %+v", original.Retention)
	}
}

func TestMeetingRedactRedactsConsentBasis(t *testing.T) {
	redacted, report, err := consentBasisMeeting().Redact(NewPatternRedactor(DefaultRedactionPolicy()))
	if err != nil {
		t.Fatalf("Redact: %v", err)
	}
	if strings.Contains(redacted.Consent.Basis, "ada@example.com") || !strings.Contains(redacted.Consent.Basis, "[REDACTED]") {
		t.Fatalf("meeting consent basis not redacted: %q", redacted.Consent.Basis)
	}
	if strings.Contains(redacted.Consent.Participants[0].Basis, "555") {
		t.Fatalf("participant consent basis not redacted: %q", redacted.Consent.Participants[0].Basis)
	}
	fields := map[string]bool{}
	for _, match := range report.Matches {
		fields[match.Field] = true
	}
	for _, want := range []string{"meeting.consent.basis", "meeting.consent.participants[0].basis"} {
		if !fields[want] {
			t.Fatalf("report lacks field %q: %+v", want, report.Matches)
		}
	}

	data, err := redacted.ExportJSONWithOptions(ExportOptions{RequireRedaction: true})
	if err != nil {
		t.Fatalf("ExportJSONWithOptions: %v", err)
	}
	if strings.Contains(string(data), "ada@example.com") || strings.Contains(string(data), "555 0100") {
		t.Fatalf("redacted export leaks consent basis: %s", data)
	}
}
