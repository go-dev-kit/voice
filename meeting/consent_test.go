package meeting

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestMeetingJSONOmitsUnsetConsentAndRetention(t *testing.T) {
	data, err := json.Marshal(sampleMeeting())
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	for _, key := range []string{"consent", "retention"} {
		if strings.Contains(string(data), `"`+key+`"`) {
			t.Fatalf("unset %s must be omitted from meeting JSON: %s", key, data)
		}
	}
	exported, err := sampleMeeting().ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON: %v", err)
	}
	for _, key := range []string{"consent", "retention"} {
		if strings.Contains(string(exported), `"`+key+`"`) {
			t.Fatalf("unset %s must be omitted from export JSON: %s", key, exported)
		}
	}
}

func TestConsentJSONOmitsUnsetFieldsAndRoundTrips(t *testing.T) {
	data, err := json.Marshal(Consent{Status: ConsentObtained})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != `{"status":"obtained"}` {
		t.Fatalf("unexpected consent JSON %s", data)
	}
	data, err = json.Marshal(Retention{LegalHold: true})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != `{"legal_hold":true}` {
		t.Fatalf("unexpected retention JSON %s", data)
	}

	m := consentedMeeting()
	m.Retention = &Retention{Policy: "standard", RetainFor: time.Hour}
	exported, err := m.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON: %v", err)
	}
	var back Meeting
	if err := json.Unmarshal(exported, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if back.Consent == nil || back.Consent.Status != ConsentObtained || len(back.Consent.Participants) != 2 {
		t.Fatalf("consent did not round trip: %+v", back.Consent)
	}
	if back.Retention == nil || back.Retention.Policy != "standard" || back.Retention.RetainFor != time.Hour {
		t.Fatalf("retention did not round trip: %+v", back.Retention)
	}
}

func TestConsentValidation(t *testing.T) {
	cases := map[string]func(*Meeting){
		"unknown status": func(m *Meeting) { m.Consent.Status = "maybe" },
		"empty participant id": func(m *Meeting) {
			m.Consent.Participants[0].ParticipantID = " "
		},
		"duplicate participant": func(m *Meeting) {
			m.Consent.Participants[1].ParticipantID = "p1"
		},
		"unknown participant": func(m *Meeting) {
			m.Consent.Participants[1].ParticipantID = "ghost"
		},
		"bad participant status": func(m *Meeting) {
			m.Consent.Participants[0].Status = "nope"
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			m := consentedMeeting()
			mutate(&m)
			if err := m.Validate(); !errors.Is(err, ErrInvalidConsent) {
				t.Fatalf("expected ErrInvalidConsent, got %v", err)
			}
		})
	}
	if err := consentedMeeting().Validate(); err != nil {
		t.Fatalf("valid consent rejected: %v", err)
	}
	m := sampleMeeting()
	m.Consent = &Consent{}
	if err := m.Validate(); err != nil {
		t.Fatalf("empty status must be valid (unknown): %v", err)
	}
}

func TestRetentionValidation(t *testing.T) {
	m := sampleMeeting()
	m.Retention = &Retention{RetainFor: -time.Second}
	if err := m.Validate(); !errors.Is(err, ErrInvalidRetention) {
		t.Fatalf("expected ErrInvalidRetention for negative duration, got %v", err)
	}
	m.Retention = &Retention{RetainFor: time.Hour, RetainUntil: time.Now()}
	if err := m.Validate(); !errors.Is(err, ErrInvalidRetention) {
		t.Fatalf("expected ErrInvalidRetention for both set, got %v", err)
	}
	m.Retention = &Retention{Policy: "p", RetainFor: time.Hour}
	if err := m.Validate(); err != nil {
		t.Fatalf("valid retention rejected: %v", err)
	}
}

func TestDeletionDue(t *testing.T) {
	created := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	started := created.Add(time.Hour)
	ended := started.Add(time.Hour)
	until := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name string
		m    Meeting
		want time.Time
		ok   bool
	}{
		{"no retention", Meeting{CreatedAt: created}, time.Time{}, false},
		{"no deadline", Meeting{CreatedAt: created, Retention: &Retention{Policy: "x"}}, time.Time{}, false},
		{"retain until", Meeting{CreatedAt: created, Retention: &Retention{RetainUntil: until}}, until, true},
		{"from end", Meeting{CreatedAt: created, StartedAt: started, EndedAt: ended, Retention: &Retention{RetainFor: 24 * time.Hour}}, ended.Add(24 * time.Hour), true},
		{"from start", Meeting{CreatedAt: created, StartedAt: started, Retention: &Retention{RetainFor: time.Hour}}, started.Add(time.Hour), true},
		{"from created", Meeting{CreatedAt: created, Retention: &Retention{RetainFor: time.Hour}}, created.Add(time.Hour), true},
		{"duration without base", Meeting{Retention: &Retention{RetainFor: time.Hour}}, time.Time{}, false},
		{"legal hold", Meeting{CreatedAt: created, Retention: &Retention{RetainUntil: until, LegalHold: true}}, time.Time{}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := tt.m.DeletionDue()
			if ok != tt.ok || !got.Equal(tt.want) {
				t.Fatalf("DeletionDue = %v, %v; want %v, %v", got, ok, tt.want, tt.ok)
			}
		})
	}

	m := Meeting{CreatedAt: created, Retention: &Retention{RetainUntil: until}}
	if m.IsDeletionDue(until.Add(-time.Nanosecond)) {
		t.Fatal("not due before deadline")
	}
	if !m.IsDeletionDue(until) {
		t.Fatal("due at deadline")
	}
	m.Retention.LegalHold = true
	if m.IsDeletionDue(until.Add(time.Hour)) {
		t.Fatal("legal hold must block deletion")
	}
}

func TestRequireConsentEnforcement(t *testing.T) {
	opts := ExportOptions{RequireConsent: true}

	exports := map[string]func(Meeting) error{
		"json":     func(m Meeting) error { _, err := m.ExportJSONWithOptions(opts); return err },
		"markdown": func(m Meeting) error { _, err := m.ExportMarkdownWithOptions(opts); return err },
		"srt":      func(m Meeting) error { _, err := m.ExportSRTWithOptions(opts); return err },
		"vtt":      func(m Meeting) error { _, err := m.ExportWebVTTWithOptions(opts); return err },
		"docx":     func(m Meeting) error { _, err := m.ExportDOCXWithOptions(opts); return err },
	}

	mutations := map[string]func(*Meeting){
		"nil consent":      func(m *Meeting) { m.Consent = nil },
		"unknown":          func(m *Meeting) { m.Consent.Status = ConsentUnknown },
		"declined":         func(m *Meeting) { m.Consent.Status = ConsentDeclined },
		"withdrawn":        func(m *Meeting) { m.Consent.Status = ConsentWithdrawn },
		"participant gone": func(m *Meeting) { m.Consent.Participants[1].Status = ConsentWithdrawn },
		"participant gap":  func(m *Meeting) { m.Consent.Participants = m.Consent.Participants[:1] },
	}
	for exportName, export := range exports {
		t.Run(exportName+"/allowed", func(t *testing.T) {
			if err := export(consentedMeeting()); err != nil {
				t.Fatalf("consented export failed: %v", err)
			}
		})
		for name, mutate := range mutations {
			t.Run(exportName+"/"+name, func(t *testing.T) {
				m := consentedMeeting()
				mutate(&m)
				if err := export(m); !errors.Is(err, ErrConsentRequired) {
					t.Fatalf("expected ErrConsentRequired, got %v", err)
				}
			})
		}
	}

	// Default options never gate on consent.
	m := sampleMeeting()
	if _, err := m.ExportMarkdown(); err != nil {
		t.Fatalf("default export must not require consent: %v", err)
	}
	if _, err := m.ExportDOCX(); err != nil {
		t.Fatalf("default docx export must not require consent: %v", err)
	}
}

func TestRequireConsentMeetingLevelOnly(t *testing.T) {
	m := sampleMeeting()
	m.Consent = &Consent{Status: ConsentObtained, Basis: "recording notice"}
	if _, err := m.ExportMarkdownWithOptions(ExportOptions{RequireConsent: true}); err != nil {
		t.Fatalf("meeting-level consent without participant records must pass: %v", err)
	}
}

func consentedMeeting() Meeting {
	m := sampleMeeting()
	at := time.Date(2026, 7, 4, 11, 0, 0, 0, time.UTC)
	m.Consent = &Consent{
		Status:     ConsentObtained,
		Basis:      "explicit",
		ObtainedAt: at,
		Participants: []ParticipantConsent{
			{ParticipantID: "p1", Status: ConsentObtained, ObtainedAt: at},
			{ParticipantID: "p2", Status: ConsentObtained, ObtainedAt: at},
		},
	}
	return m
}

func TestRetentionJSONUsesIntegerSeconds(t *testing.T) {
	r := Retention{Policy: "std", RetainFor: 90 * time.Minute, LegalHold: true}
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != `{"policy":"std","retain_for_seconds":5400,"legal_hold":true}` {
		t.Fatalf("unexpected retention JSON %s", data)
	}
	var back Retention
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if back != r {
		t.Fatalf("round trip mismatch: %+v != %+v", back, r)
	}

	until := time.Date(2027, 1, 2, 3, 4, 5, 0, time.UTC)
	data, err = json.Marshal(&Retention{RetainUntil: until})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if strings.Contains(string(data), "retain_for") {
		t.Fatalf("zero RetainFor must be omitted: %s", data)
	}
	back = Retention{}
	if err := json.Unmarshal(data, &back); err != nil || !back.RetainUntil.Equal(until) || back.RetainFor != 0 {
		t.Fatalf("retain_until round trip: %+v err=%v", back, err)
	}
}

func TestRetentionJSONRoundTripsThroughMeetingAndExport(t *testing.T) {
	m := consentedMeeting()
	m.Retention = &Retention{Policy: "standard", RetainFor: 36 * time.Hour}

	aggregate, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	exported, err := m.ExportJSON()
	if err != nil {
		t.Fatalf("ExportJSON: %v", err)
	}
	for name, data := range map[string][]byte{"aggregate": aggregate, "export": exported} {
		if !strings.Contains(string(data), `"retain_for_seconds": 129600`) && !strings.Contains(string(data), `"retain_for_seconds":129600`) {
			t.Fatalf("%s JSON lacks retain_for_seconds: %s", name, data)
		}
		if strings.Contains(string(data), `"retain_for"`) {
			t.Fatalf("%s JSON still has nanosecond retain_for: %s", name, data)
		}
		var back Meeting
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("%s Unmarshal: %v", name, err)
		}
		if back.Retention == nil || back.Retention.RetainFor != 36*time.Hour {
			t.Fatalf("%s retention did not round trip: %+v", name, back.Retention)
		}
	}
}

func TestRetentionJSONRejectsInvalidValues(t *testing.T) {
	for name, input := range map[string]string{
		"negative":   `{"retain_for_seconds":-1}`,
		"fractional": `{"retain_for_seconds":1.5}`,
		"exponent":   `{"retain_for_seconds":1e3}`,
		"overflow":   `{"retain_for_seconds":9223372037}`,
		"huge":       `{"retain_for_seconds":99999999999999999999}`,
		"string":     `{"retain_for_seconds":"abc"}`,
	} {
		var r Retention
		if err := json.Unmarshal([]byte(input), &r); err == nil {
			t.Errorf("%s: expected error, got %+v", name, r)
		}
	}
	var ok Retention
	if err := json.Unmarshal([]byte(`{"retain_for_seconds":9223372036}`), &ok); err != nil || ok.RetainFor != 9223372036*time.Second {
		t.Fatalf("max valid seconds: %+v err=%v", ok, err)
	}
	if _, err := json.Marshal(Retention{RetainFor: -time.Second}); !errors.Is(err, ErrInvalidRetention) {
		t.Fatalf("negative marshal: %v", err)
	}
	if _, err := json.Marshal(Retention{RetainFor: 1500 * time.Millisecond}); !errors.Is(err, ErrInvalidRetention) {
		t.Fatalf("fractional marshal: %v", err)
	}
}
