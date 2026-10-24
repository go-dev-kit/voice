// Package meeting defines product-level meeting intelligence artifacts.
package meeting

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

var (
	ErrMeetingIDRequired = errors.New("meeting id is required")
	ErrInvalidTiming     = errors.New("meeting timing is invalid")
)

// SourceKind identifies how meeting audio or transcript data entered the system.
type SourceKind string

const (
	SourceUnknown         SourceKind = "unknown"
	SourceUpload          SourceKind = "upload"
	SourceMicrophone      SourceKind = "microphone"
	SourceDesktopAudio    SourceKind = "desktop_audio"
	SourceMeetingPlatform SourceKind = "meeting_platform"
)

// ActionItemStatus captures the lifecycle state of a generated or reviewed task.
type ActionItemStatus string

const (
	ActionItemProposed  ActionItemStatus = "proposed"
	ActionItemOpen      ActionItemStatus = "open"
	ActionItemDone      ActionItemStatus = "done"
	ActionItemDismissed ActionItemStatus = "dismissed"
)

// Source describes the origin of a meeting artifact without storing raw secrets.
type Source struct {
	Kind      SourceKind `json:"kind"`
	Provider  string     `json:"provider,omitempty"`
	Reference string     `json:"reference,omitempty"`
}

// Meeting is the product-level aggregate for transcript, speakers, and notes.
type Meeting struct {
	ID           string           `json:"id"`
	Title        string           `json:"title,omitempty"`
	Language     string           `json:"language,omitempty"`
	Source       Source           `json:"source"`
	Participants []Participant    `json:"participants,omitempty"`
	Transcript   []TranscriptTurn `json:"transcript,omitempty"`
	Summary      MeetingSummary   `json:"summary,omitempty"`
	Redacted     bool             `json:"redacted,omitempty"`
	Consent      *Consent         `json:"consent,omitempty"`
	Retention    *Retention       `json:"retention,omitempty"`
	StartedAt    time.Time        `json:"started_at,omitempty"`
	EndedAt      time.Time        `json:"ended_at,omitempty"`
	CreatedAt    time.Time        `json:"created_at"`
	ProcessedAt  time.Time        `json:"processed_at,omitempty"`
}

// Participant connects a human-readable attendee to a diarization speaker ID.
type Participant struct {
	ID          string `json:"id"`
	DisplayName string `json:"display_name,omitempty"`
	Role        string `json:"role,omitempty"`
	SpeakerID   string `json:"speaker_id,omitempty"`
}

// TranscriptTurn is one speaker-attributed transcript segment.
type TranscriptTurn struct {
	ID            string        `json:"id,omitempty"`
	SpeakerID     string        `json:"speaker_id"`
	ParticipantID string        `json:"participant_id,omitempty"`
	SpeakerLabel  string        `json:"speaker_label,omitempty"`
	Text          string        `json:"text"`
	StartTime     time.Duration `json:"start_time"`
	EndTime       time.Duration `json:"end_time"`
	Confidence    float32       `json:"confidence"`
	Words         []Word        `json:"words,omitempty"`
}

// Word is a word-level transcript token with relative meeting timing.
type Word struct {
	Text       string        `json:"text"`
	StartTime  time.Duration `json:"start_time"`
	EndTime    time.Duration `json:"end_time"`
	Confidence float32       `json:"confidence"`
}

// MeetingSummary holds reviewed or generated notes for a meeting.
type MeetingSummary struct {
	Overview      string       `json:"overview,omitempty"`
	Topics        []Topic      `json:"topics,omitempty"`
	Decisions     []Decision   `json:"decisions,omitempty"`
	ActionItems   []ActionItem `json:"action_items,omitempty"`
	FollowUps     []FollowUp   `json:"follow_ups,omitempty"`
	OpenQuestions []Question   `json:"open_questions,omitempty"`
	Risks         []Risk       `json:"risks,omitempty"`
}

// Topic identifies a major discussion area.
type Topic struct {
	ID        string        `json:"id,omitempty"`
	Title     string        `json:"title"`
	Summary   string        `json:"summary,omitempty"`
	StartTime time.Duration `json:"start_time,omitempty"`
	EndTime   time.Duration `json:"end_time,omitempty"`
}

// Decision captures an explicit outcome or agreement from a meeting.
type Decision struct {
	ID          string        `json:"id,omitempty"`
	Text        string        `json:"text"`
	OwnerID     string        `json:"owner_id,omitempty"`
	StartTime   time.Duration `json:"start_time,omitempty"`
	EvidenceIDs []string      `json:"evidence_ids,omitempty"`
}

// ActionItem captures a task extracted from a meeting.
type ActionItem struct {
	ID          string           `json:"id,omitempty"`
	Text        string           `json:"text"`
	OwnerID     string           `json:"owner_id,omitempty"`
	OwnerName   string           `json:"owner_name,omitempty"`
	DueAt       time.Time        `json:"due_at,omitempty"`
	Status      ActionItemStatus `json:"status,omitempty"`
	EvidenceIDs []string         `json:"evidence_ids,omitempty"`
}

// FollowUp captures outbound communication or next-step work after the meeting.
type FollowUp struct {
	ID          string   `json:"id,omitempty"`
	Text        string   `json:"text"`
	RecipientID string   `json:"recipient_id,omitempty"`
	Channel     string   `json:"channel,omitempty"`
	EvidenceIDs []string `json:"evidence_ids,omitempty"`
}

// Question captures unresolved discussion points.
type Question struct {
	ID          string   `json:"id,omitempty"`
	Text        string   `json:"text"`
	OwnerID     string   `json:"owner_id,omitempty"`
	EvidenceIDs []string `json:"evidence_ids,omitempty"`
}

// Risk captures a concern, blocker, or dependency raised in the meeting.
type Risk struct {
	ID          string   `json:"id,omitempty"`
	Text        string   `json:"text"`
	Severity    string   `json:"severity,omitempty"`
	EvidenceIDs []string `json:"evidence_ids,omitempty"`
}

// Validate checks meeting artifact consistency without requiring summary data.
func (m Meeting) Validate() error {
	if strings.TrimSpace(m.ID) == "" {
		return ErrMeetingIDRequired
	}
	if !m.StartedAt.IsZero() && !m.EndedAt.IsZero() && m.EndedAt.Before(m.StartedAt) {
		return fmt.Errorf("%w: ended_at before started_at", ErrInvalidTiming)
	}
	if err := validateParticipants(m.Participants); err != nil {
		return err
	}
	if err := m.validateConsentRetention(); err != nil {
		return err
	}
	for index, turn := range m.Transcript {
		if err := turn.Validate(); err != nil {
			return fmt.Errorf("transcript turn %d: %w", index, err)
		}
	}
	return nil
}

// Duration returns the best-known meeting duration.
func (m Meeting) Duration() time.Duration {
	if !m.StartedAt.IsZero() && !m.EndedAt.IsZero() && m.EndedAt.After(m.StartedAt) {
		return m.EndedAt.Sub(m.StartedAt)
	}
	var maxEnd time.Duration
	for _, turn := range m.Transcript {
		if turn.EndTime > maxEnd {
			maxEnd = turn.EndTime
		}
	}
	return maxEnd
}

// SpeakerTalkTime returns total transcript duration by diarization speaker ID.
func (m Meeting) SpeakerTalkTime() map[string]time.Duration {
	talkTime := make(map[string]time.Duration)
	for _, turn := range m.Transcript {
		if turn.SpeakerID == "" {
			continue
		}
		talkTime[turn.SpeakerID] += turn.Duration()
	}
	return talkTime
}

// ParticipantForSpeaker returns the participant associated with a speaker ID.
func (m Meeting) ParticipantForSpeaker(speakerID string) (Participant, bool) {
	for _, participant := range m.Participants {
		if participant.SpeakerID == speakerID {
			return participant, true
		}
	}
	return Participant{}, false
}

// Validate checks transcript turn consistency.
func (t TranscriptTurn) Validate() error {
	if strings.TrimSpace(t.SpeakerID) == "" {
		return errors.New("speaker id is required")
	}
	if t.EndTime < t.StartTime {
		return fmt.Errorf("%w: turn end before start", ErrInvalidTiming)
	}
	for index, word := range t.Words {
		if err := word.Validate(); err != nil {
			return fmt.Errorf("word %d: %w", index, err)
		}
	}
	return nil
}

// Duration returns the turn duration.
func (t TranscriptTurn) Duration() time.Duration {
	if t.EndTime <= t.StartTime {
		return 0
	}
	return t.EndTime - t.StartTime
}

// Validate checks word timing consistency.
func (w Word) Validate() error {
	if strings.TrimSpace(w.Text) == "" {
		return errors.New("word text is required")
	}
	if w.EndTime < w.StartTime {
		return fmt.Errorf("%w: word end before start", ErrInvalidTiming)
	}
	return nil
}

func validateParticipants(participants []Participant) error {
	participantIDs := make(map[string]struct{}, len(participants))
	speakerIDs := make(map[string]struct{}, len(participants))
	for index, participant := range participants {
		id := strings.TrimSpace(participant.ID)
		if id == "" {
			return fmt.Errorf("participant %d: id is required", index)
		}
		if _, exists := participantIDs[id]; exists {
			return fmt.Errorf("participant %d: duplicate id %q", index, id)
		}
		participantIDs[id] = struct{}{}

		speakerID := strings.TrimSpace(participant.SpeakerID)
		if speakerID == "" {
			continue
		}
		if _, exists := speakerIDs[speakerID]; exists {
			return fmt.Errorf("participant %d: duplicate speaker id %q", index, speakerID)
		}
		speakerIDs[speakerID] = struct{}{}
	}
	return nil
}
