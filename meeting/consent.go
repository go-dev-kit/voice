package meeting

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

var (
	ErrInvalidConsent   = errors.New("meeting consent is invalid")
	ErrInvalidRetention = errors.New("meeting retention is invalid")
	ErrConsentRequired  = errors.New("meeting export requires consent")
)

// ConsentStatus captures whether recording and processing consent was given.
type ConsentStatus string

const (
	ConsentUnknown   ConsentStatus = "unknown"
	ConsentObtained  ConsentStatus = "obtained"
	ConsentDeclined  ConsentStatus = "declined"
	ConsentWithdrawn ConsentStatus = "withdrawn"
)

// Valid reports whether the status is a known value. The empty status is
// treated as ConsentUnknown and is valid.
func (s ConsentStatus) Valid() bool {
	switch s {
	case "", ConsentUnknown, ConsentObtained, ConsentDeclined, ConsentWithdrawn:
		return true
	default:
		return false
	}
}

// Consent records meeting-level consent and optional per-participant records.
type Consent struct {
	Status       ConsentStatus        `json:"status"`
	Basis        string               `json:"basis,omitempty"`
	ObtainedAt   time.Time            `json:"obtained_at,omitzero"`
	Participants []ParticipantConsent `json:"participants,omitempty"`
}

// ParticipantConsent records consent for one meeting participant.
type ParticipantConsent struct {
	ParticipantID string        `json:"participant_id"`
	Status        ConsentStatus `json:"status"`
	Basis         string        `json:"basis,omitempty"`
	ObtainedAt    time.Time     `json:"obtained_at,omitzero"`
}

// Validate checks consent consistency. Participant references are checked
// against the meeting by Meeting.Validate.
func (c Consent) Validate() error {
	if !c.Status.Valid() {
		return fmt.Errorf("%w: unknown status %q", ErrInvalidConsent, c.Status)
	}
	seen := make(map[string]struct{}, len(c.Participants))
	for index, record := range c.Participants {
		id := strings.TrimSpace(record.ParticipantID)
		if id == "" {
			return fmt.Errorf("%w: participant record %d: participant id is required", ErrInvalidConsent, index)
		}
		if _, exists := seen[id]; exists {
			return fmt.Errorf("%w: participant record %d: duplicate participant id %q", ErrInvalidConsent, index, id)
		}
		seen[id] = struct{}{}
		if !record.Status.Valid() {
			return fmt.Errorf("%w: participant record %d: unknown status %q", ErrInvalidConsent, index, record.Status)
		}
	}
	return nil
}

// Obtained reports whether consent covers the meeting: the meeting-level
// status is obtained and, when per-participant records exist, every record is
// obtained and every meeting participant has a record.
func (c Consent) Obtained(participants []Participant) bool {
	if c.Status != ConsentObtained {
		return false
	}
	if len(c.Participants) == 0 {
		return true
	}
	obtained := make(map[string]bool, len(c.Participants))
	for _, record := range c.Participants {
		if record.Status != ConsentObtained {
			return false
		}
		obtained[strings.TrimSpace(record.ParticipantID)] = true
	}
	for _, participant := range participants {
		if !obtained[strings.TrimSpace(participant.ID)] {
			return false
		}
	}
	return true
}

// Retention describes how long meeting artifacts may be kept.
type Retention struct {
	Policy      string        `json:"policy,omitempty"`
	RetainUntil time.Time     `json:"retain_until,omitzero"`
	RetainFor   time.Duration `json:"retain_for,omitempty"`
	LegalHold   bool          `json:"legal_hold,omitempty"`
}

// Validate checks retention consistency.
func (r Retention) Validate() error {
	if r.RetainFor < 0 {
		return fmt.Errorf("%w: retain_for is negative", ErrInvalidRetention)
	}
	if !r.RetainUntil.IsZero() && r.RetainFor > 0 {
		return fmt.Errorf("%w: retain_until and retain_for are mutually exclusive", ErrInvalidRetention)
	}
	return nil
}

// DeletionDue returns when the meeting becomes due for deletion. It reports
// false when no retention is set, no deadline is configured, or a legal hold
// applies. A RetainFor duration counts from the meeting end, falling back to
// its start, then its creation time.
func (m Meeting) DeletionDue() (time.Time, bool) {
	r := m.Retention
	if r == nil || r.LegalHold {
		return time.Time{}, false
	}
	if !r.RetainUntil.IsZero() {
		return r.RetainUntil, true
	}
	if r.RetainFor <= 0 {
		return time.Time{}, false
	}
	for _, base := range []time.Time{m.EndedAt, m.StartedAt, m.CreatedAt} {
		if !base.IsZero() {
			return base.Add(r.RetainFor), true
		}
	}
	return time.Time{}, false
}

// IsDeletionDue reports whether the retention deadline has passed at now.
func (m Meeting) IsDeletionDue(now time.Time) bool {
	due, ok := m.DeletionDue()
	return ok && !now.Before(due)
}

func (m Meeting) validateConsentRetention() error {
	if m.Consent != nil {
		if err := m.Consent.Validate(); err != nil {
			return err
		}
		ids := make(map[string]struct{}, len(m.Participants))
		for _, participant := range m.Participants {
			ids[strings.TrimSpace(participant.ID)] = struct{}{}
		}
		for index, record := range m.Consent.Participants {
			if _, ok := ids[strings.TrimSpace(record.ParticipantID)]; !ok {
				return fmt.Errorf("%w: participant record %d: unknown participant %q", ErrInvalidConsent, index, record.ParticipantID)
			}
		}
	}
	if m.Retention != nil {
		return m.Retention.Validate()
	}
	return nil
}
