package meeting

import (
	"errors"
	"regexp"
	"sort"
	"strings"
)

var ErrRedactorRequired = errors.New("meeting redactor is required")

const defaultRedactionReplacement = "[REDACTED]"

var (
	emailPattern = regexp.MustCompile(`(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b`)
	phonePattern = regexp.MustCompile(`(?:\+?\d[\d\s().-]{7,}\d)`)
	urlPattern   = regexp.MustCompile(`(?i)\b(?:https?://|www\.)[^\s<>()]+`)
)

// Redactor redacts sensitive text and returns match metadata without retaining raw values.
type Redactor interface {
	Redact(field string, text string) (string, []RedactionMatch, error)
}

// RedactionPolicy configures the built-in pattern redactor.
type RedactionPolicy struct {
	RedactEmails bool     `json:"redact_emails"`
	RedactPhones bool     `json:"redact_phones"`
	RedactURLs   bool     `json:"redact_urls"`
	Terms        []string `json:"terms,omitempty"`
	Replacement  string   `json:"replacement,omitempty"`
}

// RedactionMatch records one redaction without exposing the matched sensitive value.
type RedactionMatch struct {
	Field       string `json:"field"`
	Kind        string `json:"kind"`
	Start       int    `json:"start"`
	End         int    `json:"end"`
	Replacement string `json:"replacement"`
}

// RedactionReport summarizes redaction performed on a meeting.
type RedactionReport struct {
	Redacted bool             `json:"redacted"`
	Matches  []RedactionMatch `json:"matches,omitempty"`
}

// PatternRedactor redacts emails, phone-like numbers, URLs, and configured terms.
type PatternRedactor struct {
	policy RedactionPolicy
	terms  []string
}

// DefaultRedactionPolicy returns a conservative stdlib-only meeting redaction policy.
func DefaultRedactionPolicy() RedactionPolicy {
	return RedactionPolicy{
		RedactEmails: true,
		RedactPhones: true,
		RedactURLs:   true,
		Replacement:  defaultRedactionReplacement,
	}
}

// NewPatternRedactor creates the built-in stdlib redactor.
func NewPatternRedactor(policy RedactionPolicy) *PatternRedactor {
	policy = normalizeRedactionPolicy(policy)
	return &PatternRedactor{
		policy: policy,
		terms:  cleanStringList(policy.Terms),
	}
}

// Redact redacts one string field and returns redaction metadata.
func (r *PatternRedactor) Redact(field string, text string) (string, []RedactionMatch, error) {
	if r == nil {
		return "", nil, ErrRedactorRequired
	}
	if text == "" {
		return text, nil, nil
	}

	candidates := r.findMatches(field, text)
	if len(candidates) == 0 {
		return text, nil, nil
	}
	matches := selectNonOverlappingMatches(candidates)
	return replaceMatches(text, matches, r.policy.Replacement), matches, nil
}

// Redact returns a copy of the meeting with sensitive text redacted.
func (m Meeting) Redact(redactor Redactor) (Meeting, RedactionReport, error) {
	if redactor == nil {
		return Meeting{}, RedactionReport{}, ErrRedactorRequired
	}
	if err := m.Validate(); err != nil {
		return Meeting{}, RedactionReport{}, err
	}
	m = cloneMeeting(m)

	report := RedactionReport{}
	var err error
	m.Title, report, err = redactField(redactor, report, "meeting.title", m.Title)
	if err != nil {
		return Meeting{}, RedactionReport{}, err
	}
	m.Source.Provider, report, err = redactField(redactor, report, "meeting.source.provider", m.Source.Provider)
	if err != nil {
		return Meeting{}, RedactionReport{}, err
	}
	m.Source.Reference, report, err = redactField(redactor, report, "meeting.source.reference", m.Source.Reference)
	if err != nil {
		return Meeting{}, RedactionReport{}, err
	}

	for index := range m.Participants {
		prefix := "meeting.participants." + m.Participants[index].ID
		m.Participants[index].DisplayName, report, err = redactField(redactor, report, prefix+".display_name", m.Participants[index].DisplayName)
		if err != nil {
			return Meeting{}, RedactionReport{}, err
		}
		m.Participants[index].Role, report, err = redactField(redactor, report, prefix+".role", m.Participants[index].Role)
		if err != nil {
			return Meeting{}, RedactionReport{}, err
		}
	}

	for index := range m.Transcript {
		prefix := "meeting.transcript." + turnFieldID(m.Transcript[index], index)
		m.Transcript[index].SpeakerLabel, report, err = redactField(redactor, report, prefix+".speaker_label", m.Transcript[index].SpeakerLabel)
		if err != nil {
			return Meeting{}, RedactionReport{}, err
		}
		m.Transcript[index].Text, report, err = redactField(redactor, report, prefix+".text", m.Transcript[index].Text)
		if err != nil {
			return Meeting{}, RedactionReport{}, err
		}
		for wordIndex := range m.Transcript[index].Words {
			field := prefix + ".words." + intString(wordIndex) + ".text"
			m.Transcript[index].Words[wordIndex].Text, report, err = redactField(redactor, report, field, m.Transcript[index].Words[wordIndex].Text)
			if err != nil {
				return Meeting{}, RedactionReport{}, err
			}
		}
	}

	if m.Consent != nil {
		m.Consent.Basis, report, err = redactField(redactor, report, "meeting.consent.basis", m.Consent.Basis)
		if err != nil {
			return Meeting{}, RedactionReport{}, err
		}
		for index := range m.Consent.Participants {
			field := "meeting.consent.participants[" + intString(index) + "].basis"
			m.Consent.Participants[index].Basis, report, err = redactField(redactor, report, field, m.Consent.Participants[index].Basis)
			if err != nil {
				return Meeting{}, RedactionReport{}, err
			}
		}
	}

	m.Summary, report, err = redactSummary(redactor, report, m.Summary)
	if err != nil {
		return Meeting{}, RedactionReport{}, err
	}

	m.Redacted = true
	report.Redacted = true
	return m, report, nil
}

func normalizeRedactionPolicy(policy RedactionPolicy) RedactionPolicy {
	if policy.Replacement == "" {
		policy.Replacement = defaultRedactionReplacement
	}
	return policy
}

func (r *PatternRedactor) findMatches(field string, text string) []RedactionMatch {
	matches := make([]RedactionMatch, 0)
	replacement := r.policy.Replacement
	if r.policy.RedactEmails {
		matches = append(matches, regexpMatches(field, "email", replacement, text, emailPattern)...)
	}
	if r.policy.RedactURLs {
		matches = append(matches, regexpMatches(field, "url", replacement, text, urlPattern)...)
	}
	if r.policy.RedactPhones {
		matches = append(matches, regexpMatches(field, "phone", replacement, text, phonePattern)...)
	}
	for _, term := range r.terms {
		matches = append(matches, literalMatches(field, "term", replacement, text, term)...)
	}
	return matches
}

func regexpMatches(field string, kind string, replacement string, text string, pattern *regexp.Regexp) []RedactionMatch {
	indexes := pattern.FindAllStringIndex(text, -1)
	matches := make([]RedactionMatch, 0, len(indexes))
	for _, index := range indexes {
		matches = append(matches, RedactionMatch{
			Field:       field,
			Kind:        kind,
			Start:       index[0],
			End:         index[1],
			Replacement: replacement,
		})
	}
	return matches
}

func literalMatches(field string, kind string, replacement string, text string, term string) []RedactionMatch {
	term = strings.TrimSpace(term)
	if term == "" {
		return nil
	}
	lowerText := strings.ToLower(text)
	lowerTerm := strings.ToLower(term)
	matches := make([]RedactionMatch, 0)
	offset := 0
	for {
		index := strings.Index(lowerText[offset:], lowerTerm)
		if index < 0 {
			break
		}
		start := offset + index
		end := start + len(term)
		matches = append(matches, RedactionMatch{
			Field:       field,
			Kind:        kind,
			Start:       start,
			End:         end,
			Replacement: replacement,
		})
		offset = end
	}
	return matches
}

func selectNonOverlappingMatches(matches []RedactionMatch) []RedactionMatch {
	sort.SliceStable(matches, func(i int, j int) bool {
		if matches[i].Start == matches[j].Start {
			return matches[i].End > matches[j].End
		}
		return matches[i].Start < matches[j].Start
	})

	selected := make([]RedactionMatch, 0, len(matches))
	lastEnd := -1
	for _, match := range matches {
		if match.Start < lastEnd {
			continue
		}
		selected = append(selected, match)
		lastEnd = match.End
	}
	return selected
}

func replaceMatches(text string, matches []RedactionMatch, replacement string) string {
	var b strings.Builder
	last := 0
	for _, match := range matches {
		b.WriteString(text[last:match.Start])
		b.WriteString(replacement)
		last = match.End
	}
	b.WriteString(text[last:])
	return b.String()
}

func redactField(redactor Redactor, report RedactionReport, field string, value string) (string, RedactionReport, error) {
	redacted, matches, err := redactor.Redact(field, value)
	if err != nil {
		return "", RedactionReport{}, err
	}
	report.Matches = append(report.Matches, matches...)
	return redacted, report, nil
}

func redactSummary(redactor Redactor, report RedactionReport, summary MeetingSummary) (MeetingSummary, RedactionReport, error) {
	var err error
	summary.Overview, report, err = redactField(redactor, report, "meeting.summary.overview", summary.Overview)
	if err != nil {
		return MeetingSummary{}, RedactionReport{}, err
	}
	for index := range summary.Topics {
		prefix := "meeting.summary.topics." + artifactFieldID(summary.Topics[index].ID, index)
		summary.Topics[index].Title, report, err = redactField(redactor, report, prefix+".title", summary.Topics[index].Title)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
		summary.Topics[index].Summary, report, err = redactField(redactor, report, prefix+".summary", summary.Topics[index].Summary)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
	}
	for index := range summary.Decisions {
		field := "meeting.summary.decisions." + artifactFieldID(summary.Decisions[index].ID, index) + ".text"
		summary.Decisions[index].Text, report, err = redactField(redactor, report, field, summary.Decisions[index].Text)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
	}
	for index := range summary.ActionItems {
		prefix := "meeting.summary.action_items." + artifactFieldID(summary.ActionItems[index].ID, index)
		summary.ActionItems[index].Text, report, err = redactField(redactor, report, prefix+".text", summary.ActionItems[index].Text)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
		summary.ActionItems[index].OwnerName, report, err = redactField(redactor, report, prefix+".owner_name", summary.ActionItems[index].OwnerName)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
	}
	for index := range summary.FollowUps {
		field := "meeting.summary.follow_ups." + artifactFieldID(summary.FollowUps[index].ID, index) + ".text"
		summary.FollowUps[index].Text, report, err = redactField(redactor, report, field, summary.FollowUps[index].Text)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
	}
	for index := range summary.OpenQuestions {
		field := "meeting.summary.open_questions." + artifactFieldID(summary.OpenQuestions[index].ID, index) + ".text"
		summary.OpenQuestions[index].Text, report, err = redactField(redactor, report, field, summary.OpenQuestions[index].Text)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
	}
	for index := range summary.Risks {
		field := "meeting.summary.risks." + artifactFieldID(summary.Risks[index].ID, index) + ".text"
		summary.Risks[index].Text, report, err = redactField(redactor, report, field, summary.Risks[index].Text)
		if err != nil {
			return MeetingSummary{}, RedactionReport{}, err
		}
	}
	return summary, report, nil
}

func turnFieldID(turn TranscriptTurn, index int) string {
	if strings.TrimSpace(turn.ID) != "" {
		return strings.TrimSpace(turn.ID)
	}
	return intString(index)
}

func artifactFieldID(id string, index int) string {
	if strings.TrimSpace(id) != "" {
		return strings.TrimSpace(id)
	}
	return intString(index)
}

func intString(value int) string {
	if value == 0 {
		return "0"
	}
	var digits [20]byte
	i := len(digits)
	for value > 0 {
		i--
		digits[i] = byte('0' + value%10)
		value /= 10
	}
	return string(digits[i:])
}

func cloneMeeting(m Meeting) Meeting {
	m.Participants = append([]Participant(nil), m.Participants...)
	m.Transcript = cloneTranscriptTurns(m.Transcript)
	m.Summary = cloneMeetingSummary(m.Summary)
	m.Consent = cloneConsent(m.Consent)
	m.Retention = cloneRetention(m.Retention)
	return m
}

func cloneConsent(c *Consent) *Consent {
	if c == nil {
		return nil
	}
	cloned := *c
	cloned.Participants = append([]ParticipantConsent(nil), c.Participants...)
	return &cloned
}

func cloneRetention(r *Retention) *Retention {
	if r == nil {
		return nil
	}
	cloned := *r
	return &cloned
}

func cloneTranscriptTurns(turns []TranscriptTurn) []TranscriptTurn {
	if len(turns) == 0 {
		return nil
	}
	cloned := make([]TranscriptTurn, len(turns))
	copy(cloned, turns)
	for index := range cloned {
		cloned[index].Words = append([]Word(nil), cloned[index].Words...)
	}
	return cloned
}

func cloneMeetingSummary(summary MeetingSummary) MeetingSummary {
	summary.Topics = append([]Topic(nil), summary.Topics...)
	summary.Decisions = cloneDecisions(summary.Decisions)
	summary.ActionItems = cloneActionItems(summary.ActionItems)
	summary.FollowUps = cloneFollowUps(summary.FollowUps)
	summary.OpenQuestions = cloneQuestions(summary.OpenQuestions)
	summary.Risks = cloneRisks(summary.Risks)
	return summary
}

func cloneDecisions(items []Decision) []Decision {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]Decision, len(items))
	copy(cloned, items)
	for index := range cloned {
		cloned[index].EvidenceIDs = append([]string(nil), cloned[index].EvidenceIDs...)
	}
	return cloned
}

func cloneActionItems(items []ActionItem) []ActionItem {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]ActionItem, len(items))
	copy(cloned, items)
	for index := range cloned {
		cloned[index].EvidenceIDs = append([]string(nil), cloned[index].EvidenceIDs...)
	}
	return cloned
}

func cloneFollowUps(items []FollowUp) []FollowUp {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]FollowUp, len(items))
	copy(cloned, items)
	for index := range cloned {
		cloned[index].EvidenceIDs = append([]string(nil), cloned[index].EvidenceIDs...)
	}
	return cloned
}

func cloneQuestions(items []Question) []Question {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]Question, len(items))
	copy(cloned, items)
	for index := range cloned {
		cloned[index].EvidenceIDs = append([]string(nil), cloned[index].EvidenceIDs...)
	}
	return cloned
}

func cloneRisks(items []Risk) []Risk {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]Risk, len(items))
	copy(cloned, items)
	for index := range cloned {
		cloned[index].EvidenceIDs = append([]string(nil), cloned[index].EvidenceIDs...)
	}
	return cloned
}
