package asr

import (
	"fmt"
	"time"
)

// StreamCapacityError reports that a new streaming session could not be
// admitted because every configured stream slot is active. Callers can recover
// it through errors.As even when Service wraps it with session context.
type StreamCapacityError struct {
	Limit  int
	Active int
}

func (e *StreamCapacityError) Error() string {
	return fmt.Sprintf("ASR streaming capacity reached: %d active streams (limit %d)", e.Active, e.Limit)
}

// Error wraps ASR operation failures without importing the root package.
type Error struct {
	Op        string
	SessionID string
	Err       error
}

func (e *Error) Error() string {
	if e.SessionID != "" {
		return fmt.Sprintf("ASR %s failed for session %s: %v", e.Op, e.SessionID, e.Err)
	}
	return fmt.Sprintf("ASR %s failed: %v", e.Op, e.Err)
}

func (e *Error) Unwrap() error {
	return e.Err
}

func newError(op string, err error) *Error {
	return &Error{Op: op, Err: err}
}

func newSessionError(op, sessionID string, err error) *Error {
	return &Error{Op: op, SessionID: sessionID, Err: err}
}

// AudioTooLongError reports input longer than one offline decode accepts. The
// model refuses it rather than silently transcribing only a prefix; wrap the
// model in a SegmentedTranscriber (or use NewOfflineTranscriber) to decode
// long audio. Callers can recover it through errors.As.
type AudioTooLongError struct {
	Duration time.Duration
	Limit    time.Duration
}

func (e *AudioTooLongError) Error() string {
	return fmt.Sprintf("audio is %s long, one offline decode accepts at most %s", e.Duration, e.Limit)
}
