package asr

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"go.glpx.pro/gdk/voice/types"
)

// These tests are hermetic: they exercise the public FinishStream entry point
// against fake models/streams with no native weights, proving the finalization
// path is reachable by sessionID (not only via a VAD endpoint) and that it
// flushes the underlying stream exactly once.

func TestFinishStreamFinalizesBufferedSession(t *testing.T) {
	service := newTestService()
	defer service.Close()

	model := &finalizableTestModel{
		testModel: testModel{name: "fake", language: "en", quantization: "int8"},
	}
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	ctx := context.Background()
	sessionID := "finish-session"

	// Feed a chunk so the session has buffered audio to finalize.
	audio := make([]float32, service.config.ChunkSize)
	if _, err := service.ProcessAudioChunk(ctx, sessionID, audio); err != nil {
		t.Fatalf("process audio chunk failed: %v", err)
	}

	final, err := service.FinishStream(ctx, sessionID)
	if err != nil {
		t.Fatalf("FinishStream failed: %v", err)
	}
	if final == nil {
		t.Fatal("expected a final transcription")
	}
	if final.IsPartial {
		t.Fatal("FinishStream must return a non-partial result")
	}
	if final.Text != "final transcript" {
		t.Fatalf("expected finalized text, got %q", final.Text)
	}
	if model.finishCalls != 1 {
		t.Fatalf("expected FinishAudio called exactly once, got %d", model.finishCalls)
	}
	if len(model.finishAudio) != 0 {
		t.Fatalf("FinishStream must flush without replaying buffered audio, got %d samples", len(model.finishAudio))
	}
}

func TestFinishStreamFlushesNativeStreamOnce(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)

	service := newTestService()
	service.config.DefaultModel = model.Name()
	defer service.Close()
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	ctx := context.Background()
	sessionID := "native-session"

	audio := make([]float32, service.config.ChunkSize)
	if _, err := service.ProcessAudioChunk(ctx, sessionID, audio); err != nil {
		t.Fatalf("process audio chunk failed: %v", err)
	}

	final, err := service.FinishStream(ctx, sessionID)
	if err != nil {
		t.Fatalf("FinishStream failed: %v", err)
	}
	if final == nil || final.IsPartial {
		t.Fatal("expected a non-partial final transcription")
	}
	if final.Text == "" {
		t.Fatal("expected non-empty finalized text from the model")
	}

	if len(recognizer.streams) != 1 {
		t.Fatalf("expected a single native stream, got %d", len(recognizer.streams))
	}
	stream := recognizer.streams[0]
	if stream.inputFinishedCount != 1 {
		t.Fatalf("expected InputFinished called exactly once, got %d", stream.inputFinishedCount)
	}
	if len(stream.samples) != len(audio) {
		t.Fatalf("FinishStream replayed audio: native stream received %d samples, want %d", len(stream.samples), len(audio))
	}
}

func TestFinishStreamUnknownSessionReturnsEmptyFinal(t *testing.T) {
	service := newTestService()
	defer service.Close()

	final, err := service.FinishStream(context.Background(), "never-seen")
	if err != nil {
		t.Fatalf("finalizing an unknown session should not error: %v", err)
	}
	if final == nil {
		t.Fatal("expected an empty final transcription, got nil")
	}
	if final.IsPartial {
		t.Fatal("empty final must be non-partial")
	}
	if final.Text != "" {
		t.Fatalf("expected empty text for an unknown session, got %q", final.Text)
	}
}

func TestFinishStreamWithoutAcceptedSpeechResetsSessionVAD(t *testing.T) {
	service := newTestService()
	defer service.Close()

	session, err := service.streaming.getOrCreateSession("silence-only")
	if err != nil {
		t.Fatalf("create session: %v", err)
	}
	detector := &recordingVADDetector{}
	session.mu.Lock()
	session.vadService = &VADService{detector: detector}
	if err := session.State.Buffer.Append([]float32{0, 0}); err != nil {
		session.mu.Unlock()
		t.Fatalf("buffer silence: %v", err)
	}
	session.mu.Unlock()

	final, err := service.FinishStream(context.Background(), "silence-only")
	if err != nil {
		t.Fatalf("finish silence-only stream: %v", err)
	}
	if final == nil || final.IsPartial || final.Text != "" {
		t.Fatalf("expected empty final result, got %+v", final)
	}
	if !detector.closed {
		t.Fatal("FinishStream must reset session-owned VAD even when no speech reached ASR")
	}
	if session.State.Buffer.Size() != 0 {
		t.Fatalf("session buffer size = %d, want 0", session.State.Buffer.Size())
	}
}

func TestFinishStreamValidatesArguments(t *testing.T) {
	service := newTestService()
	defer service.Close()

	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := service.FinishStream(nil, "session"); err == nil { //nolint:staticcheck // deliberately passing nil ctx to assert validation
		t.Fatal("expected error for nil context")
	}
	if _, err := service.FinishStream(context.Background(), ""); err == nil {
		t.Fatal("expected error for empty sessionID")
	}
}

// TestFinishStreamRaceWithIdleCleanup finalizes sessions while the idle reaper
// (and RemoveSession) close the same sessions concurrently. Before the
// per-session lock this raced on state.ASRState / the native online stream; it
// must run clean under -race.
func TestFinishStreamRaceWithIdleCleanup(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)

	service := newTestServiceWithCapacity(50)
	service.config.DefaultModel = model.Name()
	defer service.Close()
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}
	// Force the reaper to consider every session expired immediately.
	service.streaming.config.IdleTimeout = 0

	ctx := context.Background()
	audio := make([]float32, service.config.ChunkSize)

	var wg sync.WaitGroup
	for i := range 50 {
		sessionID := fmt.Sprintf("race-%d", i)
		// Seed the session with buffered audio and a native ASR stream.
		if _, err := service.ProcessAudioChunk(ctx, sessionID, audio); err != nil && !errors.Is(err, errSessionRetired) {
			t.Fatalf("seed process failed: %v", err)
		}

		wg.Add(3)
		go func() { defer wg.Done(); _, _ = service.FinishStream(ctx, sessionID) }()
		go func() { defer wg.Done(); service.streaming.cleanupExpiredSessions() }()
		go func() { defer wg.Done(); _ = service.streaming.RemoveSession(sessionID) }()
	}
	wg.Wait()
}

var _ types.ASRService = (*Service)(nil)
