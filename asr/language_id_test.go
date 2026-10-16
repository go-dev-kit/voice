package asr

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestSherpaLanguageIdentifierIdentifyDelegatesToNative proves Identify
// forwards the audio/sample rate to the native seam and maps the native
// result's language code into a LanguageResult, without touching real model
// weights.
func TestSherpaLanguageIdentifierIdentifyDelegatesToNative(t *testing.T) {
	native := &fakeLanguageIdentifier{lang: "en"}
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: native}

	audio := []float32{0.1, 0.2, 0.3}
	got, err := l.Identify(context.Background(), audio, 16000)
	if err != nil {
		t.Fatalf("identify failed: %v", err)
	}
	if got.Language != "en" {
		t.Fatalf("unexpected language %q", got.Language)
	}
	if native.lastRate != 16000 {
		t.Fatalf("expected native seam to receive sample rate 16000, got %d", native.lastRate)
	}
	if len(native.lastAudio) != len(audio) {
		t.Fatalf("expected native seam to receive %d samples, got %d", len(audio), len(native.lastAudio))
	}
	if native.streamCloseCount != 1 {
		t.Fatalf("expected stream to be closed exactly once, got %d", native.streamCloseCount)
	}
}

func TestSherpaLanguageIdentifierIdentifyRejectsEmptyAudio(t *testing.T) {
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: &fakeLanguageIdentifier{}}
	if _, err := l.Identify(context.Background(), nil, 16000); err == nil {
		t.Fatal("empty audio should fail before native calls")
	}
	if _, err := l.Identify(context.Background(), []float32{}, 16000); err == nil {
		t.Fatal("empty audio should fail before native calls")
	}
}

func TestSherpaLanguageIdentifierIdentifyRejectsInvalidSampleRate(t *testing.T) {
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: &fakeLanguageIdentifier{}}
	if _, err := l.Identify(context.Background(), []float32{0.1}, 0); err == nil {
		t.Fatal("zero sample rate should fail")
	}
	if _, err := l.Identify(context.Background(), []float32{0.1}, -1); err == nil {
		t.Fatal("negative sample rate should fail")
	}
}

func TestSherpaLanguageIdentifierIdentifyRejectsNilContext(t *testing.T) {
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: &fakeLanguageIdentifier{}}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := l.Identify(nil, []float32{0.1}, 16000); err == nil { //nolint:staticcheck // intentional nil-context guard test
		t.Fatal("nil context should fail")
	}
}

func TestSherpaLanguageIdentifierIdentifyRejectsCancelledContext(t *testing.T) {
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: &fakeLanguageIdentifier{lang: "en"}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := l.Identify(ctx, []float32{0.1}, 16000); err == nil {
		t.Fatal("canceled context should fail")
	}
}

func TestSherpaLanguageIdentifierIdentifyAfterCloseFails(t *testing.T) {
	native := &fakeLanguageIdentifier{lang: "en"}
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: native}

	if err := l.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if _, err := l.Identify(context.Background(), []float32{0.1}, 16000); err == nil {
		t.Fatal("identify after close should fail")
	}
}

func TestSherpaLanguageIdentifierCloseIdempotent(t *testing.T) {
	native := &fakeLanguageIdentifier{}
	l := &SherpaLanguageIdentifier{config: &LanguageIDConfig{}, identifier: native}

	if err := l.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
	if native.closeCount != 1 {
		t.Fatalf("expected exactly one native close, got %d", native.closeCount)
	}
}

func TestNewSherpaLanguageIdentifierRejectsNilConfig(t *testing.T) {
	if _, err := NewSherpaLanguageIdentifier(nil); err == nil {
		t.Fatal("nil config should fail")
	}
}

func TestLanguageIDConfigValidateRequiresEncoderAndDecoderPaths(t *testing.T) {
	config := DefaultLanguageIDConfig()
	if err := config.Validate(); err == nil {
		t.Fatal("missing encoder/decoder paths should fail validation")
	}
}

func TestLanguageIDConfigValidateRejectsMissingFile(t *testing.T) {
	config := DefaultLanguageIDConfig()
	config.EncoderPath = filepath.Join(t.TempDir(), "does-not-exist-encoder.onnx")
	config.DecoderPath = filepath.Join(t.TempDir(), "does-not-exist-decoder.onnx")
	if err := config.Validate(); err == nil {
		t.Fatal("nonexistent model paths should fail validation")
	}
}

func TestLanguageIDConfigValidateAcceptsExistingFiles(t *testing.T) {
	config := DefaultLanguageIDConfig()
	config.EncoderPath = languageIDTempFile(t, "encoder.onnx")
	config.DecoderPath = languageIDTempFile(t, "decoder.onnx")
	if err := config.Validate(); err != nil {
		t.Fatalf("valid config should validate: %v", err)
	}
}

func TestBuildSpokenLanguageIdentificationConfigMapsFields(t *testing.T) {
	config := &LanguageIDConfig{
		EncoderPath:  "encoder.onnx",
		DecoderPath:  "decoder.onnx",
		TailPaddings: 500,
		Provider:     "cpu",
		NumThreads:   3,
		Debug:        true,
	}
	mapped := buildSpokenLanguageIdentificationConfig(config)
	if mapped.Whisper.Encoder != "encoder.onnx" {
		t.Fatalf("encoder path was not mapped, got %q", mapped.Whisper.Encoder)
	}
	if mapped.Whisper.Decoder != "decoder.onnx" {
		t.Fatalf("decoder path was not mapped, got %q", mapped.Whisper.Decoder)
	}
	if mapped.Whisper.TailPaddings != 500 {
		t.Fatalf("expected tail paddings 500, got %d", mapped.Whisper.TailPaddings)
	}
	if mapped.NumThreads != 3 {
		t.Fatalf("expected three threads, got %d", mapped.NumThreads)
	}
	if mapped.Debug != 1 {
		t.Fatalf("expected debug=1, got %d", mapped.Debug)
	}
	if mapped.Provider != "cpu" {
		t.Fatalf("unexpected provider %q", mapped.Provider)
	}
}

func languageIDTempFile(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	return path
}

// fakeLanguageIdentifier is a hermetic fake for nativeLanguageIdentifier that
// returns a known language without loading real model weights.
type fakeLanguageIdentifier struct {
	lang       string
	createErr  error
	computeErr error

	lastRate         int
	lastAudio        []float32
	streamCloseCount int
	closeCount       int
}

func (f *fakeLanguageIdentifier) CreateStream() (lidStream, error) {
	if f.createErr != nil {
		return nil, f.createErr
	}
	return &fakeLIDStream{parent: f}, nil
}

func (f *fakeLanguageIdentifier) Compute(stream lidStream) (string, error) {
	if f.computeErr != nil {
		return "", f.computeErr
	}
	return f.lang, nil
}

func (f *fakeLanguageIdentifier) Close() error {
	f.closeCount++
	return nil
}

type fakeLIDStream struct {
	parent *fakeLanguageIdentifier
}

func (s *fakeLIDStream) AcceptWaveform(sampleRate int, samples []float32) error {
	s.parent.lastRate = sampleRate
	s.parent.lastAudio = samples
	return nil
}

func (s *fakeLIDStream) Close() error {
	s.parent.streamCloseCount++
	return nil
}
