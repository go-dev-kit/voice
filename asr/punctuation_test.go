package asr

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// TestSherpaPunctuationRestoreDelegatesToNative proves Restore forwards the
// raw text to the native seam and returns whatever punctuated string comes
// back, without touching real model weights.
func TestSherpaPunctuationRestoreDelegatesToNative(t *testing.T) {
	native := &fakePunctuation{result: "Hello, world!"}
	p := &SherpaPunctuation{config: &PunctuationConfig{}, punctuation: native}

	got, err := p.Restore(context.Background(), "hello world")
	if err != nil {
		t.Fatalf("restore failed: %v", err)
	}
	if got != "Hello, world!" {
		t.Fatalf("unexpected restored text %q", got)
	}
	if native.lastText != "hello world" {
		t.Fatalf("expected native seam to receive raw text, got %q", native.lastText)
	}
}

func TestSherpaPunctuationRestoreRejectsEmptyText(t *testing.T) {
	p := &SherpaPunctuation{config: &PunctuationConfig{}, punctuation: &fakePunctuation{}}
	if _, err := p.Restore(context.Background(), "   "); err == nil {
		t.Fatal("empty text should fail before native calls")
	}
}

func TestSherpaPunctuationRestoreRejectsNilContext(t *testing.T) {
	p := &SherpaPunctuation{config: &PunctuationConfig{}, punctuation: &fakePunctuation{}}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := p.Restore(nil, "hello"); err == nil { //nolint:staticcheck // intentional nil-context guard test
		t.Fatal("nil context should fail")
	}
}

func TestSherpaPunctuationRestoreRejectsCancelledContext(t *testing.T) {
	p := &SherpaPunctuation{config: &PunctuationConfig{}, punctuation: &fakePunctuation{result: "ignored"}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := p.Restore(ctx, "hello"); err == nil {
		t.Fatal("canceled context should fail")
	}
}

func TestSherpaPunctuationRestoreAfterCloseFails(t *testing.T) {
	native := &fakePunctuation{result: "Hello."}
	p := &SherpaPunctuation{config: &PunctuationConfig{}, punctuation: native}

	if err := p.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if _, err := p.Restore(context.Background(), "hello"); err == nil {
		t.Fatal("restore after close should fail")
	}
}

func TestSherpaPunctuationCloseIdempotent(t *testing.T) {
	native := &fakePunctuation{}
	p := &SherpaPunctuation{config: &PunctuationConfig{}, punctuation: native}

	if err := p.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
	if native.closeCount != 1 {
		t.Fatalf("expected exactly one native close, got %d", native.closeCount)
	}
}

func TestNewSherpaPunctuationRejectsNilConfig(t *testing.T) {
	if _, err := NewSherpaPunctuation(nil); err == nil {
		t.Fatal("nil config should fail")
	}
}

func TestPunctuationConfigValidateRequiresModelPath(t *testing.T) {
	config := DefaultPunctuationConfig()
	if err := config.Validate(); err == nil {
		t.Fatal("missing model path should fail validation")
	}
}

func TestPunctuationConfigValidateRejectsMissingFile(t *testing.T) {
	config := DefaultPunctuationConfig()
	config.ModelPath = filepath.Join(t.TempDir(), "does-not-exist.onnx")
	if err := config.Validate(); err == nil {
		t.Fatal("nonexistent model path should fail validation")
	}
}

func TestPunctuationConfigValidateAcceptsExistingFile(t *testing.T) {
	config := DefaultPunctuationConfig()
	config.ModelPath = punctuationTempFile(t, "model.onnx")
	if err := config.Validate(); err != nil {
		t.Fatalf("valid config should validate: %v", err)
	}
}

func TestBuildOfflinePunctuationConfigMapsFields(t *testing.T) {
	config := &PunctuationConfig{
		ModelPath:  "model.onnx",
		Provider:   "cpu",
		NumThreads: 3,
		Debug:      true,
	}
	mapped := buildOfflinePunctuationConfig(config)
	if mapped.Model.CtTransformer != "model.onnx" {
		t.Fatalf("model path was not mapped, got %q", mapped.Model.CtTransformer)
	}
	if mapped.Model.NumThreads != 3 {
		t.Fatalf("expected three threads, got %d", mapped.Model.NumThreads)
	}
	if mapped.Model.Debug != 1 {
		t.Fatalf("expected debug=1, got %d", mapped.Model.Debug)
	}
	if mapped.Model.Provider != "cpu" {
		t.Fatalf("unexpected provider %q", mapped.Model.Provider)
	}
}

func punctuationTempFile(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	return path
}

type fakePunctuation struct {
	result     string
	lastText   string
	closeCount int
}

func (f *fakePunctuation) AddPunct(text string) string {
	f.lastText = text
	return f.result
}

func (f *fakePunctuation) Close() error {
	f.closeCount++
	return nil
}
