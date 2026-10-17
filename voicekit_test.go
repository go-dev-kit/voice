package voicekit

import (
	"errors"
	"os"
	"testing"

	"go.glpx.pro/gdk/voice/diarization"
	"go.glpx.pro/gdk/voice/speaker"
)

func TestVoiceKitCloseIsIdempotent(t *testing.T) {
	vk, err := NewVoiceKit(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		if err := vk.Close(); err != nil {
			t.Fatalf("close %d: %v", i, err)
		}
	}
}

// TestNewVoiceKitClosesStartedComponentsOnFailure starts the speaker manager
// (which locks its Pebble data directory), fails the diarization manager that
// starts after it, and checks the directory can be opened again: NewVoiceKit
// must close what it already started. Needs VOICEKIT_SPEAKER_MODEL.
func TestNewVoiceKitClosesStartedComponentsOnFailure(t *testing.T) {
	modelPath := os.Getenv("VOICEKIT_SPEAKER_MODEL")
	if modelPath == "" {
		t.Skip("set VOICEKIT_SPEAKER_MODEL to run the partial-failure cleanup test")
	}
	dataDir := t.TempDir()
	config := DefaultConfig()
	config.Speaker.ModelPath = modelPath
	config.Speaker.DataDir = dataDir
	config.Diarization.Enabled = true
	config.Diarization.Backend = diarization.BackendBasic

	failure := errors.New("diarization failed to start")
	original := newDiarizationManager
	newDiarizationManager = func(*diarization.DiarizationConfig, diarization.SpeakerDatabase, diarization.EmbeddingExtractor) (*diarization.Manager, error) {
		return nil, failure
	}
	t.Cleanup(func() { newDiarizationManager = original })

	if _, err := NewVoiceKit(config); !errors.Is(err, failure) {
		t.Fatalf("NewVoiceKit = %v, want the diarization failure", err)
	}

	reopened, err := speaker.NewManager(&speaker.Config{ModelPath: modelPath, NumThreads: 1, Provider: "cpu", Threshold: 0.5, DataDir: dataDir})
	if err != nil {
		t.Fatalf("speaker data directory still locked after the failed start: %v", err)
	}
	reopened.Close()
}
