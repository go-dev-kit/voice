package asr

import (
	"context"
	"os"
	"strings"
	"testing"

	"go.glpx.pro/gdk/voice/audio"
)

// TestWhisperAutoDetectIntegration decodes a real recording with an empty
// Whisper language and checks that the detected code comes back on the result.
// It needs a multilingual Whisper export:
//
//	VOICEKIT_WHISPER_ENCODER, VOICEKIT_WHISPER_DECODER, VOICEKIT_WHISPER_TOKENS
//	VOICEKIT_WHISPER_WAV         - 16 kHz mono WAV
//	VOICEKIT_WHISPER_EXPECT_LANG - language code the recording is spoken in
func TestWhisperAutoDetectIntegration(t *testing.T) {
	encoder, decoder, tokens := os.Getenv("VOICEKIT_WHISPER_ENCODER"), os.Getenv("VOICEKIT_WHISPER_DECODER"), os.Getenv("VOICEKIT_WHISPER_TOKENS")
	wav, want := os.Getenv("VOICEKIT_WHISPER_WAV"), os.Getenv("VOICEKIT_WHISPER_EXPECT_LANG")
	if encoder == "" || decoder == "" || tokens == "" || wav == "" || want == "" {
		t.Skip("set VOICEKIT_WHISPER_* env vars to run the Whisper auto-detect integration")
	}
	config := DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOffline
	config.DefaultModel = "integration_whisper_auto"
	config.Offline.ModelFamily = OfflineFamilyWhisper
	config.Offline.EncoderPath, config.Offline.DecoderPath, config.Offline.TokensPath = encoder, decoder, tokens

	for _, language := range []string{"", "auto"} {
		config.Offline.Language = language
		model, err := NewSherpaOfflineModel(&config)
		if err != nil {
			t.Fatalf("language %q: create model: %v", language, err)
		}
		data, err := os.ReadFile(wav)
		if err != nil {
			t.Fatal(err)
		}
		samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
		if err != nil {
			t.Fatal(err)
		}
		result, err := model.Transcribe(context.Background(), samples, 16000)
		_ = model.Close()
		if err != nil {
			t.Fatalf("language %q: transcribe: %v", language, err)
		}
		if strings.TrimSpace(result.Text) == "" || result.Language != want {
			t.Fatalf("language %q: got language %q text %q, want language %q", language, result.Language, result.Text, want)
		}
		t.Logf("language %q -> detected %q: %s", language, result.Language, result.Text)
	}
}
