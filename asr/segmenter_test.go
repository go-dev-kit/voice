package asr

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"go.glpx.pro/gdk/voice/audio"
)

func TestNewSpeechSegmenterProviders(t *testing.T) {
	for _, provider := range []string{"", VADProviderNone, VADProviderEnergy} {
		seg, err := NewSpeechSegmenter(&VADConfig{Provider: provider})
		if err != nil || seg != nil {
			t.Fatalf("provider %q: got (%v, %v), want whole-buffer nil segmenter", provider, seg, err)
		}
	}
	if seg, err := NewSpeechSegmenter(nil); err != nil || seg != nil {
		t.Fatalf("nil config: got (%v, %v)", seg, err)
	}
	tests := []struct {
		config VADConfig
		want   string
	}{
		{VADConfig{Provider: "webrtc"}, "unsupported VAD provider"},
		{VADConfig{Provider: "silero"}, "model path is required"},
		{VADConfig{Provider: VADProviderTen, ModelPath: "/does/not/exist.onnx"}, "VAD model path"},
		{VADConfig{Provider: VADProviderSilero, ModelPath: t.TempDir()}, "is a directory"},
	}
	for _, tt := range tests {
		if _, err := NewSpeechSegmenter(&tt.config); err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%+v: got %v, want error containing %q", tt.config, err, tt.want)
		}
	}
}

func TestNewOfflineTranscriberRejectsBadVAD(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Offline.ModelFamily = OfflineFamilySenseVoice
	config.Offline.ModelPath = tempModelFile(t, "model.onnx")
	config.VADProvider = VADProviderSilero
	if _, err := NewOfflineTranscriber(&config); err == nil || !strings.Contains(err.Error(), "VAD model path") {
		t.Fatalf("got %v, want VAD model path error", err)
	}
}

// TestSherpaSegmenterIntegration runs the Silero segmenter on a real
// recording. It needs VOICEKIT_VAD_MODEL and VOICEKIT_LONG_AUDIO_WAV (16 kHz).
func TestSherpaSegmenterIntegration(t *testing.T) {
	model, wavPath := os.Getenv("VOICEKIT_VAD_MODEL"), os.Getenv("VOICEKIT_LONG_AUDIO_WAV")
	if model == "" || wavPath == "" {
		t.Skip("set VOICEKIT_VAD_MODEL and VOICEKIT_LONG_AUDIO_WAV to run the Silero segmenter integration")
	}
	seg, err := NewSpeechSegmenter(&VADConfig{Provider: VADProviderSilero, ModelPath: model})
	if err != nil {
		t.Fatal(err)
	}
	defer seg.Close()
	samples := readMonoWAV(t, wavPath)

	spans, err := seg.SpeechSpans(context.Background(), samples, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(spans) == 0 {
		t.Fatal("no speech found")
	}
	var speech int
	for i, s := range spans {
		if s.Start < 0 || s.End > len(samples) || s.Start >= s.End || (i > 0 && s.Start < spans[i-1].End) {
			t.Fatalf("span %d invalid or unordered: %+v", i, s)
		}
		speech += s.End - s.Start
	}
	t.Logf("%d spans, %.1f s of speech in %.1f s", len(spans), float64(speech)/16000, float64(len(samples))/16000)

	// The same audio at 48 kHz must give the same spans in 48 kHz samples.
	up, err := audio.NewResampler(&audio.ResampleConfig{Method: audio.MethodSinc, FilterLength: 16}).Resample(samples, 16000, 48000)
	if err != nil {
		t.Fatal(err)
	}
	spans48, err := seg.SpeechSpans(context.Background(), up, 48000)
	if err != nil {
		t.Fatal(err)
	}
	if len(spans48) != len(spans) {
		t.Fatalf("48 kHz input found %d spans, 16 kHz found %d", len(spans48), len(spans))
	}
	for i := range spans {
		if d := spans48[i].Start/3 - spans[i].Start; d > 1600 || d < -1600 {
			t.Fatalf("span %d start moved by %d samples at 48 kHz", i, d)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := seg.SpeechSpans(ctx, samples, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled: got %v", err)
	}
	if spans, err := seg.SpeechSpans(context.Background(), nil, 16000); err != nil || spans != nil {
		t.Fatalf("empty audio: got (%v, %v)", spans, err)
	}
	if _, err := seg.SpeechSpans(context.Background(), samples, 0); err == nil {
		t.Fatal("expected error for zero sample rate")
	}
}

func BenchmarkSherpaSegmenter(b *testing.B) {
	model, wavPath := os.Getenv("VOICEKIT_VAD_MODEL"), os.Getenv("VOICEKIT_LONG_AUDIO_WAV")
	if model == "" || wavPath == "" {
		b.Skip("set VOICEKIT_VAD_MODEL and VOICEKIT_LONG_AUDIO_WAV")
	}
	seg, err := NewSpeechSegmenter(&VADConfig{Provider: VADProviderSilero, ModelPath: model})
	if err != nil {
		b.Fatal(err)
	}
	defer seg.Close()
	data, err := os.ReadFile(wavPath)
	if err != nil {
		b.Fatal(err)
	}
	samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(samples) * 4))
	for b.Loop() {
		if _, err := seg.SpeechSpans(context.Background(), samples, 16000); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(samples))/16000*float64(b.N)/b.Elapsed().Seconds(), "audio-s/s")
}
