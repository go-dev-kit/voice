package asr

import (
	"context"
	"errors"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"go.glpx.pro/gdk/voice/audio"
)

// TestLongAudioIntegration decodes a long recording through
// NewOfflineTranscriber with a real model and Silero VAD, and checks that the
// transcript covers the whole reference. It reads the VOICEKIT_OFFLINE_ASR_*
// model variables (see TestSherpaOfflineModelIntegration) plus:
//
//	VOICEKIT_OFFLINE_ASR_MODEL_TYPE - sherpa model_type (nemo_transducer for Parakeet)
//	VOICEKIT_OFFLINE_ASR_LANGUAGE   - decode language (defaults to en)
//	VOICEKIT_VAD_MODEL              - Silero VAD model (optional; without it the
//	                                  audio is cut at quiet points)
//	VOICEKIT_LONG_AUDIO_WAV         - 16 kHz mono 16-bit WAV, ideally >= 60 s
//	VOICEKIT_LONG_AUDIO_REF         - reference transcript of that WAV
//	VOICEKIT_LONG_AUDIO_MAX_WER     - fail above this word error rate (default 0.35)
//
// longAudioConfigFromEnv builds the offline config the long-audio test and
// benchmark share; ok is false when the model or audio is not configured.
func longAudioConfigFromEnv() (config Config, wavPath, refPath string, ok bool) {
	config = DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOffline
	config.DefaultModel = "integration_long_audio"
	config.NumThreads = 4
	if v, err := strconv.Atoi(os.Getenv("VOICEKIT_OFFLINE_ASR_THREADS")); err == nil && v > 0 {
		config.NumThreads = v
	}
	config.Offline.ModelFamily = os.Getenv("VOICEKIT_OFFLINE_ASR_FAMILY")
	config.Offline.ModelType = os.Getenv("VOICEKIT_OFFLINE_ASR_MODEL_TYPE")
	config.Offline.TokensPath = os.Getenv("VOICEKIT_OFFLINE_ASR_TOKENS")
	config.Offline.ModelPath = os.Getenv("VOICEKIT_OFFLINE_ASR_MODEL")
	config.Offline.EncoderPath = os.Getenv("VOICEKIT_OFFLINE_ASR_ENCODER")
	config.Offline.DecoderPath = os.Getenv("VOICEKIT_OFFLINE_ASR_DECODER")
	config.Offline.JoinerPath = os.Getenv("VOICEKIT_OFFLINE_ASR_JOINER")
	config.Offline.Language = os.Getenv("VOICEKIT_OFFLINE_ASR_LANGUAGE")
	if config.Offline.Language == "" {
		config.Offline.Language = "en"
	}
	if vad := os.Getenv("VOICEKIT_VAD_MODEL"); vad != "" {
		config.VADProvider = VADProviderSilero
		config.VADModelPath = vad
	}
	wavPath, refPath = os.Getenv("VOICEKIT_LONG_AUDIO_WAV"), os.Getenv("VOICEKIT_LONG_AUDIO_REF")
	ok = offlineIntegrationConfigured(config.Offline) && wavPath != "" && refPath != ""
	return config, wavPath, refPath, ok
}

// BenchmarkOfflineTranscriberLongAudio decodes the long recording end to end
// (segmentation, every chunk, stitching) and reports real-time factor.
func BenchmarkOfflineTranscriberLongAudio(b *testing.B) {
	config, wavPath, _, ok := longAudioConfigFromEnv()
	if !ok {
		b.Skip("set VOICEKIT_OFFLINE_ASR_*, VOICEKIT_LONG_AUDIO_WAV and VOICEKIT_LONG_AUDIO_REF")
	}
	data, err := os.ReadFile(wavPath)
	if err != nil {
		b.Fatal(err)
	}
	samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
	if err != nil {
		b.Fatal(err)
	}
	transcriber, err := NewOfflineTranscriber(&config)
	if err != nil {
		b.Fatal(err)
	}
	defer transcriber.Close()
	for b.Loop() {
		if _, err := transcriber.Transcribe(context.Background(), samples, 16000); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(float64(len(samples))/16000*float64(b.N)/b.Elapsed().Seconds(), "x-realtime")
}

func TestLongAudioIntegration(t *testing.T) {
	config, wavPath, refPath, ok := longAudioConfigFromEnv()
	if !ok {
		t.Skip("set VOICEKIT_OFFLINE_ASR_*, VOICEKIT_LONG_AUDIO_WAV and VOICEKIT_LONG_AUDIO_REF to run the long-audio integration")
	}
	maxWER := 0.35
	if v := os.Getenv("VOICEKIT_LONG_AUDIO_MAX_WER"); v != "" {
		var err error
		if maxWER, err = strconv.ParseFloat(v, 64); err != nil {
			t.Fatalf("VOICEKIT_LONG_AUDIO_MAX_WER: %v", err)
		}
	}

	samples := readMonoWAV(t, wavPath)
	ref, err := os.ReadFile(refPath)
	if err != nil {
		t.Fatal(err)
	}
	duration := samplesToDuration(len(samples), 16000)

	transcriber, err := NewOfflineTranscriber(&config)
	if err != nil {
		t.Fatalf("NewOfflineTranscriber: %v", err)
	}
	defer transcriber.Close()

	start := time.Now()
	result, err := transcriber.Transcribe(context.Background(), samples, 16000)
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	elapsed := time.Since(start)

	for _, seg := range result.Segments {
		t.Logf("segment %6.2fs-%6.2fs: %s", seg.StartTime.Seconds(), seg.EndTime.Seconds(), seg.Text)
		if seg.EndTime-seg.StartTime > MaxDecodeDuration+time.Second {
			t.Errorf("segment %s-%s is longer than the decode cap", seg.StartTime, seg.EndTime)
		}
	}
	wer := wordErrorRate(normalizeForWER(string(ref)), normalizeForWER(result.Text))
	t.Logf("audio %s, %d segments, decoded in %s (%.1fx real time), WER %.2f%%", duration, len(result.Segments), elapsed, duration.Seconds()/elapsed.Seconds(), 100*wer)
	t.Logf("hypothesis: %s", result.Text)
	if wer > maxWER {
		t.Fatalf("WER %.2f%% above %.2f%%: long audio was not transcribed in full", 100*wer, 100*maxWER)
	}

	// A single decode of the same input must refuse or cover it, never
	// silently keep a prefix.
	if model, ok := transcriber.inner.(*SherpaOfflineModel); ok && model.maxDecode > 0 {
		_, err := model.Transcribe(context.Background(), samples, 16000)
		var tooLong *AudioTooLongError
		if !errors.As(err, &tooLong) {
			t.Fatalf("single decode of %s returned %v, want *AudioTooLongError", duration, err)
		}
	}
}

func readMonoWAV(t *testing.T, path string) []float32 {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
	if err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	return samples
}

func normalizeForWER(s string) []string {
	s = strings.ToLower(s)
	s = strings.Map(func(r rune) rune {
		if strings.ContainsRune(".,;:!?\"'()[]«»„“”‚‘’-–—", r) {
			return ' '
		}
		return r
	}, s)
	return strings.Fields(s)
}

// wordErrorRate is the word-level Levenshtein distance over the reference length.
func wordErrorRate(ref, hyp []string) float64 {
	if len(ref) == 0 {
		if len(hyp) == 0 {
			return 0
		}
		return 1
	}
	prev := make([]int, len(hyp)+1)
	cur := make([]int, len(hyp)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ref); i++ {
		cur[0] = i
		for j := 1; j <= len(hyp); j++ {
			cost := 1
			if ref[i-1] == hyp[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev, cur = cur, prev
	}
	return float64(prev[len(hyp)]) / float64(len(ref))
}
