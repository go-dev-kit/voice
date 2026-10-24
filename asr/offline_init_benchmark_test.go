package asr

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"go.glpx.pro/gdk/voice/audio"
)

// These benchmarks quantify, with real offline models, the one-time cost of
// building a recognizer, the first-decode penalty on a fresh recognizer, and
// steady-state decode speed, to decide whether a recognizer pool or warm-up
// is justified. They skip unless the model layout below is available.
//
//	VOICEKIT_BENCH_MODELS_DIR - directory holding
//	    sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/{encoder,decoder,joiner}.int8.onnx + tokens.txt
//	    sherpa-onnx-whisper-small/small-{encoder,decoder}.int8.onnx + small-tokens.txt
//	    silero_vad.onnx (optional; without it long audio is cut at quiet points)
//	VOICEKIT_BENCH_WAV        - 16 kHz mono 16-bit WAV, ideally 60 s (long-audio RTF)
//	VOICEKIT_BENCH_CLIP_WAV   - 16 kHz mono WAV of one utterance (first decode, language)
//
// Init cost spans seconds, so run with -benchtime=1x:
//
//	VOICEKIT_BENCH_MODELS_DIR=... VOICEKIT_BENCH_WAV=... VOICEKIT_BENCH_CLIP_WAV=... \
//	  go test -run='^$' -bench='OfflineRecognizer|OfflineTranscriberRTF|WhisperLanguage' -benchtime=1x -count=5 ./asr
type offlineBenchModel struct {
	name   string
	config func(dir string) Config
}

func offlineBenchBase(dir string) Config {
	config := DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOffline
	config.DefaultModel = "bench_offline"
	config.NumThreads = 4
	if vad := filepath.Join(dir, "silero_vad.onnx"); fileExists(vad) {
		config.VADProvider = VADProviderSilero
		config.VADModelPath = vad
	}
	return config
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

var offlineBenchModels = []offlineBenchModel{
	{"parakeet", func(dir string) Config {
		config := offlineBenchBase(dir)
		m := filepath.Join(dir, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
		config.Offline.ModelFamily = OfflineFamilyTransducer
		config.Offline.ModelType = "nemo_transducer"
		config.Offline.TokensPath = filepath.Join(m, "tokens.txt")
		config.Offline.EncoderPath = filepath.Join(m, "encoder.int8.onnx")
		config.Offline.DecoderPath = filepath.Join(m, "decoder.int8.onnx")
		config.Offline.JoinerPath = filepath.Join(m, "joiner.int8.onnx")
		config.Offline.Language = "de"
		return config
	}},
	{"whisper", func(dir string) Config {
		config := offlineBenchBase(dir)
		m := filepath.Join(dir, "sherpa-onnx-whisper-small")
		config.Offline.ModelFamily = OfflineFamilyWhisper
		config.Offline.TokensPath = filepath.Join(m, "small-tokens.txt")
		config.Offline.EncoderPath = filepath.Join(m, "small-encoder.int8.onnx")
		config.Offline.DecoderPath = filepath.Join(m, "small-decoder.int8.onnx")
		config.Offline.Language = "de"
		return config
	}},
}

// offlineBenchEach runs fn once per locally available model.
func offlineBenchEach(b *testing.B, fn func(b *testing.B, config Config)) {
	dir := os.Getenv("VOICEKIT_BENCH_MODELS_DIR")
	if dir == "" {
		b.Skip("set VOICEKIT_BENCH_MODELS_DIR to benchmark offline recognizers")
	}
	for _, m := range offlineBenchModels {
		config := m.config(dir)
		if !offlineIntegrationConfigured(config.Offline) || !fileExists(config.Offline.EncoderPath) {
			continue
		}
		b.Run(m.name, func(b *testing.B) { fn(b, config) })
	}
}

func benchWAV(b *testing.B, env string) []float32 {
	path := os.Getenv(env)
	if path == "" {
		b.Skipf("set %s", env)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
	if err != nil {
		b.Fatal(err)
	}
	return samples
}

// BenchmarkOfflineRecognizerInit measures building and closing one native
// offline recognizer: model load, graph optimisation, session creation.
func BenchmarkOfflineRecognizerInit(b *testing.B) {
	offlineBenchEach(b, func(b *testing.B, config Config) {
		b.ReportAllocs()
		for b.Loop() {
			model, err := NewSherpaOfflineModel(&config)
			if err != nil {
				b.Fatal(err)
			}
			if err := model.Close(); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkOfflineRecognizerFirstDecode times the first decode on a freshly
// built recognizer (init excluded). Compare with the Steady variant to see
// the penalty a warm-up decode would hide.
func BenchmarkOfflineRecognizerFirstDecode(b *testing.B) {
	clip := benchWAV(b, "VOICEKIT_BENCH_CLIP_WAV")
	offlineBenchEach(b, func(b *testing.B, config Config) {
		for b.Loop() {
			b.StopTimer()
			model, err := NewSherpaOfflineModel(&config)
			if err != nil {
				b.Fatal(err)
			}
			b.StartTimer()
			if _, err := model.Transcribe(context.Background(), clip, 16000); err != nil {
				b.Fatal(err)
			}
			b.StopTimer()
			_ = model.Close()
			b.StartTimer()
		}
	})
}

// BenchmarkOfflineRecognizerSteadyDecode times a decode on a recognizer that
// already decoded the same clip once.
func BenchmarkOfflineRecognizerSteadyDecode(b *testing.B) {
	clip := benchWAV(b, "VOICEKIT_BENCH_CLIP_WAV")
	offlineBenchEach(b, func(b *testing.B, config Config) {
		model, err := NewSherpaOfflineModel(&config)
		if err != nil {
			b.Fatal(err)
		}
		defer model.Close()
		if _, err := model.Transcribe(context.Background(), clip, 16000); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for b.Loop() {
			if _, err := model.Transcribe(context.Background(), clip, 16000); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(len(clip))/16000*float64(b.N)/b.Elapsed().Seconds(), "x-realtime")
	})
}

// BenchmarkOfflineTranscriberRTF decodes the long recording through
// NewOfflineTranscriber (Silero VAD when available) on a warm transcriber and
// reports the real-time factor.
func BenchmarkOfflineTranscriberRTF(b *testing.B) {
	samples := benchWAV(b, "VOICEKIT_BENCH_WAV")
	offlineBenchEach(b, func(b *testing.B, config Config) {
		transcriber, err := NewOfflineTranscriber(&config)
		if err != nil {
			b.Fatal(err)
		}
		defer transcriber.Close()
		if _, err := transcriber.Transcribe(context.Background(), samples[:16000*10], 16000); err != nil {
			b.Fatal(err)
		}
		b.ResetTimer()
		for b.Loop() {
			if _, err := transcriber.Transcribe(context.Background(), samples, 16000); err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(len(samples))/16000*float64(b.N)/b.Elapsed().Seconds(), "x-realtime")
	})
}

// BenchmarkWhisperLanguage compares a decode with Whisper language
// auto-detection (empty language) against a forced "de" on one utterance.
func BenchmarkWhisperLanguage(b *testing.B) {
	clip := benchWAV(b, "VOICEKIT_BENCH_CLIP_WAV")
	dir := os.Getenv("VOICEKIT_BENCH_MODELS_DIR")
	if dir == "" {
		b.Skip("set VOICEKIT_BENCH_MODELS_DIR")
	}
	config := offlineBenchModels[1].config(dir)
	if !fileExists(config.Offline.EncoderPath) {
		b.Skip("whisper small model not found")
	}
	for _, lang := range []struct{ name, code string }{{"auto", ""}, {"forced_de", "de"}} {
		b.Run(lang.name, func(b *testing.B) {
			config := config
			config.Offline.Language = lang.code
			model, err := NewSherpaOfflineModel(&config)
			if err != nil {
				b.Fatal(err)
			}
			defer model.Close()
			if _, err := model.Transcribe(context.Background(), clip, 16000); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for b.Loop() {
				if _, err := model.Transcribe(context.Background(), clip, 16000); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
