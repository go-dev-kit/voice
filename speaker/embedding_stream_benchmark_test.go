package speaker

import (
	"os"
	"testing"

	"github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"

	"go.glpx.pro/gdk/voice/audio"
)

// These benchmarks split the cost of one speaker embedding into extractor
// init, native stream create/delete, and the full extraction, to decide
// whether pooling streams is worth it (streams are single use today). They
// need a real model and skip otherwise:
//
//	VOICEKIT_SPEAKER_MODEL - speaker embedding ONNX model
//	VOICEKIT_BENCH_CLIP_WAV - 16 kHz mono WAV of one utterance
//
//	go test -run='^$' -bench=SpeakerStream -benchmem -count=10 ./speaker
func speakerBenchExtractor(b *testing.B) (*SherpaExtractorAdapter, []float32) {
	b.Helper()
	model, wav := os.Getenv("VOICEKIT_SPEAKER_MODEL"), os.Getenv("VOICEKIT_BENCH_CLIP_WAV")
	if model == "" || wav == "" {
		b.Skip("set VOICEKIT_SPEAKER_MODEL and VOICEKIT_BENCH_CLIP_WAV to benchmark speaker streams")
	}
	data, err := os.ReadFile(wav)
	if err != nil {
		b.Fatal(err)
	}
	samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
	if err != nil {
		b.Fatal(err)
	}
	extractor := NewSherpaExtractorAdapter(&sherpa_onnx.SpeakerEmbeddingExtractorConfig{Model: model, NumThreads: 2, Provider: "cpu"})
	if extractor == nil {
		b.Fatal("create speaker embedding extractor")
	}
	b.Cleanup(extractor.Delete)
	return extractor, samples
}

// BenchmarkSpeakerStreamExtractorInit measures loading the extractor model.
func BenchmarkSpeakerStreamExtractorInit(b *testing.B) {
	model := os.Getenv("VOICEKIT_SPEAKER_MODEL")
	if model == "" {
		b.Skip("set VOICEKIT_SPEAKER_MODEL")
	}
	for b.Loop() {
		extractor := NewSherpaExtractorAdapter(&sherpa_onnx.SpeakerEmbeddingExtractorConfig{Model: model, NumThreads: 2, Provider: "cpu"})
		if extractor == nil {
			b.Fatal("create speaker embedding extractor")
		}
		extractor.Delete()
	}
}

// BenchmarkSpeakerStreamCreate measures creating and deleting an empty native
// stream, the cost a stream pool would save per extraction.
func BenchmarkSpeakerStreamCreate(b *testing.B) {
	extractor, _ := speakerBenchExtractor(b)
	b.ReportAllocs()
	for b.Loop() {
		stream := extractor.CreateStream()
		if stream == nil {
			b.Fatal("create stream")
		}
		stream.Delete()
	}
}

// BenchmarkSpeakerStreamExtract measures a full extraction: fresh stream,
// waveform, finish, compute, delete.
func BenchmarkSpeakerStreamExtract(b *testing.B) {
	extractor, samples := speakerBenchExtractor(b)
	b.ReportAllocs()
	for b.Loop() {
		stream := extractor.CreateStream()
		stream.AcceptWaveform(16000, samples)
		stream.InputFinished()
		if !extractor.IsReady(stream) {
			b.Fatal("stream not ready")
		}
		if emb := extractor.Compute(stream); len(emb) == 0 {
			b.Fatal("empty embedding")
		}
		stream.Delete()
	}
}
