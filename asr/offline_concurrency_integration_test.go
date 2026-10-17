package asr

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// concurrencyClip returns the first 20 s of the long-audio fixture and a
// config for the configured offline model, or ok=false when unset.
func concurrencyClip(tb testing.TB) (Config, []float32, bool) {
	config, wavPath, _, ok := longAudioConfigFromEnv()
	if !ok {
		return config, nil, false
	}
	samples := readMonoWAV(tb, wavPath)
	return config, samples[:min(len(samples), 20*16000)], true
}

// TestSherpaOfflineModelConcurrentDecodeIntegration checks that decodes
// sharing one native recognizer give the same text as a lone decode.
func TestSherpaOfflineModelConcurrentDecodeIntegration(t *testing.T) {
	config, clip, ok := concurrencyClip(t)
	if !ok {
		t.Skip("set VOICEKIT_OFFLINE_ASR_*, VOICEKIT_LONG_AUDIO_WAV and VOICEKIT_LONG_AUDIO_REF")
	}
	config.NumThreads = 1
	config.Offline.MaxConcurrentDecodes = 4
	model, err := NewSherpaOfflineModel(&config)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	want, err := model.Transcribe(context.Background(), clip, 16000)
	if err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	got := make([]string, 16)
	errs := make([]error, len(got))
	for i := range got {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r, err := model.Transcribe(context.Background(), clip, 16000)
			if err == nil {
				got[i] = r.Text
			}
			errs[i] = err
		}()
	}
	wg.Wait()
	for i := range got {
		if errs[i] != nil || got[i] != want.Text {
			t.Fatalf("concurrent decode %d: %v %q, want %q", i, errs[i], got[i], want.Text)
		}
	}
}

// BenchmarkSherpaOfflineModelConcurrency measures decode throughput for
// concurrency x threads splits of the same core budget.
func BenchmarkSherpaOfflineModelConcurrency(b *testing.B) {
	config, clip, ok := concurrencyClip(b)
	if !ok {
		b.Skip("set VOICEKIT_OFFLINE_ASR_*, VOICEKIT_LONG_AUDIO_WAV and VOICEKIT_LONG_AUDIO_REF")
	}
	cores := config.NumThreads
	for _, decodes := range []int{1, 2, 4} {
		if decodes > cores {
			continue
		}
		b.Run(fmt.Sprintf("decodes=%d_threads=%d", decodes, cores/decodes), func(b *testing.B) {
			cfg := config
			cfg.NumThreads = cores / decodes
			cfg.Offline.MaxConcurrentDecodes = decodes
			model, err := NewSherpaOfflineModel(&cfg)
			if err != nil {
				b.Fatal(err)
			}
			defer model.Close()
			b.SetParallelism(decodes)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := model.Transcribe(context.Background(), clip, 16000); err != nil {
						b.Error(err)
						return
					}
				}
			})
			b.ReportMetric(float64(len(clip))/16000*float64(b.N)/b.Elapsed().Seconds(), "audio-s/s")
		})
	}
}
