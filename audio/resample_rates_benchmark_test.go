package audio

import (
	"fmt"
	"testing"
)

// BenchmarkResampleSincRates measures the band-limited resampler on one minute
// of mono audio for the telephony upsample (8 kHz) and the common downsample
// sources, so the cost can be set against model decode time. Run:
//
//	go test -run='^$' -bench=ResampleSincRates -benchmem -count=10 ./audio
func BenchmarkResampleSincRates(b *testing.B) {
	for _, src := range []int{8000, 44100, 48000} {
		b.Run(fmt.Sprintf("%dHz_to_16k", src), func(b *testing.B) {
			in := longTone(src, 60)
			r := NewResampler(&ResampleConfig{Method: MethodSinc, Quality: QualityHigh, FilterLength: 32})

			b.SetBytes(int64(len(in)) * 4)
			b.ReportAllocs()
			for b.Loop() {
				out, err := r.Resample(in, src, 16000)
				if err != nil {
					b.Fatal(err)
				}
				if len(out) == 0 {
					b.Fatal("no samples")
				}
			}
			b.ReportMetric(60/(b.Elapsed().Seconds()/float64(b.N)), "x-realtime")
		})
	}
}
