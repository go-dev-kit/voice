package audio

import (
	"fmt"
	"testing"
	"time"
)

// longTone is seconds of a two-tone signal at rate, so the resampler works on
// non-trivial input rather than silence.
func longTone(rate int, seconds int) []float32 {
	n := rate * seconds
	a, b := sine(1000, rate, n, 0.4), sine(3100, rate, n, 0.2)
	for i := range a {
		a[i] += b[i]
	}
	return a
}

// BenchmarkDecodeMono measures the ASR ingest path for a one-minute upload:
// parse a 48 kHz stereo 16-bit WAV, mix to mono, resample to 16 kHz.
func BenchmarkDecodeMono(b *testing.B) {
	const rate, seconds = 48000, 60
	left, right := longTone(rate, seconds), sine(440, rate, rate*seconds, 0.3)
	stereo := make([]float32, 2*len(left))
	for i := range left {
		stereo[2*i], stereo[2*i+1] = left[i], right[i]
	}
	wavData := pcm16WAV(rate, 2, stereo)

	b.SetBytes(int64(len(wavData)))
	b.ReportAllocs()
	for b.Loop() {
		out, err := DecodeMono(wavData, FormatWAV, 16000, DecodeLimits{MaxDuration: 10 * time.Minute})
		if err != nil {
			b.Fatal(err)
		}
		if len(out) == 0 {
			b.Fatal("no samples")
		}
	}
}

// BenchmarkResampleSinc measures the band-limited resampler alone on one
// minute of mono audio, the common downsampling cases for speech recognition.
func BenchmarkResampleSinc(b *testing.B) {
	for _, src := range []int{48000, 44100} {
		b.Run(fmt.Sprintf("%gk_to_16k", float64(src)/1000), func(b *testing.B) {
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
		})
	}
}
