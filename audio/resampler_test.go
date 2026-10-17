package audio

import (
	"fmt"
	"math"
	"testing"
)

func newTestResampler(method ResampleMethod, filterLength int) *Resampler {
	return NewResampler(&ResampleConfig{Method: method, Quality: QualityHigh, FilterLength: filterLength})
}

// bandLimitedMethods are the two resamplers that low-pass before decimating.
var bandLimitedMethods = []ResampleMethod{MethodSinc, MethodLanczos}

func TestResampleOutputLength(t *testing.T) {
	rates := []struct{ src, dst int }{
		{48000, 16000}, {44100, 16000}, {16000, 48000}, {16000, 44100},
		{8000, 16000}, {96000, 16000}, {22050, 16000}, {24000, 16000},
	}
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
		for _, r := range rates {
			t.Run(fmt.Sprintf("%s/%d-%d", method, r.src, r.dst), func(t *testing.T) {
				in := sine(440, r.src, r.src/2+7, 0.5)
				out, err := newTestResampler(method, 16).Resample(in, r.src, r.dst)
				if err != nil {
					t.Fatal(err)
				}
				if want := expectedResampleLen(len(in), r.src, r.dst); len(out) != want {
					t.Errorf("got %d samples, want %d", len(out), want)
				}
				for i, v := range out {
					if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
						t.Fatalf("sample %d is %v", i, v)
					}
				}
			})
		}
	}
}

func TestResampleBandLimitedPassband(t *testing.T) {
	tests := []struct {
		name     string
		src, dst int
		freq     float64
	}{
		{"1 kHz 48k to 16k", 48000, 16000, 1000},
		{"1 kHz 44.1k to 16k", 44100, 16000, 1000},
		{"3 kHz 48k to 16k", 48000, 16000, 3000},
		{"1 kHz 16k to 48k", 16000, 48000, 1000},
		{"1 kHz 16k to 44.1k", 16000, 44100, 1000},
		{"1 kHz 8k to 16k", 8000, 16000, 1000},
		{"1 kHz 96k to 16k", 96000, 16000, 1000},
	}
	for _, method := range bandLimitedMethods {
		for _, tt := range tests {
			t.Run(string(method)+"/"+tt.name, func(t *testing.T) {
				const amp = 0.5
				in := sine(tt.freq, tt.src, tt.src, amp)
				out, err := newTestResampler(method, 32).Resample(in, tt.src, tt.dst)
				if err != nil {
					t.Fatal(err)
				}
				want := amp / math.Sqrt2
				if db := gainDB(rmsInterior(out, 0.1), want); math.Abs(db) > 1 {
					t.Errorf("passband gain %.3f dB, want within 1 dB", db)
				}
			})
		}
	}
}

func TestResampleBandLimitedStopband(t *testing.T) {
	// Content above the target Nyquist (8 kHz) must be removed, not folded back
	// into the passband as a linear or cubic interpolator would.
	tests := []struct {
		name     string
		src, dst int
		freq     float64
	}{
		{"12 kHz 48k to 16k", 48000, 16000, 12000},
		{"20 kHz 48k to 16k", 48000, 16000, 20000},
		{"10 kHz 44.1k to 16k", 44100, 16000, 10000},
		{"18 kHz 44.1k to 16k", 44100, 16000, 18000},
		{"6 kHz 16k to 8k", 16000, 8000, 6000},
	}
	for _, method := range bandLimitedMethods {
		for _, tt := range tests {
			t.Run(string(method)+"/"+tt.name, func(t *testing.T) {
				const amp = 0.5
				in := sine(tt.freq, tt.src, tt.src, amp)
				out, err := newTestResampler(method, 32).Resample(in, tt.src, tt.dst)
				if err != nil {
					t.Fatal(err)
				}
				ref := amp / math.Sqrt2
				if db := gainDB(rmsInterior(out, 0.1), ref); db > -40 {
					t.Errorf("stopband gain %.1f dB, want at most -40 dB", db)
				}
			})
		}
	}
}

func TestResampleDefaultFilterLengths(t *testing.T) {
	// FilterLength <= 0 selects the default; tiny lengths are floored. All of
	// them must still keep a passband tone and reject an aliasing one.
	for _, length := range []int{-1, 0, 1, 4, 16, 64} {
		for _, method := range bandLimitedMethods {
			t.Run(fmt.Sprintf("%s/len%d", method, length), func(t *testing.T) {
				r := newTestResampler(method, length)
				pass, err := r.Resample(sine(1000, 48000, 48000, 0.5), 48000, 16000)
				if err != nil {
					t.Fatal(err)
				}
				if db := gainDB(rmsInterior(pass, 0.1), 0.5/math.Sqrt2); math.Abs(db) > 1 {
					t.Errorf("passband gain %.3f dB", db)
				}
				// Longer kernels have a narrower transition band; 14 kHz is far
				// enough above the 8 kHz cutoff for even the shortest kernel to reject.
				stop, err := r.Resample(sine(14000, 48000, 48000, 0.5), 48000, 16000)
				if err != nil {
					t.Fatal(err)
				}
				limit := -20.0
				if length == 0 || length >= 16 || length < 0 {
					limit = -40
				}
				if db := gainDB(rmsInterior(stop, 0.1), 0.5/math.Sqrt2); db > limit {
					t.Errorf("stopband gain %.1f dB, want at most %.0f dB", db, limit)
				}
			})
		}
	}
}

func TestResampleConstantSignal(t *testing.T) {
	const level = 0.25
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
		for _, r := range []struct{ src, dst int }{{48000, 16000}, {44100, 16000}, {16000, 48000}, {16000, 22050}} {
			t.Run(fmt.Sprintf("%s/%d-%d", method, r.src, r.dst), func(t *testing.T) {
				in := make([]float32, r.src/4)
				for i := range in {
					in[i] = level
				}
				out, err := newTestResampler(method, 16).Resample(in, r.src, r.dst)
				if err != nil {
					t.Fatal(err)
				}
				// Every output, edges included: edge samples are clamped, so a
				// constant stays constant, and filter weights are normalised.
				for i, v := range out {
					if math.Abs(float64(v)-level) > 1e-5 {
						t.Fatalf("sample %d = %v, want %v", i, v, level)
					}
				}
			})
		}
	}
}

func TestResampleInterpolatorsReproduceLinearRamp(t *testing.T) {
	// Linear and Catmull-Rom interpolation are exact on a straight line, so an
	// output sample must equal the ramp at its own time: any half-sample shift
	// shows up as a constant offset of half a step.
	const step = 0.0001
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic} {
		for _, r := range []struct{ src, dst int }{{16000, 48000}, {16000, 32000}, {48000, 16000}, {44100, 16000}, {16000, 44100}} {
			t.Run(fmt.Sprintf("%s/%d-%d", method, r.src, r.dst), func(t *testing.T) {
				in := make([]float32, 4000)
				for i := range in {
					in[i] = float32(float64(i) * step)
				}
				out, err := newTestResampler(method, 16).Resample(in, r.src, r.dst)
				if err != nil {
					t.Fatal(err)
				}
				ratio := float64(r.dst) / float64(r.src)
				checked := 0
				// Stay clear of the first and last input samples, where the
				// neighborhood is clamped rather than extended.
				for i := range out {
					srcPos := float64(i) / ratio
					if srcPos < 2 || srcPos > float64(len(in)-4) {
						continue
					}
					want := srcPos * step
					if math.Abs(float64(out[i])-want) > 2e-6 {
						t.Fatalf("out[%d] = %.8f, want %.8f (input position %.3f)", i, out[i], want, srcPos)
					}
					checked++
				}
				if checked < len(out)/2 {
					t.Fatalf("only %d of %d samples checked", checked, len(out))
				}
			})
		}
	}
}

func TestResampleCubicUsesFourPointNeighbourhood(t *testing.T) {
	// A parabola is not reproduced by linear interpolation but is by a
	// four-point cubic away from the edges: guards against cubic degrading
	// to linear.
	in := make([]float32, 200)
	for i := range in {
		x := float64(i) / 100
		in[i] = float32(x * x)
	}
	cubic, _ := newTestResampler(MethodCubic, 16).Resample(in, 16000, 48000)
	linear, _ := newTestResampler(MethodLinear, 16).Resample(in, 16000, 48000)
	var cubicErr, linearErr float64
	for i := 12; i < len(cubic)-12; i++ {
		x := float64(i) / 3 / 100
		cubicErr = math.Max(cubicErr, math.Abs(float64(cubic[i])-x*x))
		linearErr = math.Max(linearErr, math.Abs(float64(linear[i])-x*x))
	}
	if cubicErr >= linearErr/5 {
		t.Errorf("cubic error %.2e not clearly below linear error %.2e", cubicErr, linearErr)
	}
}

func TestResampleSameRateIsACopy(t *testing.T) {
	in := []float32{0.1, -0.2, 0.3, 0.4}
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
		out, err := newTestResampler(method, 16).Resample(in, 16000, 16000)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceNear(t, out, in, 0)
		out[0] = 99
		if in[0] == 99 {
			t.Errorf("%s: same-rate output aliases the input", method)
		}
		in[0] = 0.1
	}
}

func TestResampleEdgeCases(t *testing.T) {
	r := newTestResampler(MethodSinc, 16)

	t.Run("empty input", func(t *testing.T) {
		out, err := r.Resample(nil, 48000, 16000)
		if err != nil || len(out) != 0 {
			t.Fatalf("got %v, %v", out, err)
		}
	})
	t.Run("single sample", func(t *testing.T) {
		for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
			out, err := newTestResampler(method, 16).Resample([]float32{0.5}, 8000, 16000)
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != 2 {
				t.Fatalf("%s: got %d samples, want 2", method, len(out))
			}
			for _, v := range out {
				assertNear(t, v, 0.5, 1e-5)
			}
		}
	})
	t.Run("shorter than the filter", func(t *testing.T) {
		out, err := newTestResampler(MethodSinc, 64).Resample([]float32{1, 1, 1}, 48000, 16000)
		if err != nil {
			t.Fatal(err)
		}
		for _, v := range out {
			assertNear(t, v, 1, 1e-5)
		}
	})
	t.Run("unknown method falls back to linear", func(t *testing.T) {
		in := sine(440, 16000, 1600, 0.5)
		got, err := newTestResampler("bogus", 16).Resample(in, 16000, 8000)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := newTestResampler(MethodLinear, 16).Resample(in, 16000, 8000)
		assertSliceNear(t, got, want, 0)
	})
	t.Run("nil config defaults to linear", func(t *testing.T) {
		in := sine(440, 16000, 1600, 0.5)
		got, err := NewResampler(nil).Resample(in, 16000, 8000)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := newTestResampler(MethodLinear, 16).Resample(in, 16000, 8000)
		assertSliceNear(t, got, want, 0)
	})
}

func TestResampleRejectsInvalidRates(t *testing.T) {
	r := newTestResampler(MethodSinc, 16)
	in := make([]float32, 100)
	tests := []struct {
		name     string
		src, dst int
	}{
		{"zero source", 0, 16000},
		{"negative source", -1, 16000},
		{"zero target", 16000, 0},
		{"negative target", 16000, -16000},
		{"source below range", 7999, 16000},
		{"source above range", 192001, 16000},
		{"target below range", 16000, 7999},
		{"target above range", 16000, 192001},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if out, err := r.Resample(in, tt.src, tt.dst); err == nil {
				t.Fatalf("expected an error, got %d samples", len(out))
			}
		})
	}
	// The range check comes before the empty-input shortcut.
	if _, err := r.Resample(nil, 100, 16000); err == nil {
		t.Error("empty input with an invalid rate should still error")
	}
}

func TestResampleDoesNotModifyInput(t *testing.T) {
	in := sine(1000, 48000, 4800, 0.5)
	snapshot := append([]float32(nil), in...)
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
		if _, err := newTestResampler(method, 16).Resample(in, 48000, 16000); err != nil {
			t.Fatal(err)
		}
		assertSliceNear(t, in, snapshot, 0)
	}
}

func TestResampleConcurrent(t *testing.T) {
	// Resample documents itself as thread-safe; run it under -race.
	r := newTestResampler(MethodSinc, 32)
	in := sine(1000, 48000, 9600, 0.5)
	done := make(chan error, 8)
	for range 8 {
		go func() {
			out, err := r.Resample(in, 48000, 16000)
			if err == nil && len(out) != expectedResampleLen(len(in), 48000, 16000) {
				err = fmt.Errorf("got %d samples", len(out))
			}
			done <- err
		}()
	}
	for range 8 {
		if err := <-done; err != nil {
			t.Error(err)
		}
	}
}

func TestConvertChannels(t *testing.T) {
	r := newTestResampler(MethodLinear, 16)
	tests := []struct {
		name     string
		in       []float32
		src, dst int
		want     []float32
	}{
		{"mono to stereo", []float32{0.1, -0.2, 0.3}, 1, 2, []float32{0.1, 0.1, -0.2, -0.2, 0.3, 0.3}},
		{"stereo to mono", []float32{1, 0, 0.5, -0.5, 0.2, 0.4}, 2, 1, []float32{0.5, 0, 0.3}},
		{"opposite channels cancel", []float32{0.5, -0.5, -0.25, 0.25}, 2, 1, []float32{0, 0}},
		{"same channels copies", []float32{0.1, 0.2, 0.3}, 2, 2, []float32{0.1, 0.2, 0.3}},
		{"empty", nil, 1, 2, []float32{}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := r.ConvertChannels(tt.in, tt.src, tt.dst)
			if err != nil {
				t.Fatal(err)
			}
			assertSliceNear(t, got, tt.want, 1e-7)
		})
	}
	for _, c := range []struct{ src, dst int }{{3, 1}, {1, 3}, {6, 2}, {2, 6}} {
		if _, err := r.ConvertChannels(make([]float32, 12), c.src, c.dst); err == nil {
			t.Errorf("%d -> %d channels: expected an error", c.src, c.dst)
		}
	}
}

func TestNormalizeAudio(t *testing.T) {
	r := newTestResampler(MethodLinear, 16)

	t.Run("reaches target RMS", func(t *testing.T) {
		in := sine(440, 16000, 16000, 0.2) // RMS 0.1414
		out := r.NormalizeAudio(in, 0.3)
		if got := rmsInterior(out, 0); math.Abs(got-0.3) > 1e-3 {
			t.Errorf("RMS = %.4f, want 0.3", got)
		}
	})
	t.Run("gain is capped at 4x", func(t *testing.T) {
		in := []float32{0.001, -0.001, 0.001, -0.001}
		out := r.NormalizeAudio(in, 0.5)
		assertSliceNear(t, out, []float32{0.004, -0.004, 0.004, -0.004}, 1e-7)
	})
	t.Run("silence is returned unchanged", func(t *testing.T) {
		in := make([]float32, 8)
		assertSliceNear(t, r.NormalizeAudio(in, 0.1), in, 0)
	})
	t.Run("empty", func(t *testing.T) {
		if out := r.NormalizeAudio(nil, 0.1); len(out) != 0 {
			t.Errorf("got %v", out)
		}
	})
}

func TestResampleHelpers(t *testing.T) {
	if got := GetResampleRatio(48000, 16000); math.Abs(got-1.0/3) > 1e-12 {
		t.Errorf("GetResampleRatio = %v", got)
	}
	if got := EstimateOutputLength(48000, GetResampleRatio(48000, 24000)); got != 24000 {
		t.Errorf("EstimateOutputLength = %d, want 24000", got)
	}
	for _, c := range []struct{ v, lo, hi, want int }{{-5, 0, 10, 0}, {5, 0, 10, 5}, {15, 0, 10, 10}, {0, 0, 10, 0}, {10, 0, 10, 10}} {
		if got := clamp(c.v, c.lo, c.hi); got != c.want {
			t.Errorf("clamp(%d,%d,%d) = %d, want %d", c.v, c.lo, c.hi, got, c.want)
		}
	}
	if sinc(0) != 1 || math.Abs(sinc(1)) > 1e-15 || math.Abs(sinc(2)) > 1e-15 {
		t.Error("sinc must be 1 at 0 and vanish at integers")
	}
	if lanczosWindow(0) != 1 || math.Abs(lanczosWindow(1)) > 1e-15 {
		t.Error("lanczos window must be 1 at the center and 0 at the edge")
	}
	if blackmanWindow(0) < 0.999 || math.Abs(blackmanWindow(1)) > 1e-12 {
		t.Errorf("blackman window must be 1 at the center and 0 at the edge: %v %v", blackmanWindow(0), blackmanWindow(1))
	}
}
