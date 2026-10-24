package audio

import (
	"math"
	"testing"
)

// A zero kernel weight sum must not leave pooled garbage in the output.
func TestResampleWindowedSincZeroWeightsWritesEverySample(t *testing.T) {
	const n = 64
	dirty := DefaultAudioBufferPool.Get(n * 2)
	for i := range dirty {
		dirty[i] = 99
	}
	DefaultAudioBufferPool.Put(dirty)

	input := make([]float32, n)
	for i := range input {
		input[i] = 0.25
	}
	zero := func(float64) float64 { return 0 }
	out := resampleWindowedSinc(input, 2, n*2, 8, zero)
	for i, v := range out {
		if v != 0.25 {
			t.Fatalf("output[%d] = %v, want nearest input sample 0.25", i, v)
		}
	}
}

func TestWAVDataSizeFits(t *testing.T) {
	cases := []struct {
		size int
		want bool
	}{
		{0, true},
		{1 << 20, true},
		{math.MaxUint32 - 36, true},
		{math.MaxUint32 - 35, false},
		{math.MaxUint32, false},
		{-1, false},
	}
	for _, tc := range cases {
		if got := wavDataSizeFits(tc.size); got != tc.want {
			t.Errorf("wavDataSizeFits(%d) = %v, want %v", tc.size, got, tc.want)
		}
	}
}
