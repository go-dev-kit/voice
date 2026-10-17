package audio

import "testing"

func TestAudioBufferPoolSizeClasses(t *testing.T) {
	p := NewAudioBufferPool()
	// Sizes straddle each class boundary; Get must always return exactly size
	// elements, whether the buffer is fresh, recycled, or too small to reuse.
	for _, size := range []int{0, 1, 2048, 2049, 16384, 16385, 131072, 200000} {
		for round := range 3 {
			buf := p.Get(size)
			if len(buf) != size {
				t.Fatalf("size %d round %d: len = %d", size, round, len(buf))
			}
			for i := range buf {
				buf[i] = float32(i)
			}
			p.Put(buf)
		}
	}
	if p.GetStats()["pool_type"] == nil {
		t.Error("GetStats should describe the pool")
	}
}

func TestAudioBufferPoolGrowsPastRecycledCapacity(t *testing.T) {
	p := NewAudioBufferPool()
	p.Put(make([]float32, 10))
	buf := p.Get(2000)
	if len(buf) != 2000 {
		t.Fatalf("len = %d, want 2000", len(buf))
	}
}

func TestTypedPools(t *testing.T) {
	f := NewFloat32BufferPool(16)
	for _, size := range []int{4, 16, 100} {
		buf := f.Get(size)
		if len(buf) != size {
			t.Errorf("float32 pool: len = %d, want %d", len(buf), size)
		}
		f.Put(buf)
	}

	i := NewInt16BufferPool(16)
	for _, size := range []int{4, 16, 100} {
		buf := i.Get(size)
		if len(buf) != size {
			t.Errorf("int16 pool: len = %d, want %d", len(buf), size)
		}
		i.Put(buf)
	}

	b := NewByteBufferPool(16)
	for _, size := range []int{4, 16, 100} {
		buf := b.Get(size)
		if len(buf) != size {
			t.Errorf("byte pool: len = %d, want %d", len(buf), size)
		}
		b.Put(buf)
	}
}
