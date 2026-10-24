package audio

import (
	"sync"
)

// AudioBufferPool manages size-classed buffer pools for audio processing.
//
// Ownership rule for this package's exported pools (including
// [DefaultAudioBufferPool]): Get and Put are for buffers whose lifetime a
// single call controls. Values returned to CALLERS — for example by
// [Resampler.Resample] — are never returned to the pool by this package and
// must not be Put by callers either: the slice would be handed out again
// while its old contents are still referenced. Pool results that escape a
// call are garbage-collected like any other slice.
type AudioBufferPool struct {
	small  sync.Pool // 1KB - 8KB (1024 - 8192 float32 values)
	medium sync.Pool // 8KB - 64KB (8192 - 65536 float32 values)
	large  sync.Pool // 64KB - 512KB (65536 - 524288 float32 values)
}

// NewAudioBufferPool creates a new audio buffer pool with size-classed pools
func NewAudioBufferPool() *AudioBufferPool {
	return &AudioBufferPool{
		small: sync.Pool{
			New: func() any {
				// Default to 2048 float32 values (8KB)
				buf := make([]float32, 0, 2048)
				return &buf
			},
		},
		medium: sync.Pool{
			New: func() any {
				// Default to 16384 float32 values (64KB)
				buf := make([]float32, 0, 16384)
				return &buf
			},
		},
		large: sync.Pool{
			New: func() any {
				// Default to 131072 float32 values (512KB)
				buf := make([]float32, 0, 131072)
				return &buf
			},
		},
	}
}

// Get retrieves a buffer from the appropriate size class pool
func (p *AudioBufferPool) Get(size int) []float32 {
	var pool *sync.Pool
	switch {
	case size <= 2048: // <= 8KB
		pool = &p.small
	case size <= 16384: // <= 64KB
		pool = &p.medium
	default: // > 64KB
		pool = &p.large
	}

	buf := pool.Get()
	if buf == nil {
		return make([]float32, size)
	}

	slicePtr := buf.(*[]float32)
	slice := *slicePtr
	if cap(slice) < size {
		return make([]float32, size)
	}

	return slice[:size]
}

// Put returns a buffer to the appropriate size class pool
func (p *AudioBufferPool) Put(buf []float32) {
	// Reset length but keep capacity
	buf = buf[:0]

	var pool *sync.Pool
	switch {
	case cap(buf) <= 2048: // <= 8KB
		pool = &p.small
	case cap(buf) <= 16384: // <= 64KB
		pool = &p.medium
	default: // > 64KB
		pool = &p.large
	}

	pool.Put(&buf)
}

// GetStats returns statistics about pool usage (for monitoring)
func (p *AudioBufferPool) GetStats() map[string]any {
	// Note: sync.Pool doesn't provide built-in statistics
	// This is a placeholder for future monitoring integration
	return map[string]any{
		"pool_type":    "size_classed_audio_buffer",
		"size_classes": []string{"small (<=8KB)", "medium (<=64KB)", "large (>64KB)"},
	}
}

// Float32BufferPool provides a simpler interface for float32 slice pooling
type Float32BufferPool struct {
	pool sync.Pool
}

// NewFloat32BufferPool creates a buffer pool for float32 slices
func NewFloat32BufferPool(defaultCapacity int) *Float32BufferPool {
	return &Float32BufferPool{
		pool: sync.Pool{
			New: func() any {
				buf := make([]float32, 0, defaultCapacity)
				return &buf
			},
		},
	}
}

// Get retrieves a float32 slice from the pool
func (p *Float32BufferPool) Get(size int) []float32 {
	buf := p.pool.Get()
	if buf == nil {
		return make([]float32, size)
	}

	slicePtr := buf.(*[]float32)
	slice := *slicePtr
	if cap(slice) < size {
		return make([]float32, size)
	}

	return slice[:size]
}

// Put returns a float32 slice to the pool
func (p *Float32BufferPool) Put(buf []float32) {
	buf = buf[:0] // Reset length, keep capacity
	p.pool.Put(&buf)
}

// Int16BufferPool provides pooling for int16 slices (commonly used in audio)
type Int16BufferPool struct {
	pool sync.Pool
}

// NewInt16BufferPool creates a buffer pool for int16 slices
func NewInt16BufferPool(defaultCapacity int) *Int16BufferPool {
	return &Int16BufferPool{
		pool: sync.Pool{
			New: func() any {
				buf := make([]int16, 0, defaultCapacity)
				return &buf
			},
		},
	}
}

// Get retrieves an int16 slice from the pool
func (p *Int16BufferPool) Get(size int) []int16 {
	buf := p.pool.Get()
	if buf == nil {
		return make([]int16, size)
	}

	slicePtr := buf.(*[]int16)
	slice := *slicePtr
	if cap(slice) < size {
		return make([]int16, size)
	}

	return slice[:size]
}

// Put returns an int16 slice to the pool
func (p *Int16BufferPool) Put(buf []int16) {
	buf = buf[:0] // Reset length, keep capacity
	p.pool.Put(&buf)
}

// ByteBufferPool provides pooling for byte slices
type ByteBufferPool struct {
	pool sync.Pool
}

// NewByteBufferPool creates a buffer pool for byte slices
func NewByteBufferPool(defaultCapacity int) *ByteBufferPool {
	return &ByteBufferPool{
		pool: sync.Pool{
			New: func() any {
				buf := make([]byte, 0, defaultCapacity)
				return &buf
			},
		},
	}
}

// Get retrieves a byte slice from the pool
func (p *ByteBufferPool) Get(size int) []byte {
	buf := p.pool.Get()
	if buf == nil {
		return make([]byte, size)
	}

	slicePtr := buf.(*[]byte)
	slice := *slicePtr
	if cap(slice) < size {
		return make([]byte, size)
	}

	return slice[:size]
}

// Put returns a byte slice to the pool
func (p *ByteBufferPool) Put(buf []byte) {
	buf = buf[:0] // Reset length, keep capacity
	p.pool.Put(&buf)
}

// Global pools for common use cases
var (
	// DefaultAudioBufferPool provides size-classed pooling for audio samples
	DefaultAudioBufferPool = NewAudioBufferPool()

	// Float32Pool provides general-purpose float32 slice pooling
	Float32Pool = NewFloat32BufferPool(4096)

	// Int16Pool provides int16 slice pooling for PCM audio
	Int16Pool = NewInt16BufferPool(4096)

	// BytePool provides byte slice pooling for I/O operations
	BytePool = NewByteBufferPool(8192)
)
