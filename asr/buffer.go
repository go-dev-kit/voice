package asr

import (
	"sync"

	"go.glpx.pro/gdk/voice/types"
)

// AudioBuffer implements the AudioBufferInterface with circular buffer optimization
type AudioBuffer struct {
	buffer     []float32
	readIdx    int // Read index for circular buffer
	writeIdx   int // Write index for circular buffer
	size       int // Current number of elements in buffer
	chunkSize  int
	overlap    int
	sampleRate int
	capacity   int
	mu         sync.RWMutex
	pool       *sync.Pool
}

// NewAudioBuffer creates a new audio buffer with circular buffer optimization
func NewAudioBuffer(config *types.StreamingConfig) *AudioBuffer {
	buffer := &AudioBuffer{
		chunkSize:  config.ChunkSize,
		overlap:    config.OverlapSize,
		sampleRate: config.SampleRate,
		capacity:   config.BufferSize,
		readIdx:    0,
		writeIdx:   0,
		size:       0,
		pool: &sync.Pool{
			New: func() any {
				buf := make([]float32, 0, config.ChunkSize*2)
				return &buf
			},
		},
	}

	buffer.buffer = make([]float32, buffer.capacity)
	return buffer
}

// Append adds audio data to the buffer using circular buffer optimization (O(1) instead of O(n))
func (b *AudioBuffer) Append(audio []float32) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	audioLen := len(audio)
	if audioLen == 0 {
		return nil
	}

	// If new data exceeds capacity, we need to make room
	if audioLen > b.capacity {
		// Data is larger than buffer capacity, replace entire buffer
		copy(b.buffer, audio[len(audio)-b.capacity:])
		b.readIdx = 0
		b.writeIdx = 0
		b.size = b.capacity
		return nil
	}

	// Make room for new data if necessary
	if b.size+audioLen > b.capacity {
		// Calculate how much to remove from the front
		removeCount := b.size + audioLen - b.capacity
		b.readIdx = (b.readIdx + removeCount) % b.capacity
		b.size -= removeCount
	}

	// Copy data into the circular buffer in at most two contiguous runs:
	// up to the physical end of the backing array, then the wrapped rest.
	// audioLen is never greater than capacity here, so two runs always
	// cover the chunk.
	first := min(audioLen, b.capacity-b.writeIdx)
	copy(b.buffer[b.writeIdx:], audio[:first])
	if rest := audioLen - first; rest > 0 {
		copy(b.buffer, audio[first:])
	}
	b.writeIdx = (b.writeIdx + audioLen) % b.capacity
	b.size += audioLen

	return nil
}

// GetRecentChunk returns the most recent chunk of audio for processing
func (b *AudioBuffer) GetRecentChunk() []float32 {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.size == 0 {
		return nil
	}

	chunkSize := b.chunkSize
	if chunkSize > b.size {
		chunkSize = b.size
	}

	// Calculate start position for the most recent chunk
	startIdx := (b.writeIdx - chunkSize + b.capacity) % b.capacity

	// Create result slice
	resultPtr := b.pool.Get().(*[]float32)
	result := (*resultPtr)[:0]
	if cap(result) < chunkSize {
		result = make([]float32, chunkSize)
	} else {
		result = result[:chunkSize]
	}

	// Copy data from circular buffer to linear result
	for i := 0; i < chunkSize; i++ {
		idx := (startIdx + i) % b.capacity
		result[i] = b.buffer[idx]
	}

	return result
}

// IsReady returns true if the buffer has enough data for processing
func (b *AudioBuffer) IsReady() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size >= b.chunkSize
}

// Reset clears the buffer
func (b *AudioBuffer) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.readIdx = 0
	b.writeIdx = 0
	b.size = 0
}

// Size returns the current buffer size
func (b *AudioBuffer) Size() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.size
}

// Capacity returns the buffer capacity
func (b *AudioBuffer) Capacity() int {
	return b.capacity
}

// ReturnBuffer returns a buffer slice to the pool for reuse
func (b *AudioBuffer) ReturnBuffer(buf []float32) {
	buf = buf[:0] // Reset length but keep capacity
	b.pool.Put(&buf)
}
