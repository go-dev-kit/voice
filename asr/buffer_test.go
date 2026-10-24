package asr

import (
	"slices"
	"testing"

	"go.glpx.pro/gdk/voice/types"
)

func TestAudioBuffer_Append(t *testing.T) {
	config := &types.StreamingConfig{
		ChunkSize:   16000,
		OverlapSize: 1600,
		BufferSize:  48000,
		SampleRate:  16000,
	}

	buffer := NewAudioBuffer(config)

	// Test appending audio
	audio := make([]float32, 16000)
	for i := range audio {
		audio[i] = float32(i%100) / 100.0
	}

	err := buffer.Append(audio)
	if err != nil {
		t.Fatalf("Failed to append audio: %v", err)
	}

	if buffer.Size() != 16000 {
		t.Errorf("Expected buffer size 16000, got %d", buffer.Size())
	}

	// Test GetRecentChunk
	chunk := buffer.GetRecentChunk()
	if len(chunk) != 16000 {
		t.Errorf("Expected chunk size 16000, got %d", len(chunk))
	}

	// Test IsReady
	if !buffer.IsReady() {
		t.Error("Buffer should be ready")
	}
}

func TestAudioBuffer_Overflow(t *testing.T) {
	config := &types.StreamingConfig{
		ChunkSize:   16000,
		OverlapSize: 1600,
		BufferSize:  32000, // Small buffer to test overflow
		SampleRate:  16000,
	}

	buffer := NewAudioBuffer(config)

	// Fill buffer
	audio1 := make([]float32, 16000)
	err := buffer.Append(audio1)
	if err != nil {
		t.Fatalf("Failed to append first chunk: %v", err)
	}

	// Try to append more, should cause overflow/shifting
	audio2 := make([]float32, 16001) // One more than remaining capacity
	err = buffer.Append(audio2)
	if err != nil {
		t.Fatalf("Failed to append second chunk: %v", err)
	}

	// Buffer should have shifted and contain the most recent data
	if buffer.Size() != 32000 {
		t.Errorf("Expected buffer size 32000 after overflow, got %d", buffer.Size())
	}
}

func TestAudioBufferAppendAfterOversizedChunkWrapsWriteIndex(t *testing.T) {
	config := &types.StreamingConfig{
		ChunkSize:   4,
		OverlapSize: 1,
		BufferSize:  8,
		SampleRate:  16000,
	}
	buffer := NewAudioBuffer(config)

	if err := buffer.Append([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}); err != nil {
		t.Fatalf("append oversized chunk: %v", err)
	}
	if err := buffer.Append([]float32{11, 12}); err != nil {
		t.Fatalf("append after oversized chunk: %v", err)
	}

	got := buffer.GetRecentChunk()
	defer buffer.ReturnBuffer(got)
	want := []float32{9, 10, 11, 12}
	if !slices.Equal(got, want) {
		t.Fatalf("recent chunk = %v, want %v", got, want)
	}
}

// BenchmarkAudioBufferAppend measures the ring-buffer write path that runs
// once per streamed audio chunk (see AudioBuffer.Append). The chunk size
// matches the streaming service default; capacity is a power of two so the
// wrap falls mid-chunk, exercising both contiguous runs of the copy.
func BenchmarkAudioBufferAppend(b *testing.B) {
	config := &types.StreamingConfig{
		ChunkSize:   1024,
		OverlapSize: 64,
		BufferSize:  8192,
		SampleRate:  16000,
	}
	buffer := NewAudioBuffer(config)
	chunk := make([]float32, 1024)
	for i := range chunk {
		chunk[i] = float32(i) / 1024
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if err := buffer.Append(chunk); err != nil {
			b.Fatal(err)
		}
	}
}
