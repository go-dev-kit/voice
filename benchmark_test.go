package voicekit

import (
	"math"
	"testing"
	"time"

	"go.glpx.pro/gdk/voice/audio"
)

var benchmarkBytesSink []byte
var benchmarkFloatSink float32
var benchmarkSliceSink []float32

// TestBenchmarkRunner tests the benchmark runner functionality
func TestBenchmarkRunner(t *testing.T) {
	runner := NewBenchmarkRunner(t.TempDir())

	// Test basic functionality
	if runner == nil {
		t.Fatal("BenchmarkRunner creation failed")
	}

	if len(runner.results) != 0 {
		t.Errorf("Expected empty results map, got %d entries", len(runner.results))
	}
}

// BenchmarkVoiceKitAudioConversionPipeline benchmarks real local audio conversion work.
func BenchmarkVoiceKitAudioConversionPipeline(b *testing.B) {
	sampleRate := 44100
	duration := 0.25
	numSamples := int(float64(sampleRate) * duration)
	audioData := make([]float32, numSamples*2)
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioData[i*2] = float32(math.Sin(2 * math.Pi * 440 * t))
		audioData[i*2+1] = float32(math.Sin(2 * math.Pi * 880 * t))
	}

	converter, err := audio.NewConverter(audio.DefaultConverterConfig())
	if err != nil {
		b.Fatal(err)
	}
	inputConfig := &audio.AudioConfig{
		Format:        audio.FormatWAV,
		SampleRate:    sampleRate,
		Channels:      2,
		BitsPerSample: 16,
	}
	outputConfig := &audio.AudioConfig{
		Format:        audio.FormatPCM,
		SampleRate:    16000,
		Channels:      1,
		BitsPerSample: 16,
	}
	inputBytes, err := converter.ConvertFromFloat32(audioData, inputConfig)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		outputBytes, err := converter.ConvertAudio(inputBytes, inputConfig, outputConfig)
		if err != nil {
			b.Fatal(err)
		}
		benchmarkBytesSink = outputBytes
	}
}

// BenchmarkMemoryEfficiency benchmarks memory usage patterns
func BenchmarkMemoryEfficiency(b *testing.B) {
	// Test that our buffer pooling reduces allocations

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Test buffer pool usage
		buf1 := audio.DefaultAudioBufferPool.Get(4096)
		buf2 := audio.DefaultAudioBufferPool.Get(8192)
		buf3 := audio.Float32Pool.Get(2048)

		// Simulate some work
		for j := 0; j < len(buf1) && j < 100; j++ {
			buf1[j] = float32(j)
		}
		for j := 0; j < len(buf2) && j < 100; j++ {
			buf2[j] = float32(j * 2)
		}
		for j := 0; j < len(buf3) && j < 100; j++ {
			buf3[j] = float32(j * 3)
		}

		// Return to pools
		audio.DefaultAudioBufferPool.Put(buf1)
		audio.DefaultAudioBufferPool.Put(buf2)
		audio.Float32Pool.Put(buf3)
	}
}

// BenchmarkConcurrentOperations benchmarks concurrent VoiceKit operations
func BenchmarkConcurrentOperations(b *testing.B) {
	// Test concurrent performance with our optimizations

	samplesPerGoroutine := 4000

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		localData := make([]float32, samplesPerGoroutine)
		for pb.Next() {
			// Simulate concurrent audio processing
			for i := 0; i < len(localData); i++ {
				localData[i] = float32(i % 256) // Fill with test data
			}

			// Simulate processing with buffer pools
			buf := audio.DefaultAudioBufferPool.Get(len(localData))
			copy(buf, localData)

			// Simulate some computation
			for i := 1; i < len(buf); i++ {
				buf[i] = (buf[i] + buf[i-1]) * 0.5
			}

			audio.DefaultAudioBufferPool.Put(buf)
		}
	})
}

// BenchmarkDatabaseThroughput benchmarks database operation throughput
// BenchmarkOptimizationImpact benchmarks the impact of our optimizations
func BenchmarkOptimizationImpact(b *testing.B) {
	// Compare optimized vs non-optimized patterns

	b.Run("Optimized_BufferPool", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			buf := audio.DefaultAudioBufferPool.Get(4096)
			buf[0] = float32(i)
			benchmarkFloatSink += buf[0]
			audio.DefaultAudioBufferPool.Put(buf)
		}
	})

	b.Run("NonOptimized_NewAlloc", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			buf := make([]float32, 4096)
			buf[0] = float32(i)
			benchmarkFloatSink += buf[0]
			benchmarkSliceSink = buf
		}
	})
}

// BenchmarkMetricsCollection benchmarks our metrics collection overhead
func BenchmarkMetricsCollection(b *testing.B) {
	metrics := NewMetricsCollector()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		// Simulate metrics collection
		duration := time.Duration(i%1000) * time.Microsecond
		success := (i % 10) != 0 // 90% success rate

		metrics.RecordAudioProcessing(duration, success)
	}
}

// BenchmarkConfigValidation benchmarks configuration validation performance
func BenchmarkConfigValidation(b *testing.B) {
	config := DefaultConfig()
	// Set required fields for validation
	config.Speaker.DataDir = "/tmp/voicekit_bench"

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := config.Validate()
		if err != nil {
			b.Fatal(err)
		}
	}
}
