package audio

import (
	"math"
	"testing"
)

// BenchmarkResampleLinear benchmarks linear interpolation resampling
func BenchmarkResampleLinear(b *testing.B) {
	// Create test audio data (1 second of 44.1kHz audio)
	sampleRate := 44100
	duration := 1.0
	numSamples := int(float64(sampleRate) * duration)

	// Generate a sine wave test signal
	audioData := make([]float32, numSamples)
	frequency := 1000.0 // 1kHz test tone
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioData[i] = float32(math.Sin(2 * math.Pi * frequency * t))
	}

	config := &ResampleConfig{
		Method:       MethodLinear,
		Quality:      QualityMedium,
		FilterLength: 16,
		UseSIMD:      false, // Test without SIMD first
	}

	resampler := NewResampler(config)

	// Benchmark resampling from 44.1kHz to 16kHz (common ASR scenario)
	targetRate := 16000

	b.ReportAllocs()

	for b.Loop() {
		_, err := resampler.Resample(audioData, sampleRate, targetRate)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResampleCubic benchmarks cubic interpolation resampling
func BenchmarkResampleCubic(b *testing.B) {
	sampleRate := 44100
	duration := 1.0
	numSamples := int(float64(sampleRate) * duration)

	audioData := make([]float32, numSamples)
	frequency := 1000.0
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioData[i] = float32(math.Sin(2 * math.Pi * frequency * t))
	}

	config := &ResampleConfig{
		Method:       MethodCubic,
		Quality:      QualityHigh,
		FilterLength: 16,
		UseSIMD:      false,
	}

	resampler := NewResampler(config)
	targetRate := 16000

	b.ReportAllocs()

	for b.Loop() {
		_, err := resampler.Resample(audioData, sampleRate, targetRate)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkResampleLanczos benchmarks Lanczos resampling
func BenchmarkResampleLanczos(b *testing.B) {
	sampleRate := 44100
	duration := 1.0
	numSamples := int(float64(sampleRate) * duration)

	audioData := make([]float32, numSamples)
	frequency := 1000.0
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		audioData[i] = float32(math.Sin(2 * math.Pi * frequency * t))
	}

	config := &ResampleConfig{
		Method:       MethodLanczos,
		Quality:      QualityHigh,
		FilterLength: 8,
		UseSIMD:      false,
	}

	resampler := NewResampler(config)
	targetRate := 16000

	b.ReportAllocs()

	for b.Loop() {
		_, err := resampler.Resample(audioData, sampleRate, targetRate)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkConvertChannels benchmarks channel conversion
func BenchmarkConvertChannels(b *testing.B) {
	// Create stereo test data
	numSamples := 44100 // 1 second at 44.1kHz
	audioData := make([]float32, numSamples*2)

	// Generate stereo sine waves
	for i := 0; i < numSamples; i++ {
		t := float64(i) / 44100.0
		left := math.Sin(2 * math.Pi * 1000 * t)  // 1kHz left
		right := math.Sin(2 * math.Pi * 1200 * t) // 1.2kHz right
		audioData[i*2] = float32(left)
		audioData[i*2+1] = float32(right)
	}

	config := &ResampleConfig{
		Method:       MethodCubic,
		Quality:      QualityMedium,
		FilterLength: 16,
		UseSIMD:      false,
	}

	resampler := NewResampler(config)

	b.ReportAllocs()

	for b.Loop() {
		_, err := resampler.ConvertChannels(audioData, 2, 1) // Stereo to mono
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkNormalizeAudio benchmarks audio normalization
func BenchmarkNormalizeAudio(b *testing.B) {
	numSamples := 44100
	audioData := make([]float32, numSamples)

	// Generate varying amplitude signal
	for i := 0; i < numSamples; i++ {
		t := float64(i) / 44100.0
		// Varying amplitude to test normalization
		amplitude := 0.1 + 0.4*math.Sin(2*math.Pi*0.1*t)
		audioData[i] = float32(amplitude * math.Sin(2*math.Pi*1000*t))
	}

	config := &ResampleConfig{
		Method:       MethodCubic,
		Quality:      QualityMedium,
		FilterLength: 16,
		UseSIMD:      false,
	}

	resampler := NewResampler(config)
	targetRMS := float32(0.1)

	b.ReportAllocs()

	for b.Loop() {
		_ = resampler.NormalizeAudio(audioData, targetRMS)
	}
}

// BenchmarkAudioProcessingPipeline benchmarks the full audio processing pipeline
func BenchmarkAudioProcessingPipeline(b *testing.B) {
	// Create test audio (44.1kHz stereo)
	sampleRate := 44100
	duration := 0.1 // 100ms for faster benchmarking
	numSamples := int(float64(sampleRate) * duration)

	inputAudio := make([]float32, numSamples*2)
	for i := 0; i < numSamples; i++ {
		t := float64(i) / float64(sampleRate)
		left := math.Sin(2 * math.Pi * 1000 * t)
		right := math.Sin(2 * math.Pi * 1200 * t)
		inputAudio[i*2] = float32(left)
		inputAudio[i*2+1] = float32(right)
	}

	config := &ConverterConfig{
		EnableResampling:    true,
		EnableNormalization: true,
		TargetRMS:           0.1,
		TempBufferSize:      8192,
		NormalizeFactor:     32768.0,
	}

	converter, err := NewConverter(config)
	if err != nil {
		b.Fatal(err)
	}

	// Create input config (44.1kHz stereo WAV)
	inputConfig := &AudioConfig{
		Format:        FormatWAV,
		SampleRate:    sampleRate,
		Channels:      2,
		BitsPerSample: 16,
	}

	// Convert to bytes first (simulate real input)
	inputBytes, err := converter.ConvertFromFloat32(inputAudio, inputConfig)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()

	for b.Loop() {
		// Full pipeline: bytes -> float32 -> resample -> normalize -> mono
		floatData, err := converter.ConvertToFloat32(inputBytes, inputConfig)
		if err != nil {
			b.Fatal(err)
		}

		// This would be the processed audio ready for ASR
		_ = floatData
	}
}

// BenchmarkBufferPool benchmarks buffer pool operations
func BenchmarkBufferPool(b *testing.B) {
	sizes := []int{1024, 4096, 16384, 65536} // Different buffer sizes

	b.ReportAllocs()

	for b.Loop() {
		for _, size := range sizes {
			buf := DefaultAudioBufferPool.Get(size)
			// Simulate some work with the buffer
			for j := 0; j < len(buf) && j < 100; j++ {
				buf[j] = float32(j)
			}
			DefaultAudioBufferPool.Put(buf)
		}
	}
}
