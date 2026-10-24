package audio

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/go-audio/wav"
)

// Converter handles audio format conversion
type Converter struct {
	resampler *Resampler
	config    *ConverterConfig
}

// ConverterConfig represents converter configuration
type ConverterConfig struct {
	EnableResampling    bool    `json:"enable_resampling"`
	EnableNormalization bool    `json:"enable_normalization"`
	TargetRMS           float32 `json:"target_rms"`
	TempBufferSize      int     `json:"temp_buffer_size"`
	NormalizeFactor     float32 `json:"normalize_factor"`
}

// DefaultConverterConfig returns default converter configuration
func DefaultConverterConfig() *ConverterConfig {
	return &ConverterConfig{
		EnableResampling:    true,
		EnableNormalization: true,
		TargetRMS:           0.1, // Standard normalization level
		TempBufferSize:      8192,
		NormalizeFactor:     32768.0,
	}
}

// Validate validates converter configuration
func (c *ConverterConfig) Validate() error {
	var errs []error

	if c.TargetRMS <= 0 {
		errs = append(errs, fmt.Errorf("target RMS must be positive, got %f", c.TargetRMS))
	}

	if c.TargetRMS > 1.0 {
		errs = append(errs, fmt.Errorf("target RMS must not exceed 1.0, got %f", c.TargetRMS))
	}

	if c.TempBufferSize <= 0 {
		errs = append(errs, fmt.Errorf("temp buffer size must be positive, got %d", c.TempBufferSize))
	}

	if c.TempBufferSize > 1048576 { // 1MB limit
		errs = append(errs, fmt.Errorf("temp buffer size must not exceed 1MB, got %d", c.TempBufferSize))
	}

	if c.NormalizeFactor <= 0 {
		errs = append(errs, fmt.Errorf("normalize factor must be positive, got %f", c.NormalizeFactor))
	}

	if len(errs) > 0 {
		return fmt.Errorf("converter config validation failed: %v", errs)
	}

	return nil
}

// NewConverter creates a new audio converter
func NewConverter(config *ConverterConfig) (*Converter, error) {
	if config == nil {
		config = DefaultConverterConfig()
	}

	// Validate config
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid converter configuration: %w", err)
	}

	resampleConfig := &ResampleConfig{
		Method:       MethodCubic,
		Quality:      QualityMedium,
		FilterLength: 16,
		UseSIMD:      false, // SIMD integration deferred
	}

	return &Converter{
		resampler: NewResampler(resampleConfig),
		config:    config,
	}, nil
}

// ConvertAudio converts audio data between different formats and configurations.
//
// This function performs format conversion, sample rate resampling, and channel
// conversion as needed. It decodes WAV, PCM, FLAC, MP3 and Ogg/Vorbis and
// encodes WAV and PCM; M4A/AAC and compressed encoding return errors. To turn
// an upload into mono samples for speech recognition, use DecodeMono.
//
// Supported conversions:
//   - Format conversion: WAV ↔ PCM
//   - Sample rate conversion: any rate to any rate using high-quality resampling
//   - Channel conversion: mono ↔ stereo with proper downmixing/upmixing
//   - Normalization: optional audio level normalization
//
// Parameters:
//   - inputData: raw audio data in the input format
//   - inputConfig: configuration describing the input audio format
//   - outputConfig: configuration describing the desired output format
//
// Returns converted audio data or an error if conversion fails.
//
// Thread-safe: can be called concurrently from multiple goroutines.
func (c *Converter) ConvertAudio(inputData []byte, inputConfig, outputConfig *AudioConfig) ([]byte, error) {
	if len(inputData) == 0 {
		return nil, fmt.Errorf("input data is empty")
	}

	if inputConfig == nil || outputConfig == nil {
		return nil, fmt.Errorf("input and output audio configs cannot be nil")
	}

	// Validate configurations
	if err := inputConfig.ValidateConfig(); err != nil {
		return nil, fmt.Errorf("invalid input config: %v", err)
	}
	if err := outputConfig.ValidateConfig(); err != nil {
		return nil, fmt.Errorf("invalid output config: %v", err)
	}

	// Parse input format
	samples, err := c.decodeAudio(inputData, inputConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to decode input: %v", err)
	}

	// Apply processing. Channels are reduced before resampling and added
	// after it, and multichannel audio is resampled per channel, so
	// interleaved frames are never mixed by the filter. The channel layout is
	// part of the requested output, so it is converted whether or not rate
	// conversion is enabled.
	channels := inputConfig.Channels
	if outputConfig.Channels < channels {
		if samples, err = c.resampler.ConvertChannels(samples, channels, outputConfig.Channels); err != nil {
			return nil, fmt.Errorf("failed to convert channels: %v", err)
		}
		channels = outputConfig.Channels
	}
	if inputConfig.SampleRate != outputConfig.SampleRate {
		if !c.config.EnableResampling {
			// Passing the samples through would label them with the wrong rate.
			return nil, fmt.Errorf("sample rate conversion %d Hz -> %d Hz requested but resampling is disabled", inputConfig.SampleRate, outputConfig.SampleRate)
		}
		if samples, err = resampleInterleaved(c.resampler, samples, channels, inputConfig.SampleRate, outputConfig.SampleRate); err != nil {
			return nil, fmt.Errorf("failed to resample: %v", err)
		}
	}
	if outputConfig.Channels > channels {
		if samples, err = c.resampler.ConvertChannels(samples, channels, outputConfig.Channels); err != nil {
			return nil, fmt.Errorf("failed to convert channels: %v", err)
		}
	}

	// Normalize if enabled
	if c.config.EnableNormalization {
		samples = c.resampler.NormalizeAudio(samples, c.config.TargetRMS)
	}

	// Encode to output format
	outputData, err := c.encodeAudio(samples, outputConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to encode output: %v", err)
	}

	return outputData, nil
}

// resampleInterleaved resamples each channel of interleaved audio separately.
func resampleInterleaved(r *Resampler, samples []float32, channels, sourceRate, targetRate int) ([]float32, error) {
	if channels <= 1 {
		return r.Resample(samples, sourceRate, targetRate)
	}
	frames := len(samples) / channels
	var out []float32
	plane := make([]float32, frames)
	for ch := range channels {
		for i := range frames {
			plane[i] = samples[i*channels+ch]
		}
		resampled, err := r.Resample(plane, sourceRate, targetRate)
		if err != nil {
			return nil, err
		}
		if out == nil {
			out = make([]float32, len(resampled)*channels)
		}
		for i, v := range resampled {
			out[i*channels+ch] = v
		}
	}
	return out, nil
}

// decodeAudio decodes audio data to float32 samples
func (c *Converter) decodeAudio(data []byte, config *AudioConfig) ([]float32, error) {
	switch config.Format {
	case FormatWAV:
		return c.decodeWAV(data, config)
	case FormatPCM:
		return c.decodePCM(data, config)
	case FormatFLAC:
		return c.decodeFLAC(data, config)
	case FormatMP3:
		return c.decodeMP3(data, config)
	case FormatOGG:
		return c.decodeOGG(data, config)
	case FormatM4A:
		return c.decodeM4A(data, config)
	default:
		return nil, fmt.Errorf("unsupported input format: %s", config.Format)
	}
}

// encodeAudio encodes float32 samples to audio data
func (c *Converter) encodeAudio(samples []float32, config *AudioConfig) ([]byte, error) {
	switch config.Format {
	case FormatWAV:
		return c.encodeWAV(samples, config)
	case FormatPCM:
		return c.encodePCM(samples, config)
	case FormatFLAC:
		return c.encodeFLAC(samples, config)
	case FormatMP3:
		return c.encodeMP3(samples, config)
	case FormatOGG:
		return c.encodeOGG(samples, config)
	case FormatM4A:
		return c.encodeM4A(samples, config)
	default:
		return nil, fmt.Errorf("unsupported output format: %s", config.Format)
	}
}

// decodeWAV decodes WAV format using the RIFF/WAV parser instead of assuming a 44-byte header.
func (c *Converter) decodeWAV(data []byte, config *AudioConfig) ([]float32, error) {
	if err := validateWAVChunks(data); err != nil {
		return nil, err
	}
	decoder := wav.NewDecoder(bytes.NewReader(data))
	if !decoder.IsValidFile() {
		return nil, fmt.Errorf("invalid WAV file")
	}
	if decoder.WavAudioFormat != 1 {
		return nil, fmt.Errorf("unsupported WAV audio format %d: only PCM is supported", decoder.WavAudioFormat)
	}
	if int(decoder.NumChans) != config.Channels {
		return nil, fmt.Errorf("WAV channel count %d does not match config channels %d", decoder.NumChans, config.Channels)
	}
	if int(decoder.SampleRate) != config.SampleRate {
		return nil, fmt.Errorf("WAV sample rate %d does not match config sample rate %d", decoder.SampleRate, config.SampleRate)
	}
	if int(decoder.BitDepth) != config.BitsPerSample {
		return nil, fmt.Errorf("WAV bit depth %d does not match config bits per sample %d", decoder.BitDepth, config.BitsPerSample)
	}
	// A data chunk that claims more bytes than the file holds is a truncated
	// file; the converter reports it rather than decoding a prefix.
	if start, declared, ok := wavChunk(data, "data"); ok {
		size := uint64(len(data) - start)
		if declared != unknownWAVDataSize {
			if uint64(start)+uint64(declared) > uint64(len(data)) {
				return nil, fmt.Errorf("failed to decode WAV PCM data: %w", io.ErrUnexpectedEOF)
			}
			size = uint64(declared)
		}
		if size%uint64(config.GetBytesPerSample()) != 0 {
			return nil, fmt.Errorf("WAV PCM data size not aligned with sample size")
		}
	}

	decoded, err := decodeWAVNative(data, DecodeLimits{})
	if err != nil {
		return nil, err
	}
	samples := decoded.Samples
	if config.BitsPerSample == 16 && c.config.NormalizeFactor != 32768 {
		scale := 32768 / c.config.NormalizeFactor
		for i := range samples {
			samples[i] *= scale
		}
	}
	return samples, nil
}

func (c *Converter) decodePCMBytesInto(samples []float32, data []byte, config *AudioConfig) error {
	bytesPerSample := config.GetBytesPerSample()
	if len(data) != len(samples)*bytesPerSample {
		return fmt.Errorf("PCM data size not aligned with sample size")
	}

	switch config.BitsPerSample {
	case 8:
		for i, val := range data {
			samples[i] = float32(int(val)-128) / 128.0
		}
	case 16:
		for i := range samples {
			offset := i * 2
			val := int16(binary.LittleEndian.Uint16(data[offset : offset+2]))
			samples[i] = float32(val) / c.config.NormalizeFactor
		}
	case 24:
		for i := range samples {
			offset := i * 3
			val := int32(data[offset]) | int32(data[offset+1])<<8 | int32(data[offset+2])<<16
			if val&0x800000 != 0 {
				val |= ^int32(0xFFFFFF)
			}
			samples[i] = float32(val) / 8388608.0
		}
	case 32:
		for i := range samples {
			offset := i * 4
			val := int32(binary.LittleEndian.Uint32(data[offset : offset+4]))
			samples[i] = float32(val) / 2147483648.0
		}
	default:
		return fmt.Errorf("unsupported bits per sample: %d", config.BitsPerSample)
	}

	return nil
}

func (c *Converter) decodePCM(data []byte, config *AudioConfig) ([]float32, error) {
	bytesPerSample := config.GetBytesPerSample()
	if len(data)%bytesPerSample != 0 {
		return nil, fmt.Errorf("PCM data size not aligned with sample size")
	}

	sampleCount := len(data) / bytesPerSample
	samples := DefaultAudioBufferPool.Get(sampleCount)
	if err := c.decodePCMBytesInto(samples, data, config); err != nil {
		DefaultAudioBufferPool.Put(samples)
		return nil, err
	}
	return samples, nil
}

// encodeWAV encodes to WAV format (simplified implementation)
func (c *Converter) encodeWAV(samples []float32, config *AudioConfig) ([]byte, error) {
	if dataSize := len(samples) * config.GetBytesPerSample(); !wavDataSizeFits(dataSize) {
		return nil, fmt.Errorf("WAV data of %d bytes exceeds the 4 GiB RIFF size limit", dataSize)
	}
	pcmData, err := c.encodePCM(samples, config)
	if err != nil {
		return nil, err
	}

	// Create WAV header
	header := c.createWAVHeader(len(pcmData), config)
	return append(header, pcmData...), nil
}

// wavDataSizeFits reports whether a data chunk of dataSize bytes fits the
// 32-bit RIFF size fields, which also count the 36 header bytes after them.
func wavDataSizeFits(dataSize int) bool {
	return dataSize >= 0 && uint64(dataSize) <= math.MaxUint32-36
}

// createWAVHeader creates a WAV file header
func (c *Converter) createWAVHeader(dataSize int, config *AudioConfig) []byte {
	header := make([]byte, 44)

	// RIFF header
	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+dataSize)) // File size
	copy(header[8:12], "WAVE")

	// Format chunk
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)                                                                   // Chunk size
	binary.LittleEndian.PutUint16(header[20:22], 1)                                                                    // PCM format
	binary.LittleEndian.PutUint16(header[22:24], uint16(config.Channels))                                              // Channels
	binary.LittleEndian.PutUint32(header[24:28], uint32(config.SampleRate))                                            // Sample rate
	binary.LittleEndian.PutUint32(header[28:32], uint32(config.SampleRate*config.Channels*config.GetBytesPerSample())) // Byte rate
	binary.LittleEndian.PutUint16(header[32:34], uint16(config.Channels*config.GetBytesPerSample()))                   // Block align
	binary.LittleEndian.PutUint16(header[34:36], uint16(config.BitsPerSample))                                         // Bits per sample

	// Data chunk
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(dataSize)) // Data size

	return header
}

// encodePCM encodes to raw PCM data
func (c *Converter) encodePCM(samples []float32, config *AudioConfig) ([]byte, error) {
	bytesPerSample := config.GetBytesPerSample()
	outputSize := len(samples) * bytesPerSample
	output := make([]byte, outputSize)

	for i, sample := range samples {
		sample = clampSample(sample)
		switch config.BitsPerSample {
		case 8:
			output[i] = uint8(math.Round(float64((sample + 1.0) * 127.5)))
		case 16:
			offset := i * bytesPerSample
			binary.LittleEndian.PutUint16(output[offset:offset+2], uint16(floatToInt16(sample)))
		case 24:
			offset := i * bytesPerSample
			val := floatToInt24(sample)
			output[offset] = byte(val)
			output[offset+1] = byte(val >> 8)
			output[offset+2] = byte(val >> 16)
		case 32:
			offset := i * bytesPerSample
			binary.LittleEndian.PutUint32(output[offset:offset+4], uint32(floatToInt32(sample)))
		default:
			return nil, fmt.Errorf("unsupported bits per sample: %d", config.BitsPerSample)
		}
	}

	return output, nil
}

func clampSample(sample float32) float32 {
	if sample > 1 {
		return 1
	}
	if sample < -1 {
		return -1
	}
	return sample
}

func floatToInt16(sample float32) int16 {
	if sample <= -1 {
		return -32768
	}
	return int16(math.Round(float64(sample * 32767)))
}

func floatToInt24(sample float32) int32 {
	if sample <= -1 {
		return -8388608
	}
	return int32(math.Round(float64(sample * 8388607)))
}

func floatToInt32(sample float32) int32 {
	if sample <= -1 {
		return math.MinInt32
	}
	if sample >= 1 {
		return math.MaxInt32
	}
	return int32(math.Round(float64(sample) * float64(math.MaxInt32)))
}

// validateDecodedFormat checks a decoded stream's native rate/channels against
// the caller's expectations. A zero config value means "accept whatever the file
// declares", matching how the compressed formats carry their own rate/channels.
func validateDecodedFormat(config *AudioConfig, sampleRate, channels int) error {
	if config.SampleRate != 0 && config.SampleRate != sampleRate {
		return fmt.Errorf("decoded sample rate %d does not match config sample rate %d", sampleRate, config.SampleRate)
	}
	if config.Channels != 0 && config.Channels != channels {
		return fmt.Errorf("decoded channel count %d does not match config channels %d", channels, config.Channels)
	}
	return nil
}

// decodeFLAC decodes a FLAC stream (pure Go, no CGO) into interleaved float32
// samples in [-1, 1].
func (c *Converter) decodeFLAC(data []byte, config *AudioConfig) ([]float32, error) {
	decoded, err := decodeFLACNative(data, DecodeLimits{})
	if err != nil {
		return nil, err
	}
	if err := validateDecodedFormat(config, decoded.SampleRate, decoded.Channels); err != nil {
		return nil, err
	}
	return decoded.Samples, nil
}

func (c *Converter) encodeFLAC(samples []float32, config *AudioConfig) ([]byte, error) {
	return nil, fmt.Errorf("FLAC encoding not supported - requires libFLAC CGO library. Use WAV format instead")
}

// decodeMP3 decodes an MP3 stream (pure Go, no CGO) into float32 samples in
// [-1, 1]. go-mp3 always emits stereo, duplicating mono sources, so a mono
// config gets the channel average instead of an error.
func (c *Converter) decodeMP3(data []byte, config *AudioConfig) ([]float32, error) {
	decoded, err := decodeMP3Native(data, DecodeLimits{})
	if err != nil {
		return nil, err
	}
	if config.Channels == 1 {
		mono, err := (&Decoded{Samples: decoded.Samples, SampleRate: decoded.SampleRate, Channels: 2}).Mono(decoded.SampleRate)
		if err != nil {
			return nil, err
		}
		decoded = &Decoded{Samples: mono, SampleRate: decoded.SampleRate, Channels: 1}
	}
	if err := validateDecodedFormat(config, decoded.SampleRate, decoded.Channels); err != nil {
		return nil, err
	}
	return decoded.Samples, nil
}

func (c *Converter) encodeMP3(samples []float32, config *AudioConfig) ([]byte, error) {
	return nil, fmt.Errorf("MP3 encoding not supported - requires libmp3lame CGO library. Use WAV format instead")
}

// decodeOGG decodes an Ogg/Vorbis stream (pure Go, no CGO) into interleaved
// float32 samples in [-1, 1].
func (c *Converter) decodeOGG(data []byte, config *AudioConfig) ([]float32, error) {
	decoded, err := Decode(data, FormatOGG, DecodeLimits{})
	if err != nil {
		return nil, err
	}
	if err := validateDecodedFormat(config, decoded.SampleRate, decoded.Channels); err != nil {
		return nil, err
	}
	return decoded.Samples, nil
}

func (c *Converter) encodeOGG(samples []float32, config *AudioConfig) ([]byte, error) {
	return nil, fmt.Errorf("OGG encoding not supported - requires libvorbis CGO library. Use WAV format instead")
}

// decodeM4A remains unsupported: there is no maintained pure-Go AAC/M4A decoder,
// and VoiceKit avoids CGO for audio ingestion.
func (c *Converter) decodeM4A(data []byte, config *AudioConfig) ([]float32, error) {
	return nil, fmt.Errorf("M4A/AAC decoding is not supported (no pure-Go decoder); transcode to WAV, FLAC, MP3, or Ogg first")
}

func (c *Converter) encodeM4A(samples []float32, config *AudioConfig) ([]byte, error) {
	return nil, fmt.Errorf("M4A encoding not supported - requires Core Audio or ffmpeg CGO library. Use WAV format instead")
}

// ConvertToFloat32 converts compressed or encoded audio data to float32 samples.
//
// This is the primary entry point for converting audio data into the internal
// float32 format used by VoiceKit for processing. The samples are normalized
// to the range [-1, 1] for consistent processing across different input formats.
//
// Supported input formats:
//   - WAV (various bit depths and channel configurations)
//   - PCM (raw linear PCM data)
//   - Compressed formats: NOT SUPPORTED (returns informative error)
//
// Parameters:
//   - data: raw audio data bytes
//   - config: audio configuration describing the input format
//
// Returns float32 samples in range [-1, 1] or an error if conversion fails.
//
// Thread-safe: can be called concurrently from multiple goroutines.
func (c *Converter) ConvertToFloat32(data []byte, config *AudioConfig) ([]float32, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("input data cannot be empty")
	}
	if config == nil {
		return nil, fmt.Errorf("audio config cannot be nil")
	}
	if err := config.ValidateConfig(); err != nil {
		return nil, fmt.Errorf("invalid audio config: %v", err)
	}
	return c.decodeAudio(data, config)
}

// ConvertFromFloat32 converts float32 samples to specified audio format
func (c *Converter) ConvertFromFloat32(samples []float32, config *AudioConfig) ([]byte, error) {
	if len(samples) == 0 {
		return nil, fmt.Errorf("audio samples cannot be empty")
	}
	if config == nil {
		return nil, fmt.Errorf("audio config cannot be nil")
	}
	if err := config.ValidateConfig(); err != nil {
		return nil, fmt.Errorf("invalid audio config: %v", err)
	}
	return c.encodeAudio(samples, config)
}

// GetSupportedConversions returns supported format conversions
// Note: Only uncompressed formats (WAV, PCM) are fully supported.
// Compressed formats (FLAC, MP3, OGG, M4A) require external CGO libraries.
func (c *Converter) GetSupportedConversions() map[string][]string {
	return map[string][]string{
		"wav": {"pcm"}, // WAV ↔ PCM conversion supported
		"pcm": {"wav"}, // PCM ↔ WAV conversion supported
		// Compressed formats not supported without CGO dependencies
	}
}

// ValidateConversion checks if a conversion is supported
func (c *Converter) ValidateConversion(from, to AudioFormat) bool {
	supported, exists := c.GetSupportedConversions()[string(from)]
	if !exists {
		return false
	}

	for _, format := range supported {
		if format == string(to) {
			return true
		}
	}
	return false
}

// ParseWAVFile parses WAV file and returns audio data
func (c *Converter) ParseWAVFile(data []byte) ([]float32, int, error) {
	if err := validateWAVChunks(data); err != nil {
		return nil, 0, err
	}
	decoder := wav.NewDecoder(bytes.NewReader(data))
	if !decoder.IsValidFile() {
		return nil, 0, fmt.Errorf("invalid WAV file")
	}
	sampleRate := int(decoder.SampleRate)
	numChannels := int(decoder.NumChans)
	if numChannels > 2 {
		return nil, 0, fmt.Errorf("unsupported number of channels: %d", numChannels)
	}
	if decoder.WavAudioFormat != 1 {
		return nil, 0, fmt.Errorf("unsupported WAV audio format %d: only PCM is supported", decoder.WavAudioFormat)
	}
	config := &AudioConfig{Format: FormatWAV, SampleRate: sampleRate, Channels: numChannels, BitsPerSample: int(decoder.BitDepth)}
	samples, err := c.decodeWAV(data, config)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to decode audio: %v", err)
	}
	if numChannels == 2 {
		mono, err := (&Decoded{Samples: samples, SampleRate: sampleRate, Channels: 2}).Mono(sampleRate)
		if err != nil {
			return nil, 0, err
		}
		samples = mono
	}
	return samples, sampleRate, nil
}
