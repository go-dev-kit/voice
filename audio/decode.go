package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"time"

	"github.com/go-audio/wav"
	"github.com/hajimehoshi/go-mp3"
	"github.com/jfreymuth/oggvorbis"
	"github.com/mewkiz/flac"
)

// WAV format tags this package decodes.
const (
	wavFormatPCM        = 1
	wavFormatFloat      = 3
	wavFormatExtensible = 0xFFFE
)

// Decoded is an audio buffer at the rate and channel count its file declared.
type Decoded struct {
	// Samples are interleaved when Channels > 1, in [-1, 1].
	Samples    []float32
	SampleRate int
	Channels   int
}

// DecodeLimits bound what one decode may produce. Compressed formats expand
// many times over, so a small upload can otherwise decode to gigabytes.
type DecodeLimits struct {
	// MaxDuration rejects audio longer than this; zero means no limit.
	MaxDuration time.Duration
}

// DurationLimitError reports audio longer than DecodeLimits.MaxDuration.
// Decoding stops as soon as the limit is passed.
type DurationLimitError struct {
	Limit time.Duration
}

func (e *DurationLimitError) Error() string {
	return fmt.Sprintf("audio is longer than the %s limit", e.Limit)
}

// maxFrames is the frame budget for a stream at sampleRate; -1 means none.
func (l DecodeLimits) maxFrames(sampleRate int) int {
	if l.MaxDuration <= 0 {
		return -1
	}
	// Float math: Duration*rate overflows int64 for long limits at high rates.
	frames := l.MaxDuration.Seconds() * float64(sampleRate)
	if frames >= math.MaxInt32 {
		return math.MaxInt32
	}
	return int(frames)
}

func (l DecodeLimits) check(frames, sampleRate int) error {
	if limit := l.maxFrames(sampleRate); limit >= 0 && frames > limit {
		return &DurationLimitError{Limit: l.MaxDuration}
	}
	return nil
}

// Decode decodes a complete WAV (integer PCM or IEEE float), FLAC, MP3 or
// Ogg/Vorbis file at whatever rate and channel count it declares.
func Decode(data []byte, format AudioFormat, limits DecodeLimits) (*Decoded, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("input data cannot be empty")
	}
	switch format {
	case FormatWAV:
		return decodeWAVNative(data, limits)
	case FormatFLAC:
		return decodeFLACNative(data, limits)
	case FormatMP3:
		return decodeMP3Native(data, limits)
	case FormatOGG:
		return decodeOGGNative(data, limits)
	case FormatM4A, FormatAAC:
		return nil, fmt.Errorf("%s decoding is not supported (no pure-Go decoder); transcode to WAV, FLAC, MP3, or Ogg first", format)
	default:
		return nil, fmt.Errorf("unsupported input format: %s", format)
	}
}

// DecodeMono decodes a complete file (see Decode), averages its channels to
// mono, and resamples it to targetRate. This is the usual first step before
// speech recognition, which expects mono audio at the model's rate.
func DecodeMono(data []byte, format AudioFormat, targetRate int, limits DecodeLimits) ([]float32, error) {
	if targetRate <= 0 {
		return nil, fmt.Errorf("targetRate must be positive, got %d", targetRate)
	}
	decoded, err := Decode(data, format, limits)
	if err != nil {
		return nil, err
	}
	return decoded.Mono(targetRate)
}

// Mono averages the channels and resamples to targetRate.
func (d *Decoded) Mono(targetRate int) ([]float32, error) {
	if d.Channels <= 0 || d.SampleRate <= 0 {
		return nil, fmt.Errorf("invalid decoded audio: %d Hz, %d channels", d.SampleRate, d.Channels)
	}
	mono := d.Samples
	if d.Channels > 1 {
		frames := len(d.Samples) / d.Channels
		mono = make([]float32, frames)
		scale := 1 / float32(d.Channels)
		for i := range frames {
			var sum float32
			for _, v := range d.Samples[i*d.Channels : (i+1)*d.Channels] {
				sum += v
			}
			mono[i] = sum * scale
		}
	}
	if d.SampleRate == targetRate || len(mono) == 0 {
		return mono, nil
	}
	resampled, err := NewResampler(&ResampleConfig{Method: MethodSinc, Quality: QualityHigh, FilterLength: 32}).Resample(mono, d.SampleRate, targetRate)
	if err != nil {
		return nil, fmt.Errorf("resample %d Hz to %d Hz: %w", d.SampleRate, targetRate, err)
	}
	return resampled, nil
}

func decodeWAVNative(data []byte, limits DecodeLimits) (*Decoded, error) {
	if err := validateWAVChunks(data); err != nil {
		return nil, err
	}
	decoder := wav.NewDecoder(bytes.NewReader(data))
	if !decoder.IsValidFile() {
		return nil, fmt.Errorf("invalid WAV file")
	}
	if decoder.NumChans == 0 || decoder.SampleRate == 0 {
		return nil, fmt.Errorf("invalid WAV header: %d Hz, %d channels", decoder.SampleRate, decoder.NumChans)
	}
	if err := decoder.FwdToPCM(); err != nil {
		return nil, fmt.Errorf("failed to locate WAV PCM data: %w", err)
	}
	start, declared, ok := wavChunk(data, "data")
	if decoder.PCMChunk == nil || !ok {
		return nil, fmt.Errorf("WAV PCM chunk not found")
	}
	// Slice the payload out of the input rather than reading the library's
	// chunk reader: that reader counts the RIFF pad byte of an odd-sized chunk
	// as audio and reads nothing for a streamed file whose size is unknown. The
	// slice is bounded by the bytes actually present, whatever size the header
	// claims, so a forged size cannot drive an allocation.
	end := uint64(len(data))
	if declared != unknownWAVDataSize {
		end = min(end, uint64(start)+uint64(declared))
	}
	raw := data[start:end]
	if bytesPerFrame := int(decoder.NumChans) * max(int(decoder.BitDepth)/8, 1); bytesPerFrame > 0 {
		if err := limits.check(len(raw)/bytesPerFrame, int(decoder.SampleRate)); err != nil {
			return nil, err
		}
	}

	format := int(decoder.WavAudioFormat)
	bits := int(decoder.BitDepth)
	if format == wavFormatExtensible {
		format = wavExtensibleSubFormat(data)
	}
	samples, err := pcmToFloat32(raw, format, bits)
	if err != nil {
		return nil, err
	}
	return &Decoded{Samples: samples, SampleRate: int(decoder.SampleRate), Channels: int(decoder.NumChans)}, nil
}

// maxWAVFormatChunk bounds the fmt chunk; real ones are 16 to 40 bytes.
const maxWAVFormatChunk = 1024

// validateWAVChunks checks every chunk before the data chunk against the
// bytes actually present. The WAV library allocates fmt and LIST chunks at
// their declared size before reading them, so a forged size in a 44-byte file
// would otherwise allocate gigabytes.
func validateWAVChunks(data []byte) error {
	if len(data) < 12 {
		return fmt.Errorf("invalid WAV file")
	}
	for off := 12; off+8 <= len(data); {
		id := string(data[off : off+4])
		size := uint64(binary.LittleEndian.Uint32(data[off+4:]))
		if id == "data" {
			return nil
		}
		if id == "fmt " && size > maxWAVFormatChunk {
			return fmt.Errorf("invalid WAV file: fmt chunk claims %d bytes", size)
		}
		next := uint64(off) + 8 + size + size&1
		if uint64(off)+8+size > uint64(len(data)) {
			return fmt.Errorf("invalid WAV file: %q chunk claims %d bytes, only %d remain", id, size, len(data)-off-8)
		}
		off = int(next)
	}
	return nil
}

// unknownWAVDataSize is the data chunk size streaming writers (ffmpeg writing
// to a pipe) leave behind when they cannot seek back to fill in the length.
const unknownWAVDataSize = 0xFFFFFFFF

// wavExtensibleSubFormat returns the format tag held in the first two bytes of
// a WAVE_FORMAT_EXTENSIBLE SubFormat GUID (offset 24 of the fmt payload), or 0
// when the fmt chunk is too short to carry one.
func wavExtensibleSubFormat(data []byte) int {
	start, size, ok := wavChunk(data, "fmt ")
	if !ok || size < 26 || start+26 > len(data) {
		return 0
	}
	return int(binary.LittleEndian.Uint16(data[start+24:]))
}

// wavChunk finds the first chunk with the given id in a RIFF/WAVE file. It
// returns the offset of the chunk's payload and the size its header declares.
// The wav decoder reports a data size rounded up to even, and wraps it to 0
// for 0xFFFFFFFF.
func wavChunk(data []byte, id string) (start int, declared uint32, ok bool) {
	if len(data) < 12 {
		return 0, 0, false
	}
	off := 12
	for off+8 <= len(data) {
		size := binary.LittleEndian.Uint32(data[off+4:])
		if string(data[off:off+4]) == id {
			return off + 8, size, true
		}
		off += 8 + int(size) + int(size&1)
		if off < 0 {
			return 0, 0, false
		}
	}
	return 0, 0, false
}

// pcmToFloat32 converts little-endian WAV sample data. A trailing partial
// sample is ignored, as players do.
func pcmToFloat32(raw []byte, format, bits int) ([]float32, error) {
	switch {
	case format == wavFormatFloat && bits == 32:
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, nil
	case format == wavFormatFloat && bits == 64:
		out := make([]float32, len(raw)/8)
		for i := range out {
			out[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
		return out, nil
	case format != wavFormatPCM:
		return nil, fmt.Errorf("unsupported WAV audio format %d with %d bits", format, bits)
	}
	switch bits {
	case 8:
		out := make([]float32, len(raw))
		for i, v := range raw {
			out[i] = float32(int(v)-128) / 128
		}
		return out, nil
	case 16:
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = float32(int16(binary.LittleEndian.Uint16(raw[i*2:]))) / 32768
		}
		return out, nil
	case 24:
		out := make([]float32, len(raw)/3)
		for i := range out {
			v := int32(raw[i*3]) | int32(raw[i*3+1])<<8 | int32(raw[i*3+2])<<16
			if v&0x800000 != 0 {
				v |= ^int32(0xFFFFFF)
			}
			out[i] = float32(v) / 8388608
		}
		return out, nil
	case 32:
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = float32(int32(binary.LittleEndian.Uint32(raw[i*4:]))) / 2147483648
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported WAV bits per sample: %d", bits)
	}
}

func decodeFLACNative(data []byte, limits DecodeLimits) (*Decoded, error) {
	stream, err := flac.New(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize FLAC decoder: %w", err)
	}
	defer stream.Close()

	info := stream.Info
	if info.BitsPerSample == 0 || info.BitsPerSample > 32 {
		return nil, fmt.Errorf("unsupported FLAC bits per sample: %d", info.BitsPerSample)
	}
	if info.NChannels == 0 || info.SampleRate == 0 {
		return nil, fmt.Errorf("invalid FLAC stream info: %d Hz, %d channels", info.SampleRate, info.NChannels)
	}
	scale := float32(int64(1) << (info.BitsPerSample - 1))
	// The header's sample count is only a capacity hint, bounded by the input
	// size so a forged header cannot force a huge allocation.
	capacity := min(info.NSamples*uint64(info.NChannels), uint64(len(data))*4)
	samples := make([]float32, 0, capacity)
	frames := 0
	for {
		frame, err := stream.ParseNext()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to decode FLAC frame: %w", err)
		}
		n := len(frame.Subframes[0].Samples)
		frames += n
		if err := limits.check(frames, int(info.SampleRate)); err != nil {
			return nil, err
		}
		for i := range n {
			for _, subframe := range frame.Subframes {
				samples = append(samples, float32(subframe.Samples[i])/scale)
			}
		}
	}
	return &Decoded{Samples: samples, SampleRate: int(info.SampleRate), Channels: int(info.NChannels)}, nil
}

// decodeMP3Native decodes MP3; go-mp3 always emits 16-bit stereo.
func decodeMP3Native(data []byte, limits DecodeLimits) (*Decoded, error) {
	decoder, err := mp3.NewDecoder(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to initialize MP3 decoder: %w", err)
	}
	var reader io.Reader = decoder
	const bytesPerFrame = 4 // 16-bit stereo
	limit := limits.maxFrames(decoder.SampleRate())
	if limit >= 0 {
		reader = io.LimitReader(decoder, int64(limit+1)*bytesPerFrame)
	}
	raw, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode MP3 stream: %w", err)
	}
	if err := limits.check(len(raw)/bytesPerFrame, decoder.SampleRate()); err != nil {
		return nil, err
	}
	samples, err := pcmToFloat32(raw, wavFormatPCM, 16)
	if err != nil {
		return nil, err
	}
	return &Decoded{Samples: samples, SampleRate: decoder.SampleRate(), Channels: 2}, nil
}

func decodeOGGNative(data []byte, limits DecodeLimits) (*Decoded, error) {
	reader, err := oggvorbis.NewReader(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to decode Ogg/Vorbis stream: %w", err)
	}
	channels, rate := reader.Channels(), reader.SampleRate()
	if channels <= 0 || rate <= 0 {
		return nil, fmt.Errorf("invalid Ogg/Vorbis stream: %d Hz, %d channels", rate, channels)
	}
	var samples []float32
	buf := make([]float32, 8192*channels)
	for {
		n, err := reader.Read(buf)
		samples = append(samples, buf[:n]...)
		if limitErr := limits.check(len(samples)/channels, rate); limitErr != nil {
			return nil, limitErr
		}
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to decode Ogg/Vorbis stream: %w", err)
		}
	}
	return &Decoded{Samples: samples, SampleRate: rate, Channels: channels}, nil
}
