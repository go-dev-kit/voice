package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

func TestDecodeWAV(t *testing.T) {
	mono := []float64{0, 0.5, -0.5, 0.25, -1, 0.75}
	stereo := []float64{0.5, -0.5, 0.25, -0.25, 0.75, 0}

	tests := []struct {
		name     string
		spec     wavSpec
		payload  func(vals []float64) []byte
		vals     []float64
		wantChan int
		tol      float64
	}{
		{"pcm8", wavSpec{tag: 1, channels: 1, rate: 16000, bits: 8}, func(v []float64) []byte { return pcmBytes(v, 8) }, mono, 1, 1.0 / 128},
		{"pcm16", wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}, func(v []float64) []byte { return pcmBytes(v, 16) }, mono, 1, 1.0 / 32768},
		{"pcm24", wavSpec{tag: 1, channels: 1, rate: 16000, bits: 24}, func(v []float64) []byte { return pcmBytes(v, 24) }, mono, 1, 1e-6},
		{"pcm32", wavSpec{tag: 1, channels: 1, rate: 16000, bits: 32}, func(v []float64) []byte { return pcmBytes(v, 32) }, mono, 1, 1e-6},
		{"float32", wavSpec{tag: 3, channels: 1, rate: 16000, bits: 32}, func(v []float64) []byte { return floatBytes(v, 32) }, mono, 1, 1e-7},
		{"float64", wavSpec{tag: 3, channels: 1, rate: 16000, bits: 64}, func(v []float64) []byte { return floatBytes(v, 64) }, mono, 1, 1e-7},
		{"stereo pcm16", wavSpec{tag: 1, channels: 2, rate: 44100, bits: 16}, func(v []float64) []byte { return pcmBytes(v, 16) }, stereo, 2, 1.0 / 32768},
		{"stereo float32", wavSpec{tag: 3, channels: 2, rate: 48000, bits: 32}, func(v []float64) []byte { return floatBytes(v, 32) }, stereo, 2, 1e-7},
		{"extensible pcm16", wavSpec{extensible: true, subTag: 1, channels: 1, rate: 16000, bits: 16}, func(v []float64) []byte { return pcmBytes(v, 16) }, mono, 1, 1.0 / 32768},
		{"extensible pcm24 stereo", wavSpec{extensible: true, subTag: 1, channels: 2, rate: 48000, bits: 24}, func(v []float64) []byte { return pcmBytes(v, 24) }, stereo, 2, 1e-6},
		{"extensible float32", wavSpec{extensible: true, subTag: 3, channels: 1, rate: 16000, bits: 32}, func(v []float64) []byte { return floatBytes(v, 32) }, mono, 1, 1e-7},
		{"extensible float64", wavSpec{extensible: true, subTag: 3, channels: 1, rate: 16000, bits: 64}, func(v []float64) []byte { return floatBytes(v, 64) }, mono, 1, 1e-7},
		{"extensible pcm32", wavSpec{extensible: true, subTag: 1, channels: 1, rate: 16000, bits: 32}, func(v []float64) []byte { return pcmBytes(v, 32) }, mono, 1, 1e-6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := buildWAV(tt.spec, tt.payload(tt.vals))
			got, err := Decode(data, FormatWAV, DecodeLimits{})
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if got.SampleRate != tt.spec.rate || got.Channels != tt.wantChan {
				t.Fatalf("got %d Hz %d ch, want %d Hz %d ch", got.SampleRate, got.Channels, tt.spec.rate, tt.wantChan)
			}
			if len(got.Samples) != len(tt.vals) {
				t.Fatalf("got %d samples, want %d", len(got.Samples), len(tt.vals))
			}
			for i, want := range tt.vals {
				// 8-bit PCM cannot represent +1.0 exactly; clamp what the encoder clamped.
				if tt.spec.bits == 8 {
					want = math.Min(want, 127.0/128)
				}
				if math.Abs(float64(got.Samples[i])-want) > tt.tol {
					t.Errorf("sample %d: got %v want %v (tol %v)", i, got.Samples[i], want, tt.tol)
				}
			}
		})
	}
}

func TestDecodeWAVExactIntegerMapping(t *testing.T) {
	// Full-scale ends of each integer depth map to -1 and just below +1.
	tests := []struct {
		name    string
		bits    int
		payload []byte
		want    []float32
	}{
		{"8-bit", 8, []byte{0, 128, 255}, []float32{-1, 0, 127.0 / 128}},
		{"16-bit", 16, []byte{0x00, 0x80, 0x00, 0x00, 0xFF, 0x7F}, []float32{-1, 0, 32767.0 / 32768}},
		{"24-bit", 24, []byte{0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x7F}, []float32{-1, 0, 8388607.0 / 8388608}},
		{"32-bit", 32, []byte{0x00, 0x00, 0x00, 0x80, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0x7F}, []float32{-1, 0, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := buildWAV(wavSpec{tag: 1, channels: 1, rate: 8000, bits: tt.bits}, tt.payload)
			got, err := Decode(data, FormatWAV, DecodeLimits{})
			if err != nil {
				t.Fatal(err)
			}
			assertSliceNear(t, got.Samples, tt.want, 0)
		})
	}
}

func TestDecodeWAVOddTrailingByte(t *testing.T) {
	// A data chunk holding one 16-bit sample plus a stray byte: the partial
	// sample is dropped, as players do, and the odd length does not error.
	data := buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}, []byte{0x00, 0x40, 0x7F})
	got, err := Decode(data, FormatWAV, DecodeLimits{})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	assertSliceNear(t, got.Samples, []float32{0.5}, 1e-6)
}

func TestDecodeWAVChunksAfterData(t *testing.T) {
	// Metadata chunks written after the audio (LIST, id3, iXML) are not samples.
	spec := wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}
	vals := []float64{0.25, -0.25, 0.5, -0.5}
	data := buildWAV(spec, pcmBytes(vals, 16))
	data = append(data, "LIST"...)
	data = binary.LittleEndian.AppendUint32(data, 8)
	data = append(data, "INFOabcd"...)

	got, err := Decode(data, FormatWAV, DecodeLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != len(vals) {
		t.Fatalf("got %d samples, want %d: trailing chunk was decoded as audio", len(got.Samples), len(vals))
	}

	cfg := &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 1, BitsPerSample: 16}
	conv, err := NewConverter(DefaultConverterConfig())
	if err != nil {
		t.Fatal(err)
	}
	samples, err := conv.ConvertToFloat32(data, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != len(vals) {
		t.Fatalf("converter got %d samples, want %d", len(samples), len(vals))
	}
}

func TestDecodeWAVUnknownDataSize(t *testing.T) {
	// ffmpeg writing to a pipe cannot seek back, so it leaves 0xFFFFFFFF as the
	// data size: the audio runs to the end of the file.
	spec := wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}
	vals := []float64{0.25, -0.25, 0.5, -0.5, 0.125, 0}
	data := buildWAVClaiming(spec, pcmBytes(vals, 16), 0xFFFFFFFF)

	got, err := Decode(data, FormatWAV, DecodeLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Samples) != len(vals) {
		t.Fatalf("got %d samples, want %d", len(got.Samples), len(vals))
	}

	cfg := &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 1, BitsPerSample: 16}
	samples, err := newTestConverter(t).ConvertToFloat32(data, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != len(vals) {
		t.Fatalf("converter got %d samples, want %d", len(samples), len(vals))
	}
}

func TestDecodeWAVOddSizedChunkBeforeData(t *testing.T) {
	// A 3-byte chunk is followed by a pad byte before the next chunk header.
	le := binary.LittleEndian
	var data []byte
	data = append(data, "RIFF"...)
	data = le.AppendUint32(data, 0)
	data = append(data, "WAVE"...)
	data = append(data, "fmt "...)
	data = le.AppendUint32(data, 16)
	data = le.AppendUint16(data, 1)
	data = le.AppendUint16(data, 1)
	data = le.AppendUint32(data, 16000)
	data = le.AppendUint32(data, 32000)
	data = le.AppendUint16(data, 2)
	data = le.AppendUint16(data, 16)
	data = append(data, "odd "...)
	data = le.AppendUint32(data, 3)
	data = append(data, 1, 2, 3, 0) // payload plus pad
	data = append(data, "data"...)
	data = le.AppendUint32(data, 4)
	data = append(data, pcmBytes([]float64{0.5, -0.5}, 16)...)
	le.PutUint32(data[4:], uint32(len(data)-8))

	got, err := Decode(data, FormatWAV, DecodeLimits{})
	if err != nil {
		t.Fatal(err)
	}
	assertSliceNear(t, got.Samples, []float32{0.5, -0.5}, 1e-6)
}

func TestWAVDataChunkLocator(t *testing.T) {
	good := buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}, make([]byte, 10))
	start, declared, ok := wavChunk(good, "data")
	if !ok || declared != 10 || start != len(good)-10 {
		t.Errorf("got start=%d declared=%d ok=%v", start, declared, ok)
	}
	for name, data := range map[string][]byte{
		"too short":          []byte("RIFF"),
		"no data chunk":      good[:36],
		"chunk past the end": append(append([]byte(nil), good[:36]...), "junk\xff\xff\xff\xff"...),
	} {
		if _, _, ok := wavChunk(data, "data"); ok {
			t.Errorf("%s: unexpectedly found a data chunk", name)
		}
	}
}

func TestDecodeWAVEmptyDataChunk(t *testing.T) {
	data := buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}, nil)
	got, err := Decode(data, FormatWAV, DecodeLimits{})
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(got.Samples) != 0 || got.SampleRate != 16000 || got.Channels != 1 {
		t.Fatalf("got %+v, want empty 16 kHz mono", got)
	}
}

func TestDecodeErrors(t *testing.T) {
	pcm := pcmBytes([]float64{0.1, 0.2}, 16)
	tests := []struct {
		name    string
		data    []byte
		format  AudioFormat
		wantMsg string // substring; empty means any error
	}{
		{"nil input", nil, FormatWAV, "empty"},
		{"empty input", []byte{}, FormatFLAC, "empty"},
		{"not a wav", []byte("this is definitely not a RIFF file at all"), FormatWAV, "invalid WAV"},
		{"truncated header", []byte("RIFF\x04\x00\x00\x00WAVE"), FormatWAV, ""},
		{"adpcm tag", buildWAV(wavSpec{tag: 2, channels: 1, rate: 16000, bits: 16}, pcm), FormatWAV, "unsupported WAV audio format"},
		{"alaw tag", buildWAV(wavSpec{tag: 6, channels: 1, rate: 8000, bits: 8}, []byte{1, 2}), FormatWAV, "unsupported WAV audio format"},
		{"float with 16 bits", buildWAV(wavSpec{tag: 3, channels: 1, rate: 16000, bits: 16}, pcm), FormatWAV, "unsupported WAV audio format"},
		{"pcm 12 bits", buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 12}, pcm), FormatWAV, "bits per sample"},
		{"zero channels", buildWAV(wavSpec{tag: 1, channels: 0, rate: 16000, bits: 16}, pcm), FormatWAV, ""},
		{"zero rate", buildWAV(wavSpec{tag: 1, channels: 1, rate: 0, bits: 16}, pcm), FormatWAV, ""},
		{"garbage flac", []byte("not flac"), FormatFLAC, "FLAC"},
		{"garbage mp3", []byte("not mp3 data, long enough to look at"), FormatMP3, ""},
		{"garbage ogg", []byte("not an ogg stream, long enough too"), FormatOGG, "Ogg"},
		{"m4a", []byte("x"), FormatM4A, "not supported"},
		{"aac", []byte("x"), FormatAAC, "not supported"},
		{"raw pcm has no header", pcm, FormatPCM, "unsupported input format"},
		{"unknown format", pcm, AudioFormat("opus"), "unsupported input format"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Decode(tt.data, tt.format, DecodeLimits{})
			if err == nil {
				t.Fatalf("expected an error, got %+v", got)
			}
			if got != nil {
				t.Errorf("expected nil result with error, got %+v", got)
			}
			if tt.wantMsg != "" && !strings.Contains(err.Error(), tt.wantMsg) {
				t.Errorf("error %q does not mention %q", err, tt.wantMsg)
			}
		})
	}
}

func TestDecodeTruncatedCompressedStreams(t *testing.T) {
	// Cutting a stream short must never panic; a decoder either errors or
	// returns the audio it could recover.
	for _, tt := range []struct {
		file   string
		format AudioFormat
	}{
		{"tone.flac", FormatFLAC}, {"tone.mp3", FormatMP3}, {"tone.ogg", FormatOGG},
	} {
		t.Run(tt.file, func(t *testing.T) {
			data := loadFixture(t, tt.file)
			for _, n := range []int{1, 4, 20, len(data) / 4, len(data) / 2, len(data) - 1} {
				got, err := Decode(data[:n], tt.format, DecodeLimits{})
				if err == nil && got == nil {
					t.Errorf("cut at %d: nil result without error", n)
				}
			}
		})
	}
}

// TestDecodeWAVForgedDataSize is the header-bomb case from the review: a
// 108-byte file whose data chunk claims hundreds of MiB must not make the
// decoder allocate anywhere near that.
func TestDecodeWAVForgedDataSize(t *testing.T) {
	spec := wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}
	payload := pcmBytes(float64s(sine(440, 16000, 32, 0.5)), 16)

	for _, claimed := range []uint32{256 << 20, 1 << 30, 0xFFFFFFFE, unknownWAVDataSize} {
		data := buildWAVClaiming(spec, payload, claimed)
		t.Run(fmt.Sprintf("native/%d", claimed), func(t *testing.T) {
			var got *Decoded
			var err error
			allocated := allocatedBytes(func() { got, err = Decode(data, FormatWAV, DecodeLimits{}) })
			if allocated > 1<<20 {
				t.Errorf("Decode of a %d byte file allocated %d bytes", len(data), allocated)
			}
			// Decoding only what is present is acceptable; so is an error.
			if err == nil && len(got.Samples) > len(payload)/2 {
				t.Errorf("decoded %d samples from %d payload bytes", len(got.Samples), len(payload))
			}
		})
		t.Run(fmt.Sprintf("converter/%d", claimed), func(t *testing.T) {
			conv := newTestConverter(t)
			cfg := &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 1, BitsPerSample: 16}
			var err error
			allocated := allocatedBytes(func() { _, err = conv.ConvertToFloat32(data, cfg) })
			if allocated > 1<<20 {
				t.Errorf("ConvertToFloat32 of a %d byte file allocated %d bytes", len(data), allocated)
			}
			// The file is truncated relative to its header, so it is rejected,
			// except for the "size unknown" marker, which means "to the end".
			if wantErr := claimed != unknownWAVDataSize; wantErr && err == nil {
				t.Error("expected an error for a data chunk larger than the file")
			} else if !wantErr && err != nil {
				t.Errorf("streamed WAV: %v", err)
			}
		})
	}
}

func TestDecodeFLACForgedStreamInfo(t *testing.T) {
	good := loadFixture(t, "tone.flac")
	want, err := Decode(good, FormatFLAC, DecodeLimits{})
	if err != nil {
		t.Fatal(err)
	}

	// STREAMINFO starts at byte 8: the total-sample count is the low nibble of
	// byte 21 plus bytes 22..25 (36 bits). Claim 2^36-1 samples.
	forged := append([]byte(nil), good...)
	forged[21] |= 0x0F
	copy(forged[22:26], []byte{0xFF, 0xFF, 0xFF, 0xFF})

	var got *Decoded
	allocated := allocatedBytes(func() { got, err = Decode(forged, FormatFLAC, DecodeLimits{}) })
	if allocated > 4<<20 {
		t.Errorf("forged STREAMINFO made Decode allocate %d bytes for a %d byte file", allocated, len(forged))
	}
	if err != nil {
		t.Fatalf("a forged sample count should not stop the frames decoding: %v", err)
	}
	if len(got.Samples) != len(want.Samples) {
		t.Errorf("got %d samples, want %d", len(got.Samples), len(want.Samples))
	}
}

func TestDecodeFixtures(t *testing.T) {
	tests := []struct {
		file         string
		format       AudioFormat
		wantChannels int
		minFrames    int
		maxFrames    int
	}{
		// The fixtures are 0.2 s, 16 kHz, 440 Hz tones. FLAC and Vorbis are
		// exact; the MP3 decoder adds encoder delay and frame padding and always
		// emits stereo.
		{"tone.flac", FormatFLAC, 1, 3200, 3200},
		{"tone.ogg", FormatOGG, 2, 3200, 3300},
		{"tone.mp3", FormatMP3, 2, 3200, 6000},
	}
	for _, tt := range tests {
		t.Run(tt.file, func(t *testing.T) {
			got, err := Decode(loadFixture(t, tt.file), tt.format, DecodeLimits{})
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if got.SampleRate != 16000 {
				t.Errorf("sample rate = %d, want 16000", got.SampleRate)
			}
			if got.Channels != tt.wantChannels {
				t.Errorf("channels = %d, want %d", got.Channels, tt.wantChannels)
			}
			frames := len(got.Samples) / got.Channels
			if len(got.Samples)%got.Channels != 0 {
				t.Errorf("%d samples is not a whole number of %d-channel frames", len(got.Samples), got.Channels)
			}
			if frames < tt.minFrames || frames > tt.maxFrames {
				t.Errorf("frames = %d, want in [%d, %d]", frames, tt.minFrames, tt.maxFrames)
			}
			assertPlausibleAudio(t, got.Samples, tt.minFrames)

			// The tone is a clear 440 Hz: it must dominate the spectrum, which a
			// bit-depth or scaling mistake would not preserve.
			mono, err := got.Mono(16000)
			if err != nil {
				t.Fatal(err)
			}
			if peak := dominantFrequency(mono[len(mono)/4:len(mono)*3/4], 16000); math.Abs(peak-440) > 20 {
				t.Errorf("dominant frequency = %.1f Hz, want about 440", peak)
			}
		})
	}
}

func TestDecodeMonoFixtures(t *testing.T) {
	for _, tt := range []struct {
		file   string
		format AudioFormat
	}{
		{"tone.flac", FormatFLAC}, {"tone.mp3", FormatMP3}, {"tone.ogg", FormatOGG},
	} {
		data := loadFixture(t, tt.file)
		native, err := Decode(data, tt.format, DecodeLimits{})
		if err != nil {
			t.Fatal(err)
		}
		nativeFrames := len(native.Samples) / native.Channels

		t.Run(tt.file+"/same rate", func(t *testing.T) {
			got, err := DecodeMono(data, tt.format, 16000, DecodeLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != nativeFrames {
				t.Errorf("got %d samples, want %d", len(got), nativeFrames)
			}
			assertPlausibleAudio(t, got, 3000)
		})
		t.Run(tt.file+"/to 8k", func(t *testing.T) {
			got, err := DecodeMono(data, tt.format, 8000, DecodeLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if want := expectedResampleLen(nativeFrames, 16000, 8000); len(got) != want {
				t.Errorf("got %d samples, want %d", len(got), want)
			}
		})
		t.Run(tt.file+"/to 48k", func(t *testing.T) {
			got, err := DecodeMono(data, tt.format, 48000, DecodeLimits{})
			if err != nil {
				t.Fatal(err)
			}
			if want := expectedResampleLen(nativeFrames, 16000, 48000); len(got) != want {
				t.Errorf("got %d samples, want %d", len(got), want)
			}
		})
	}
}

func TestDecodeMonoWAVStereo48kTo16k(t *testing.T) {
	// 1 s of a 1 kHz tone in both channels: the mono mix keeps its level.
	tone := sine(1000, 48000, 48000, 0.5)
	stereo := make([]float32, 2*len(tone))
	for i, v := range tone {
		stereo[2*i], stereo[2*i+1] = v, v
	}
	got, err := DecodeMono(pcm16WAV(48000, 2, stereo), FormatWAV, 16000, DecodeLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if want := expectedResampleLen(len(tone), 48000, 16000); len(got) != want {
		t.Fatalf("got %d samples, want %d", len(got), want)
	}
	want := 0.5 / math.Sqrt2
	if db := gainDB(rmsInterior(got, 0.05), want); math.Abs(db) > 0.5 {
		t.Errorf("level changed by %.2f dB", db)
	}
}

func TestDecodeMonoErrors(t *testing.T) {
	wav := pcm16WAV(16000, 1, sine(440, 16000, 160, 0.5))
	for _, rate := range []int{0, -16000} {
		if _, err := DecodeMono(wav, FormatWAV, rate, DecodeLimits{}); err == nil {
			t.Errorf("targetRate %d: expected an error", rate)
		}
	}
	if _, err := DecodeMono(nil, FormatWAV, 16000, DecodeLimits{}); err == nil {
		t.Error("empty input: expected an error")
	}
	// The resampler only supports 8-192 kHz; the failure must surface, not panic.
	if _, err := DecodeMono(wav, FormatWAV, 100, DecodeLimits{}); err == nil {
		t.Error("targetRate 100: expected a resample error")
	}
}

func TestDecodeLimitsWAV(t *testing.T) {
	const rate = 16000
	// 2 s mono: 32000 frames.
	data := pcm16WAV(rate, 1, make([]float32, 2*rate))
	framePeriod := time.Second / rate

	tests := []struct {
		name      string
		limit     time.Duration
		wantError bool
	}{
		{"zero is unlimited", 0, false},
		{"negative is unlimited", -time.Second, false},
		{"well below", time.Second, true},
		{"one frame below", 2*time.Second - framePeriod, true},
		{"exactly at", 2 * time.Second, false},
		{"above", 3 * time.Second, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checkLimit(t, data, FormatWAV, tt.limit, tt.wantError)
		})
	}

	t.Run("stereo counts frames not samples", func(t *testing.T) {
		stereo := pcm16WAV(rate, 2, make([]float32, 2*2*rate))
		checkLimit(t, stereo, FormatWAV, 2*time.Second, false)
		checkLimit(t, stereo, FormatWAV, 2*time.Second-framePeriod, true)
	})
}

func TestDecodeLimitsCompressed(t *testing.T) {
	for _, tt := range []struct {
		file   string
		format AudioFormat
	}{
		{"tone.flac", FormatFLAC}, {"tone.mp3", FormatMP3}, {"tone.ogg", FormatOGG},
	} {
		t.Run(tt.file, func(t *testing.T) {
			data := loadFixture(t, tt.file)
			full, err := Decode(data, tt.format, DecodeLimits{})
			if err != nil {
				t.Fatal(err)
			}
			frames := len(full.Samples) / full.Channels
			exact := time.Duration(frames) * time.Second / time.Duration(full.SampleRate)
			if exact*time.Duration(full.SampleRate)/time.Second != time.Duration(frames) {
				t.Fatalf("test needs a limit that is a whole number of frames; %d frames at %d Hz", frames, full.SampleRate)
			}
			framePeriod := time.Second / time.Duration(full.SampleRate)

			checkLimit(t, data, tt.format, 0, false)
			checkLimit(t, data, tt.format, exact, false)
			checkLimit(t, data, tt.format, 10*exact, false)
			checkLimit(t, data, tt.format, exact-framePeriod, true)
			checkLimit(t, data, tt.format, exact/2, true)
			checkLimit(t, data, tt.format, framePeriod, true)
		})
	}
}

func TestDecodeMonoPropagatesLimit(t *testing.T) {
	data := pcm16WAV(16000, 1, make([]float32, 32000))
	_, err := DecodeMono(data, FormatWAV, 16000, DecodeLimits{MaxDuration: time.Second})
	var limitErr *DurationLimitError
	if !errors.As(err, &limitErr) {
		t.Fatalf("got %v, want *DurationLimitError", err)
	}
}

func TestMP3LimitStopsDecodingEarly(t *testing.T) {
	// The limit must bound the work done, not just the result: decoding stops a
	// frame past the budget instead of expanding the whole stream first.
	data := loadFixture(t, "tone.mp3")
	var err error
	small := allocatedBytes(func() { _, err = Decode(data, FormatMP3, DecodeLimits{MaxDuration: 20 * time.Millisecond}) })
	var limitErr *DurationLimitError
	if !errors.As(err, &limitErr) {
		t.Fatalf("got %v, want *DurationLimitError", err)
	}
	full := allocatedBytes(func() { _, err = Decode(data, FormatMP3, DecodeLimits{}) })
	if err != nil {
		t.Fatal(err)
	}
	if small >= full {
		t.Errorf("limited decode allocated %d bytes, unlimited %d: limit did not cut the work", small, full)
	}
}

// checkLimit decodes data with a MaxDuration of limit and checks that it fails
// with a *DurationLimitError exactly when wantError.
func checkLimit(t *testing.T, data []byte, format AudioFormat, limit time.Duration, wantError bool) {
	t.Helper()
	got, err := Decode(data, format, DecodeLimits{MaxDuration: limit})
	var limitErr *DurationLimitError
	switch {
	case wantError && !errors.As(err, &limitErr):
		t.Errorf("limit %s: got err=%v, want *DurationLimitError", limit, err)
	case wantError && got != nil:
		t.Errorf("limit %s: got a result alongside the limit error", limit)
	case wantError && limitErr.Limit != limit:
		t.Errorf("limit %s: error reports limit %s", limit, limitErr.Limit)
	case !wantError && err != nil:
		t.Errorf("limit %s: unexpected error: %v", limit, err)
	case !wantError && got == nil:
		t.Errorf("limit %s: nil result", limit)
	}
}

func TestDurationLimitErrorMessage(t *testing.T) {
	err := error(&DurationLimitError{Limit: 90 * time.Second})
	if !strings.Contains(err.Error(), "1m30s") {
		t.Errorf("message %q does not state the limit", err)
	}
}

func TestDecodedMono(t *testing.T) {
	tests := []struct {
		name     string
		in       Decoded
		rate     int
		want     []float32
		wantFrac float32 // tolerance
	}{
		{"mono passthrough", Decoded{Samples: []float32{0.1, -0.2, 0.3}, SampleRate: 16000, Channels: 1}, 16000, []float32{0.1, -0.2, 0.3}, 0},
		{"stereo average", Decoded{Samples: []float32{1, 0, 0.5, -0.5, 0.25, 0.75}, SampleRate: 16000, Channels: 2}, 16000, []float32{0.5, 0, 0.5}, 0},
		{"opposite channels cancel", Decoded{Samples: []float32{0.5, -0.5, 0.5, -0.5}, SampleRate: 16000, Channels: 2}, 16000, []float32{0, 0}, 0},
		{"three channels", Decoded{Samples: []float32{0.3, 0.6, 0.9, 0, 0, 0.3}, SampleRate: 16000, Channels: 3}, 16000, []float32{0.6, 0.1}, 1e-7},
		{"partial trailing frame dropped", Decoded{Samples: []float32{1, 1, 0.5, 0.5, 9}, SampleRate: 16000, Channels: 2}, 16000, []float32{1, 0.5}, 0},
		{"empty", Decoded{SampleRate: 44100, Channels: 2}, 16000, []float32{}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.in.Mono(tt.rate)
			if err != nil {
				t.Fatal(err)
			}
			assertSliceNear(t, got, tt.want, tt.wantFrac)
		})
	}

	t.Run("constant is preserved when resampling", func(t *testing.T) {
		in := make([]float32, 4800)
		for i := range in {
			in[i] = 0.25
		}
		d := Decoded{Samples: in, SampleRate: 48000, Channels: 1}
		got, err := d.Mono(16000)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != expectedResampleLen(4800, 48000, 16000) {
			t.Fatalf("got %d samples", len(got))
		}
		for _, v := range got {
			assertNear(t, v, 0.25, 1e-5)
		}
	})
}

func TestDecodedMonoInvalid(t *testing.T) {
	tests := []struct {
		name string
		in   Decoded
		rate int
	}{
		{"zero channels", Decoded{Samples: []float32{1}, SampleRate: 16000}, 16000},
		{"negative channels", Decoded{Samples: []float32{1}, SampleRate: 16000, Channels: -1}, 16000},
		{"zero sample rate", Decoded{Samples: []float32{1}, Channels: 1}, 16000},
		{"zero value", Decoded{}, 16000},
		{"target rate outside resampler range", Decoded{Samples: make([]float32, 100), SampleRate: 16000, Channels: 1}, 100},
		{"source rate outside resampler range", Decoded{Samples: make([]float32, 100), SampleRate: 100, Channels: 1}, 16000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got, err := tt.in.Mono(tt.rate); err == nil {
				t.Fatalf("expected an error, got %d samples", len(got))
			}
		})
	}
}

// dominantFrequency returns the strongest frequency in x by a coarse DFT scan
// in 5 Hz steps up to Nyquist.
func dominantFrequency(x []float32, rate int) float64 {
	best, bestPower := 0.0, -1.0
	for f := 50.0; f < float64(rate)/2; f += 5 {
		var re, im float64
		for i, v := range x {
			phase := 2 * math.Pi * f * float64(i) / float64(rate)
			re += float64(v) * math.Cos(phase)
			im += float64(v) * math.Sin(phase)
		}
		if p := re*re + im*im; p > bestPower {
			best, bestPower = f, p
		}
	}
	return best
}
