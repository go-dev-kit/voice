package audio

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"testing"
)

func newConverterWith(t *testing.T, resampling, normalization bool) *Converter {
	t.Helper()
	cfg := DefaultConverterConfig()
	cfg.EnableResampling = resampling
	cfg.EnableNormalization = normalization
	conv, err := NewConverter(cfg)
	if err != nil {
		t.Fatalf("NewConverter: %v", err)
	}
	return conv
}

func pcmConfig(rate, channels int) *AudioConfig {
	return &AudioConfig{Format: FormatPCM, SampleRate: rate, Channels: channels, BitsPerSample: 16}
}

func wavConfig(rate, channels int) *AudioConfig {
	return &AudioConfig{Format: FormatWAV, SampleRate: rate, Channels: channels, BitsPerSample: 16}
}

// int16s reads little-endian 16-bit PCM.
func int16s(t *testing.T, b []byte) []int16 {
	t.Helper()
	if len(b)%2 != 0 {
		t.Fatalf("odd PCM length %d", len(b))
	}
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

// constantStereoWAV is seconds of 16-bit stereo with a constant value per channel.
func constantStereoWAV(rate int, seconds float64, left, right int16) []byte {
	frames := int(float64(rate) * seconds)
	payload := make([]byte, 0, frames*4)
	for range frames {
		payload = binary.LittleEndian.AppendUint16(payload, uint16(left))
		payload = binary.LittleEndian.AppendUint16(payload, uint16(right))
	}
	return buildWAV(wavSpec{tag: 1, channels: 2, rate: rate, bits: 16}, payload)
}

func TestConvertAudioStereoToMonoCancels(t *testing.T) {
	// The review's case: L=+16384, R=-16384 averages to silence. The corrupted
	// resampler produced +-6540 here.
	tests := []struct {
		name       string
		resampling bool
		srcRate    int
		dstRate    int
	}{
		{"resampling 44.1k to 16k", true, 44100, 16000},
		{"resampling 48k to 16k", true, 48000, 16000},
		{"resampling 16k to 48k", true, 16000, 48000},
		{"resampling enabled, same rate", true, 16000, 16000},
		{"resampling disabled, same rate", false, 16000, 16000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv := newConverterWith(t, tt.resampling, false)
			in := constantStereoWAV(tt.srcRate, 1, 16384, -16384)
			out, err := conv.ConvertAudio(in, wavConfig(tt.srcRate, 2), pcmConfig(tt.dstRate, 1))
			if err != nil {
				t.Fatal(err)
			}
			got := int16s(t, out)
			if want := expectedResampleLen(tt.srcRate, tt.srcRate, tt.dstRate); len(got) != want {
				t.Fatalf("got %d mono samples, want %d", len(got), want)
			}
			for i, v := range got {
				if v < -2 || v > 2 {
					t.Fatalf("sample %d = %d, want ~0", i, v)
				}
			}
		})
	}
}

func TestConvertAudioStereoToMonoAverage(t *testing.T) {
	conv := newConverterWith(t, true, false)
	in := constantStereoWAV(44100, 1, 16384, 8192)
	out, err := conv.ConvertAudio(in, wavConfig(44100, 2), pcmConfig(16000, 1))
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range int16s(t, out) {
		if v < 12286 || v > 12290 {
			t.Fatalf("sample %d = %d, want 12288", i, v)
		}
	}
}

func TestConvertAudioStereoResamplesPerChannel(t *testing.T) {
	// Stereo stays stereo: each channel is resampled on its own, so +/- constants
	// stay +/- constants instead of bleeding into each other.
	conv := newConverterWith(t, true, false)
	in := constantStereoWAV(44100, 1, 16384, -16384)
	out, err := conv.ConvertAudio(in, wavConfig(44100, 2), pcmConfig(16000, 2))
	if err != nil {
		t.Fatal(err)
	}
	got := int16s(t, out)
	if len(got)%2 != 0 || len(got)/2 != expectedResampleLen(44100, 44100, 16000) {
		t.Fatalf("got %d samples, want %d frames", len(got), expectedResampleLen(44100, 44100, 16000))
	}
	for i := 0; i < len(got); i += 2 {
		if got[i] < 16382 || got[i] > 16386 || got[i+1] < -16386 || got[i+1] > -16382 {
			t.Fatalf("frame %d = (%d, %d), want (16384, -16384)", i/2, got[i], got[i+1])
		}
	}
}

func TestConvertAudioKeepsChannelsSeparate(t *testing.T) {
	// 1 kHz on the left, 3 kHz on the right, 48 kHz to 16 kHz.
	const rate = 48000
	left, right := sine(1000, rate, rate, 0.4), sine(3000, rate, rate, 0.4)
	inter := make([]float32, 2*rate)
	for i := range left {
		inter[2*i], inter[2*i+1] = left[i], right[i]
	}
	conv := newConverterWith(t, true, false)
	out, err := conv.ConvertAudio(pcm16WAV(rate, 2, inter), wavConfig(rate, 2), pcmConfig(16000, 2))
	if err != nil {
		t.Fatal(err)
	}
	vals := int16s(t, out)
	l, r := make([]float32, len(vals)/2), make([]float32, len(vals)/2)
	for i := range l {
		l[i], r[i] = float32(vals[2*i])/32768, float32(vals[2*i+1])/32768
	}
	if f := dominantFrequency(l[2000:4000], 16000); math.Abs(f-1000) > 25 {
		t.Errorf("left channel dominant frequency %.0f Hz, want 1000", f)
	}
	if f := dominantFrequency(r[2000:4000], 16000); math.Abs(f-3000) > 25 {
		t.Errorf("right channel dominant frequency %.0f Hz, want 3000", f)
	}
	for name, ch := range map[string][]float32{"left": l, "right": r} {
		if db := gainDB(rmsInterior(ch, 0.1), 0.4/math.Sqrt2); math.Abs(db) > 1 {
			t.Errorf("%s channel level changed by %.2f dB", name, db)
		}
	}
}

func TestConvertAudioMonoToStereo(t *testing.T) {
	tests := []struct {
		name       string
		resampling bool
		dstRate    int
	}{
		{"with resampling", true, 44100},
		{"resampling enabled, same rate", true, 16000},
		{"resampling disabled, same rate", false, 16000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			conv := newConverterWith(t, tt.resampling, false)
			tone := sine(440, 16000, 16000, 0.5)
			out, err := conv.ConvertAudio(pcm16WAV(16000, 1, tone), wavConfig(16000, 1), pcmConfig(tt.dstRate, 2))
			if err != nil {
				t.Fatal(err)
			}
			got := int16s(t, out)
			frames := len(got) / 2
			if want := expectedResampleLen(len(tone), 16000, tt.dstRate); frames != want || len(got)%2 != 0 {
				t.Fatalf("got %d samples, want %d stereo frames", len(got), want)
			}
			left := make([]float32, frames)
			for i := range frames {
				if got[2*i] != got[2*i+1] {
					t.Fatalf("frame %d: left %d != right %d", i, got[2*i], got[2*i+1])
				}
				left[i] = float32(got[2*i]) / 32768
			}
			if db := gainDB(rmsInterior(left, 0.1), 0.5/math.Sqrt2); math.Abs(db) > 0.5 {
				t.Errorf("level changed by %.2f dB", db)
			}
		})
	}
}

func TestConvertAudioResamplingDisabled(t *testing.T) {
	conv := newConverterWith(t, false, false)

	t.Run("channel conversion still applies", func(t *testing.T) {
		// The review found stereo->mono returned stereo-sized output here.
		in := constantStereoWAV(16000, 1, 16384, 8192)
		out, err := conv.ConvertAudio(in, wavConfig(16000, 2), pcmConfig(16000, 1))
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 32000 {
			t.Fatalf("got %d bytes, want 32000 (1 s of 16-bit mono at 16 kHz)", len(out))
		}
		assertNear(t, float32(int16s(t, out)[100]), 12288, 2)
	})
	t.Run("rate change is refused rather than mislabeled", func(t *testing.T) {
		in := constantStereoWAV(44100, 1, 100, 100)
		_, err := conv.ConvertAudio(in, wavConfig(44100, 2), pcmConfig(16000, 1))
		if err == nil || !strings.Contains(err.Error(), "resampling is disabled") {
			t.Fatalf("got %v, want a resampling-disabled error", err)
		}
	})
}

func TestConvertAudioNormalization(t *testing.T) {
	conv := newConverterWith(t, true, true)
	quiet := sine(440, 16000, 16000, 0.05) // RMS 0.035, below the 4x gain cap
	out, err := conv.ConvertAudio(pcm16WAV(16000, 1, quiet), wavConfig(16000, 1), pcmConfig(16000, 1))
	if err != nil {
		t.Fatal(err)
	}
	vals := int16s(t, out)
	f := make([]float32, len(vals))
	for i, v := range vals {
		f[i] = float32(v) / 32768
	}
	if got := rmsInterior(f, 0); math.Abs(got-0.1) > 0.002 {
		t.Errorf("normalised RMS = %.4f, want 0.1", got)
	}
}

func TestConvertAudioOutputFormats(t *testing.T) {
	tone := sine(440, 16000, 8000, 0.5)
	in := pcm16WAV(16000, 1, tone)
	conv := newConverterWith(t, true, false)

	t.Run("wav output is a valid file", func(t *testing.T) {
		out, err := conv.ConvertAudio(in, wavConfig(16000, 1), wavConfig(8000, 1))
		if err != nil {
			t.Fatal(err)
		}
		if string(out[:4]) != "RIFF" || string(out[8:12]) != "WAVE" {
			t.Fatalf("bad header % x", out[:12])
		}
		if got := binary.LittleEndian.Uint32(out[4:8]); int(got) != len(out)-8 {
			t.Errorf("RIFF size %d, want %d", got, len(out)-8)
		}
		decoded, err := Decode(out, FormatWAV, DecodeLimits{})
		if err != nil {
			t.Fatal(err)
		}
		if decoded.SampleRate != 8000 || decoded.Channels != 1 || len(decoded.Samples) != expectedResampleLen(len(tone), 16000, 8000) {
			t.Errorf("round trip: %d Hz, %d ch, %d samples", decoded.SampleRate, decoded.Channels, len(decoded.Samples))
		}
	})

	for _, bits := range []int{8, 16, 24, 32} {
		t.Run(fmt.Sprintf("pcm %d-bit round trip", bits), func(t *testing.T) {
			cfg := &AudioConfig{Format: FormatPCM, SampleRate: 16000, Channels: 1, BitsPerSample: bits}
			out, err := conv.ConvertAudio(in, wavConfig(16000, 1), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != len(tone)*bits/8 {
				t.Fatalf("got %d bytes, want %d", len(out), len(tone)*bits/8)
			}
			back, err := conv.ConvertToFloat32(out, cfg)
			if err != nil {
				t.Fatal(err)
			}
			tol := float32(2.0 / 128)
			if bits > 8 {
				tol = 1e-3
			}
			assertSliceNear(t, back, tone, tol)
		})
	}

	t.Run("compressed outputs are refused with a clear error", func(t *testing.T) {
		for _, format := range []AudioFormat{FormatFLAC, FormatMP3, FormatOGG} {
			cfg := &AudioConfig{Format: format, SampleRate: 16000, Channels: 1, Quality: 80, Bitrate: 128}
			if _, err := conv.ConvertAudio(in, wavConfig(16000, 1), cfg); err == nil || !strings.Contains(err.Error(), "not supported") {
				t.Errorf("%s: got %v, want a not-supported error", format, err)
			}
		}
	})
}

func TestConvertAudioCompressedInputs(t *testing.T) {
	conv := newConverterWith(t, true, false)
	tests := []struct {
		file    string
		in      *AudioConfig
		outRate int
	}{
		{"tone.flac", &AudioConfig{Format: FormatFLAC, SampleRate: 16000, Channels: 1}, 8000},
		{"tone.ogg", &AudioConfig{Format: FormatOGG, SampleRate: 16000, Channels: 2}, 16000},
		{"tone.mp3", &AudioConfig{Format: FormatMP3, SampleRate: 16000, Channels: 2}, 16000},
		{"tone.mp3", &AudioConfig{Format: FormatMP3, SampleRate: 16000, Channels: 1}, 16000},
	}
	for _, tt := range tests {
		t.Run(fmt.Sprintf("%s/%dch", tt.file, tt.in.Channels), func(t *testing.T) {
			out, err := conv.ConvertAudio(loadFixture(t, tt.file), tt.in, pcmConfig(tt.outRate, 1))
			if err != nil {
				t.Fatal(err)
			}
			f := make([]float32, len(out)/2)
			for i, v := range int16s(t, out) {
				f[i] = float32(v) / 32768
			}
			assertPlausibleAudio(t, f, 1500)
		})
	}
}

// TestConvertToFloat32MP3Mono is the previously broken case: DefaultMP3Config
// asks for one channel, go-mp3 always emits stereo, and the check used to fail.
func TestConvertToFloat32MP3Mono(t *testing.T) {
	data := loadFixture(t, "tone.mp3")
	conv := newConverterWith(t, true, true)

	mono, err := conv.ConvertToFloat32(data, DefaultMP3Config())
	if err != nil {
		t.Fatalf("DefaultMP3Config (1 channel): %v", err)
	}
	stereo, err := conv.ConvertToFloat32(data, &AudioConfig{Format: FormatMP3, SampleRate: 16000, Channels: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(mono)*2 != len(stereo) {
		t.Fatalf("mono has %d samples, stereo %d", len(mono), len(stereo))
	}
	for i, v := range mono {
		want := (stereo[2*i] + stereo[2*i+1]) / 2
		if math.Abs(float64(v-want)) > 1e-6 {
			t.Fatalf("mono[%d] = %v, want channel average %v", i, v, want)
		}
	}
	assertPlausibleAudio(t, mono, 3000)

	// A rate that does not match the stream is still an error.
	cfg := DefaultMP3Config()
	cfg.SampleRate = 44100
	if _, err := conv.ConvertToFloat32(data, cfg); err == nil {
		t.Error("expected a sample-rate mismatch error")
	}
}

func TestConvertAudioErrors(t *testing.T) {
	conv := newConverterWith(t, true, false)
	good := constantStereoWAV(16000, 0.1, 1000, 1000)
	tests := []struct {
		name string
		data []byte
		in   *AudioConfig
		out  *AudioConfig
	}{
		{"empty input", nil, wavConfig(16000, 2), pcmConfig(16000, 1)},
		{"nil input config", good, nil, pcmConfig(16000, 1)},
		{"nil output config", good, wavConfig(16000, 2), nil},
		{"invalid input config", good, &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 0, BitsPerSample: 16}, pcmConfig(16000, 1)},
		{"invalid output config", good, wavConfig(16000, 2), &AudioConfig{Format: FormatPCM, SampleRate: 16000, Channels: 1, BitsPerSample: 12}},
		{"garbage wav", []byte("this is not a wav file, not even close"), wavConfig(16000, 2), pcmConfig(16000, 1)},
		{"wav header disagrees with config", good, wavConfig(16000, 1), pcmConfig(16000, 1)},
		{"wav rate disagrees with config", good, wavConfig(44100, 2), pcmConfig(16000, 1)},
		{"m4a input", []byte("x"), &AudioConfig{Format: FormatM4A, SampleRate: 16000, Channels: 1, Quality: 80, Bitrate: 128}, pcmConfig(16000, 1)},
		{"six channels to mono", make([]byte, 120), pcmConfig(16000, 6), pcmConfig(16000, 1)},
		{"mono to six channels", make([]byte, 120), pcmConfig(16000, 1), pcmConfig(16000, 6)},
		{"misaligned raw pcm", make([]byte, 7), pcmConfig(16000, 1), pcmConfig(16000, 1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// "m4a input" fails Validate (unsupported), still an error as required.
			if out, err := conv.ConvertAudio(tt.data, tt.in, tt.out); err == nil {
				t.Fatalf("expected an error, got %d bytes", len(out))
			}
		})
	}
}

func TestConvertWAVTruncatedStillErrors(t *testing.T) {
	// The header claims 100 samples; only 10 follow. Bounding the allocation by
	// the input must not turn this into a silent short read.
	spec := wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}
	data := buildWAVClaiming(spec, make([]byte, 20), 200)
	_, err := newTestConverter(t).ConvertToFloat32(data, wavConfig(16000, 1))
	if err == nil {
		t.Fatal("expected an error for a truncated data chunk")
	}
	if !errors.Is(err, io.ErrUnexpectedEOF) && !strings.Contains(err.Error(), "EOF") {
		t.Errorf("error %q should report the truncation", err)
	}
}

func TestConvertWAVValidation(t *testing.T) {
	conv := newTestConverter(t)
	pcm := make([]byte, 8)
	tests := []struct {
		name string
		data []byte
		cfg  *AudioConfig
	}{
		{"float WAV is not accepted by the converter", buildWAV(wavSpec{tag: 3, channels: 1, rate: 16000, bits: 32}, pcm), &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 1, BitsPerSample: 32}},
		{"channel mismatch", buildWAV(wavSpec{tag: 1, channels: 2, rate: 16000, bits: 16}, pcm), wavConfig(16000, 1)},
		{"rate mismatch", buildWAV(wavSpec{tag: 1, channels: 1, rate: 8000, bits: 16}, pcm), wavConfig(16000, 1)},
		{"bit depth mismatch", buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 24}, pcm), wavConfig(16000, 1)},
		{"not a wav", []byte("RIFX nope nope nope nope nope"), wavConfig(16000, 1)},
		{"data not aligned with sample size", buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 24}, make([]byte, 4)), &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 1, BitsPerSample: 24}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := conv.ConvertToFloat32(tt.data, tt.cfg); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

func TestConvertWAVLargerThanTempBuffer(t *testing.T) {
	// More samples than TempBufferSize so decodeWAVPCM loops over chunks.
	vals := float64s(sine(300, 16000, 20000, 0.5))
	data := buildWAV(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}, pcmBytes(vals, 16))
	conv := newTestConverter(t) // TempBufferSize 8192
	got, err := conv.ConvertToFloat32(data, wavConfig(16000, 1))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(vals) {
		t.Fatalf("got %d samples, want %d", len(got), len(vals))
	}
	for i, v := range vals {
		if math.Abs(float64(got[i])-v) > 1.0/32768 {
			t.Fatalf("sample %d = %v, want %v", i, got[i], v)
		}
	}
}

func TestResampleInterleavedChannels(t *testing.T) {
	r := newTestResampler(MethodSinc, 32)
	const frames = 4800
	const channels = 3
	in := make([]float32, frames*channels)
	levels := [channels]float32{0.5, -0.25, 0.1}
	for i := range frames {
		for c := range channels {
			in[i*channels+c] = levels[c]
		}
	}
	out, err := resampleInterleaved(r, in, channels, 48000, 16000)
	if err != nil {
		t.Fatal(err)
	}
	if want := expectedResampleLen(frames, 48000, 16000) * channels; len(out) != want {
		t.Fatalf("got %d samples, want %d", len(out), want)
	}
	for i, v := range out {
		assertNear(t, v, levels[i%channels], 1e-5)
	}

	mono, err := resampleInterleaved(r, in[:frames], 1, 48000, 16000)
	if err != nil || len(mono) != expectedResampleLen(frames, 48000, 16000) {
		t.Fatalf("mono: %d samples, %v", len(mono), err)
	}
	if _, err := resampleInterleaved(r, in, channels, 100, 16000); err == nil {
		t.Error("expected the resampler's rate error to propagate")
	}
}

func TestConverterConfigValidation(t *testing.T) {
	valid := func() *ConverterConfig { return DefaultConverterConfig() }
	tests := []struct {
		name   string
		mutate func(*ConverterConfig)
	}{
		{"zero target RMS", func(c *ConverterConfig) { c.TargetRMS = 0 }},
		{"negative target RMS", func(c *ConverterConfig) { c.TargetRMS = -0.1 }},
		{"target RMS above one", func(c *ConverterConfig) { c.TargetRMS = 1.5 }},
		{"zero buffer", func(c *ConverterConfig) { c.TempBufferSize = 0 }},
		{"buffer above 1MB", func(c *ConverterConfig) { c.TempBufferSize = 1 << 21 }},
		{"zero normalize factor", func(c *ConverterConfig) { c.NormalizeFactor = 0 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := valid()
			tt.mutate(cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatal("Validate: expected an error")
			}
			if _, err := NewConverter(cfg); err == nil {
				t.Fatal("NewConverter: expected an error")
			}
		})
	}
	if err := valid().Validate(); err != nil {
		t.Errorf("default config invalid: %v", err)
	}
	if conv, err := NewConverter(nil); err != nil || conv == nil {
		t.Errorf("NewConverter(nil) = %v, %v; want the default converter", conv, err)
	}
}

func TestConvertFloat32Errors(t *testing.T) {
	conv := newTestConverter(t)
	if _, err := conv.ConvertToFloat32(nil, pcmConfig(16000, 1)); err == nil {
		t.Error("ConvertToFloat32: empty data should error")
	}
	if _, err := conv.ConvertToFloat32([]byte{0, 0}, nil); err == nil {
		t.Error("ConvertToFloat32: nil config should error")
	}
	if _, err := conv.ConvertToFloat32([]byte{0, 0}, &AudioConfig{Format: FormatPCM}); err == nil {
		t.Error("ConvertToFloat32: invalid config should error")
	}
	if _, err := conv.ConvertFromFloat32(nil, pcmConfig(16000, 1)); err == nil {
		t.Error("ConvertFromFloat32: empty samples should error")
	}
	if _, err := conv.ConvertFromFloat32([]float32{0}, nil); err == nil {
		t.Error("ConvertFromFloat32: nil config should error")
	}
	if _, err := conv.ConvertFromFloat32([]float32{0}, &AudioConfig{Format: FormatPCM}); err == nil {
		t.Error("ConvertFromFloat32: invalid config should error")
	}
	if _, err := conv.ConvertFromFloat32([]float32{0}, &AudioConfig{Format: FormatMP3, SampleRate: 16000, Channels: 1, Quality: 80}); err == nil {
		t.Error("ConvertFromFloat32: MP3 encoding should be unsupported")
	}
}

func TestEncodeClampsOutOfRangeSamples(t *testing.T) {
	conv := newTestConverter(t)
	got, err := conv.ConvertFromFloat32([]float32{2, -2, 1, -1, 0.5}, pcmConfig(16000, 1))
	if err != nil {
		t.Fatal(err)
	}
	assertSliceNear(t, float32sFromInt16(int16s(t, got)), []float32{32767, -32768, 32767, -32768, 16384}, 1)
}

func float32sFromInt16(v []int16) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(x)
	}
	return out
}

func TestParseWAVFile(t *testing.T) {
	conv := newTestConverter(t)

	t.Run("mono", func(t *testing.T) {
		data := pcm16WAV(8000, 1, []float32{0.5, -0.5, 0.25})
		got, rate, err := conv.ParseWAVFile(data)
		if err != nil {
			t.Fatal(err)
		}
		if rate != 8000 {
			t.Errorf("rate = %d", rate)
		}
		assertSliceNear(t, got, []float32{0.5, -0.5, 0.25}, 1e-4)
	})
	t.Run("rejects", func(t *testing.T) {
		for name, data := range map[string][]byte{
			"not a wav":       []byte("garbage garbage garbage garbage"),
			"three channels":  buildWAV(wavSpec{tag: 1, channels: 3, rate: 16000, bits: 16}, make([]byte, 12)),
			"float WAV":       buildWAV(wavSpec{tag: 3, channels: 1, rate: 16000, bits: 32}, make([]byte, 8)),
			"truncated claim": buildWAVClaiming(wavSpec{tag: 1, channels: 1, rate: 16000, bits: 16}, make([]byte, 4), 1<<28),
		} {
			if _, _, err := conv.ParseWAVFile(data); err == nil {
				t.Errorf("%s: expected an error", name)
			}
		}
	})
}

func TestSupportedConversions(t *testing.T) {
	conv := newTestConverter(t)
	if !conv.ValidateConversion(FormatWAV, FormatPCM) || !conv.ValidateConversion(FormatPCM, FormatWAV) {
		t.Error("WAV and PCM should convert both ways")
	}
	if conv.ValidateConversion(FormatWAV, FormatMP3) || conv.ValidateConversion(FormatFLAC, FormatWAV) {
		t.Error("unsupported conversions reported as supported")
	}
	if len(conv.GetSupportedConversions()) == 0 {
		t.Error("no supported conversions listed")
	}
}

func TestPCMDecodeErrors(t *testing.T) {
	conv := newTestConverter(t)
	if _, err := conv.ConvertToFloat32(make([]byte, 5), pcmConfig(16000, 1)); err == nil {
		t.Error("misaligned PCM should error")
	}
	var bad bytes.Buffer
	bad.Write(make([]byte, 8))
	if _, err := conv.decodePCM(bad.Bytes(), &AudioConfig{Format: FormatPCM, SampleRate: 16000, Channels: 1, BitsPerSample: 40}); err == nil {
		t.Error("unsupported bit depth should error")
	}
}
