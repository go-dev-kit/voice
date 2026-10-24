package audio

import (
	"fmt"
	"math"
	"testing"
)

var extremeRates = []int{4000, 6000, 8000, 11025, 22050, 44100, 48000, 96000, 192000, 384000}

// finite reports the first NaN or Inf index, or -1.
func firstNonFinite(x []float32) int {
	for i, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return i
		}
	}
	return -1
}

func TestResampleExtremeRatesSine(t *testing.T) {
	const freq = 300.0 // well below the Nyquist rate of every tested rate
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
		r := NewResampler(&ResampleConfig{Method: method, Quality: QualityHigh, FilterLength: 32})
		for _, src := range extremeRates {
			for _, dst := range extremeRates {
				if src == dst {
					continue
				}
				t.Run(fmt.Sprintf("%s/%d->%d", method, src, dst), func(t *testing.T) {
					n := src / 2 // half a second
					in := sine(freq, src, n, 0.5)
					out, err := r.Resample(in, src, dst)
					if err != nil {
						t.Fatalf("Resample: %v", err)
					}
					if want := expectedResampleLen(n, src, dst); len(out) != want {
						t.Fatalf("len = %d, want %d", len(out), want)
					}
					if i := firstNonFinite(out); i >= 0 {
						t.Fatalf("non-finite sample at %d: %v", i, out[i])
					}
					got := rmsInterior(out, 0.2)
					want := 0.5 / math.Sqrt2
					if db := gainDB(got, want); math.Abs(db) > 0.5 {
						t.Fatalf("rms gain %.3f dB (rms %.4f, want %.4f)", db, got, want)
					}
				})
			}
		}
	}
}

func TestResampleTinyInputs(t *testing.T) {
	for _, method := range []ResampleMethod{MethodLinear, MethodCubic, MethodLanczos, MethodSinc} {
		r := NewResampler(&ResampleConfig{Method: method, Quality: QualityHigh, FilterLength: 32})
		for _, n := range []int{0, 1, 2, 3, 5} {
			for _, pair := range [][2]int{{4000, 384000}, {384000, 4000}, {44100, 48000}, {48000, 8000}, {8000, 48000}} {
				t.Run(fmt.Sprintf("%s/n%d/%d->%d", method, n, pair[0], pair[1]), func(t *testing.T) {
					in := make([]float32, n)
					for i := range in {
						in[i] = 0.25
					}
					out, err := r.Resample(in, pair[0], pair[1])
					if err != nil {
						t.Fatalf("Resample: %v", err)
					}
					if want := expectedResampleLen(n, pair[0], pair[1]); len(out) != want {
						t.Fatalf("len = %d, want %d", len(out), want)
					}
					if i := firstNonFinite(out); i >= 0 {
						t.Fatalf("non-finite sample at %d", i)
					}
					// A constant input must stay constant; stale pooled data would not.
					for i, v := range out {
						if math.Abs(float64(v)-0.25) > 1e-3 {
							t.Fatalf("out[%d] = %v, want 0.25", i, v)
						}
					}
				})
			}
		}
	}
}

// TestResampleDoesNotLeakPooledData guards against output positions that are
// never written keeping the previous contents of a recycled pool buffer.
func TestResampleDoesNotLeakPooledData(t *testing.T) {
	r := NewResampler(&ResampleConfig{Method: MethodSinc, Quality: QualityHigh, FilterLength: 32})
	dirty := DefaultAudioBufferPool.Get(4096)
	for i := range dirty {
		dirty[i] = 7
	}
	DefaultAudioBufferPool.Put(dirty)
	// Silence in: every output sample must be exactly zero.
	for range 8 {
		out, err := r.Resample(make([]float32, 1000), 48000, 44100)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range out {
			if v != 0 {
				t.Fatalf("out[%d] = %v, want 0", i, v)
			}
		}
	}
}

func TestDecodeMonoExtremeRates(t *testing.T) {
	const target = 16000
	for _, rate := range extremeRates {
		for _, channels := range []int{1, 2} {
			for _, bits := range []int{8, 16, 24, 32} {
				t.Run(fmt.Sprintf("%dHz/%dch/%dbit", rate, channels, bits), func(t *testing.T) {
					n := rate / 4
					mono := sine(300, rate, n, 0.5)
					vals := make([]float64, 0, n*channels)
					for _, v := range mono {
						for range channels {
							vals = append(vals, float64(v))
						}
					}
					wav := buildWAV(wavSpec{tag: wavFormatPCM, channels: channels, rate: rate, bits: bits}, pcmBytes(vals, bits))
					out, err := DecodeMono(wav, FormatWAV, target, DecodeLimits{})
					if err != nil {
						t.Fatalf("DecodeMono: %v", err)
					}
					if want := expectedResampleLen(n, rate, target); rate != target && len(out) != want {
						t.Fatalf("len = %d, want %d", len(out), want)
					}
					if i := firstNonFinite(out); i >= 0 {
						t.Fatalf("non-finite sample at %d", i)
					}
					want := 0.5 / math.Sqrt2
					if db := gainDB(rmsInterior(out, 0.2), want); math.Abs(db) > 0.5 {
						t.Fatalf("rms gain %.3f dB", db)
					}
				})
			}
		}
	}
}

func TestDecodeMonoFloatWAVExtremeRates(t *testing.T) {
	for _, rate := range extremeRates {
		for _, bits := range []int{32, 64} {
			t.Run(fmt.Sprintf("%dHz/%dbit", rate, bits), func(t *testing.T) {
				n := rate / 4
				in := sine(300, rate, n, 0.5)
				wav := buildWAV(wavSpec{tag: wavFormatFloat, channels: 1, rate: rate, bits: bits}, floatBytes(float64s(in), bits))
				out, err := DecodeMono(wav, FormatWAV, 16000, DecodeLimits{})
				if err != nil {
					t.Fatalf("DecodeMono: %v", err)
				}
				if i := firstNonFinite(out); i >= 0 {
					t.Fatalf("non-finite sample at %d", i)
				}
				if db := gainDB(rmsInterior(out, 0.2), 0.5/math.Sqrt2); math.Abs(db) > 0.5 {
					t.Fatalf("rms gain %.3f dB", db)
				}
			})
		}
	}
}

func TestDecodeMonoShortInputs(t *testing.T) {
	for _, rate := range extremeRates {
		for _, channels := range []int{1, 2} {
			for _, frames := range []int{0, 1, 2, 3} {
				t.Run(fmt.Sprintf("%dHz/%dch/%dframes", rate, channels, frames), func(t *testing.T) {
					vals := make([]float64, frames*channels)
					for i := range vals {
						vals[i] = 0.25
					}
					wav := buildWAV(wavSpec{tag: wavFormatPCM, channels: channels, rate: rate, bits: 16}, pcmBytes(vals, 16))
					out, err := DecodeMono(wav, FormatWAV, 16000, DecodeLimits{})
					if err != nil {
						t.Fatalf("DecodeMono: %v", err)
					}
					if want := expectedResampleLen(frames, rate, 16000); rate != 16000 && len(out) != want {
						t.Fatalf("len = %d, want %d", len(out), want)
					}
					if i := firstNonFinite(out); i >= 0 {
						t.Fatalf("non-finite sample at %d", i)
					}
				})
			}
		}
	}
}

func TestDecodedMonoOddLengthInterleaved(t *testing.T) {
	// A trailing partial frame is dropped, not read out of range.
	d := &Decoded{Samples: []float32{0.5, 0.5, 0.5}, SampleRate: 16000, Channels: 2}
	out, err := d.Mono(16000)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0] != 0.5 {
		t.Fatalf("got %v, want [0.5]", out)
	}
}

func TestConvertAudioExtremeRates(t *testing.T) {
	conv, err := NewConverter(&ConverterConfig{EnableResampling: true, TargetRMS: 0.1, TempBufferSize: 8192, NormalizeFactor: 32768})
	if err != nil {
		t.Fatal(err)
	}
	rates := []int{8000, 11025, 22050, 44100, 48000, 96000, 192000}
	for _, src := range rates {
		for _, dst := range rates {
			for _, ch := range []int{1, 2} {
				t.Run(fmt.Sprintf("%d->%d/%dch", src, dst, ch), func(t *testing.T) {
					n := src / 4
					mono := sine(300, src, n, 0.5)
					vals := make([]float64, 0, n*ch)
					for _, v := range mono {
						for range ch {
							vals = append(vals, float64(v))
						}
					}
					in := &AudioConfig{Format: FormatPCM, SampleRate: src, Channels: ch, BitsPerSample: 16}
					out := &AudioConfig{Format: FormatPCM, SampleRate: dst, Channels: ch, BitsPerSample: 16}
					data, err := conv.ConvertAudio(pcmBytes(vals, 16), in, out)
					if err != nil {
						t.Fatalf("ConvertAudio: %v", err)
					}
					if want := expectedResampleLen(n, src, dst) * ch * 2; len(data) != want {
						t.Fatalf("bytes = %d, want %d", len(data), want)
					}
				})
			}
		}
	}
}

func TestConvertAudioShortInputs(t *testing.T) {
	conv, err := NewConverter(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, frames := range []int{1, 2, 3} {
		for _, ch := range []int{1, 2} {
			t.Run(fmt.Sprintf("%dframes/%dch", frames, ch), func(t *testing.T) {
				vals := make([]float64, frames*ch)
				for i := range vals {
					vals[i] = 0.25
				}
				in := &AudioConfig{Format: FormatPCM, SampleRate: 48000, Channels: ch, BitsPerSample: 16}
				out := &AudioConfig{Format: FormatWAV, SampleRate: 8000, Channels: 1, BitsPerSample: 16}
				data, err := conv.ConvertAudio(pcmBytes(vals, 16), in, out)
				if err != nil {
					t.Fatalf("ConvertAudio: %v", err)
				}
				if len(data) < 44 {
					t.Fatalf("output %d bytes, shorter than a WAV header", len(data))
				}
			})
		}
	}
}

func TestCreateWAVHeaderLargeValues(t *testing.T) {
	c, _ := NewConverter(nil)
	cfg := &AudioConfig{Format: FormatWAV, SampleRate: 192000, Channels: 8, BitsPerSample: 32}
	h := c.createWAVHeader(1<<20, cfg)
	byteRate := uint32(h[28]) | uint32(h[29])<<8 | uint32(h[30])<<16 | uint32(h[31])<<24
	if want := uint32(192000 * 8 * 4); byteRate != want {
		t.Fatalf("byte rate = %d, want %d", byteRate, want)
	}
	blockAlign := uint16(h[32]) | uint16(h[33])<<8
	if blockAlign != 32 {
		t.Fatalf("block align = %d, want 32", blockAlign)
	}
}

func TestDecodeWAVMalformedHeaders(t *testing.T) {
	payload := pcmBytes([]float64{0.1, 0.2, 0.3, 0.4}, 16)
	cases := map[string][]byte{
		"zero rate":     buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: 0, bits: 16}, payload),
		"zero channels": buildWAV(wavSpec{tag: wavFormatPCM, channels: 0, rate: 16000, bits: 16}, payload),
		"zero bits":     buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 0}, payload),
		"odd bits":      buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 12}, payload),
		"huge rate":     buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: math.MaxInt32, bits: 16}, payload),
		"huge channels": buildWAV(wavSpec{tag: wavFormatPCM, channels: 65535, rate: 16000, bits: 16}, payload),
		"huge data":     buildWAVClaiming(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 16}, payload, 0xFFFFFFF0),
		"unknown tag":   buildWAV(wavSpec{tag: 0x55, channels: 1, rate: 16000, bits: 16}, payload),
		"header only":   buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 16}, nil)[:20],
		"riff only":     []byte("RIFF\x00\x00\x00\x00WAVE"),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("panic: %v", r)
					}
				}()
				d, err := Decode(data, FormatWAV, DecodeLimits{})
				if err == nil && name != "huge data" && name != "huge rate" && name != "huge channels" {
					t.Fatalf("expected error, got %+v", d)
				}
				if d != nil {
					if _, err := d.Mono(16000); err != nil {
						return
					}
				}
			}()
		})
	}
}

func FuzzDecodeWAV(f *testing.F) {
	payload := pcmBytes([]float64{0.1, -0.2, 0.3, -0.4}, 16)
	f.Add(buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 16}, payload))
	f.Add(buildWAV(wavSpec{tag: wavFormatPCM, channels: 2, rate: 8000, bits: 24}, pcmBytes([]float64{0.1, -0.2, 0.3, -0.4}, 24)))
	f.Add(buildWAV(wavSpec{tag: wavFormatFloat, channels: 1, rate: 384000, bits: 32}, floatBytes([]float64{0.1, -0.2}, 32)))
	f.Add(buildWAV(wavSpec{extensible: true, subTag: wavFormatPCM, channels: 2, rate: 44100, bits: 16}, payload))
	f.Add(buildWAVClaiming(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 16}, payload, 0xFFFFFFFF))
	f.Add(buildWAVClaiming(wavSpec{tag: wavFormatPCM, channels: 1, rate: 16000, bits: 16}, payload, 0xFFFFFFF0))
	f.Fuzz(func(t *testing.T, data []byte) {
		d, err := Decode(data, FormatWAV, DecodeLimits{MaxDuration: 600e9})
		if err != nil {
			return
		}
		if d.Channels <= 0 || d.SampleRate <= 0 {
			t.Fatalf("accepted invalid format %d Hz, %d ch", d.SampleRate, d.Channels)
		}
		if len(d.Samples) > len(data) {
			t.Fatalf("%d samples from %d bytes", len(d.Samples), len(data))
		}
		// Mono rejects rates outside the resampler range with an error, never a panic.
		_, _ = d.Mono(16000)
	})
}

func TestConvertAudioWAVBitDepthsAmplitude(t *testing.T) {
	conv, err := NewConverter(&ConverterConfig{EnableResampling: true, TargetRMS: 0.1, TempBufferSize: 8192, NormalizeFactor: 32768})
	if err != nil {
		t.Fatal(err)
	}
	for _, bits := range []int{8, 16, 24, 32} {
		for _, pair := range [][2]int{{8000, 192000}, {192000, 8000}, {44100, 48000}} {
			t.Run(fmt.Sprintf("%dbit/%d->%d", bits, pair[0], pair[1]), func(t *testing.T) {
				src, dst := pair[0], pair[1]
				n := src / 2
				in := sine(300, src, n, 0.5)
				wav := buildWAV(wavSpec{tag: wavFormatPCM, channels: 1, rate: src, bits: bits}, pcmBytes(float64s(in), bits))
				cfgIn := &AudioConfig{Format: FormatWAV, SampleRate: src, Channels: 1, BitsPerSample: bits}
				cfgOut := &AudioConfig{Format: FormatWAV, SampleRate: dst, Channels: 1, BitsPerSample: bits}
				data, err := conv.ConvertAudio(wav, cfgIn, cfgOut)
				if err != nil {
					t.Fatalf("ConvertAudio: %v", err)
				}
				d, err := Decode(data, FormatWAV, DecodeLimits{})
				if err != nil {
					t.Fatalf("decode output: %v", err)
				}
				if d.SampleRate != dst || len(d.Samples) != expectedResampleLen(n, src, dst) {
					t.Fatalf("got %d Hz, %d samples", d.SampleRate, len(d.Samples))
				}
				if db := gainDB(rmsInterior(d.Samples, 0.2), 0.5/math.Sqrt2); math.Abs(db) > 0.6 {
					t.Fatalf("rms gain %.3f dB", db)
				}
			})
		}
	}
}
