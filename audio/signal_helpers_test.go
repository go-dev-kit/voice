package audio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// sine returns n samples of an amplitude-amp sine at freq Hz sampled at rate Hz.
func sine(freq float64, rate, n int, amp float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(amp * math.Sin(2*math.Pi*freq*float64(i)/float64(rate)))
	}
	return out
}

// float64s widens samples for the PCM encoders.
func float64s(x []float32) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v)
	}
	return out
}

// rmsInterior is the RMS of x without the first and last edge fraction, where
// resampler edge handling (clamped history) distorts the signal.
func rmsInterior(x []float32, edgeFraction float64) float64 {
	skip := int(float64(len(x)) * edgeFraction)
	body := x[skip : len(x)-skip]
	var sum float64
	for _, v := range body {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum / float64(len(body)))
}

// gainDB is the level of got relative to want, in dB.
func gainDB(got, want float64) float64 {
	return 20 * math.Log10(got/want)
}

// expectedResampleLen mirrors Resampler.Resample's output length rule.
func expectedResampleLen(n, srcRate, dstRate int) int {
	return int(float64(n) * (float64(dstRate) / float64(srcRate)))
}

// loadFixture reads a file from testdata.
func loadFixture(t testing.TB, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture %s: %v", name, err)
	}
	return data
}

// wavSpec describes the fmt chunk of a synthetic WAV file.
type wavSpec struct {
	tag        uint16 // format tag; ignored when extensible
	channels   int
	rate       int
	bits       int
	extensible bool   // write a WAVE_FORMAT_EXTENSIBLE header
	subTag     uint16 // extensible sub-format tag (1 PCM, 3 float)
}

func (s wavSpec) blockAlign() int { return s.channels * s.bits / 8 }

// buildWAV wraps payload in a RIFF/WAVE container.
func buildWAV(s wavSpec, payload []byte) []byte {
	return buildWAVClaiming(s, payload, uint32(len(payload)))
}

// buildWAVClaiming is buildWAV with a data chunk size that may disagree with
// the payload actually present, as in a truncated or forged file.
func buildWAVClaiming(s wavSpec, payload []byte, claimed uint32) []byte {
	var fmtChunk []byte
	le := binary.LittleEndian
	tag := s.tag
	if s.extensible {
		tag = wavFormatExtensible
	}
	fmtChunk = le.AppendUint16(fmtChunk, tag)
	fmtChunk = le.AppendUint16(fmtChunk, uint16(s.channels))
	fmtChunk = le.AppendUint32(fmtChunk, uint32(s.rate))
	fmtChunk = le.AppendUint32(fmtChunk, uint32(s.rate*s.blockAlign()))
	fmtChunk = le.AppendUint16(fmtChunk, uint16(s.blockAlign()))
	fmtChunk = le.AppendUint16(fmtChunk, uint16(s.bits))
	if s.extensible {
		fmtChunk = le.AppendUint16(fmtChunk, 22)             // cbSize
		fmtChunk = le.AppendUint16(fmtChunk, uint16(s.bits)) // valid bits
		fmtChunk = le.AppendUint32(fmtChunk, 0x3)            // channel mask: front L+R
		fmtChunk = le.AppendUint16(fmtChunk, s.subTag)
		fmtChunk = append(fmtChunk, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71)
	}

	pad := len(payload) % 2
	var out []byte
	out = append(out, "RIFF"...)
	out = le.AppendUint32(out, uint32(4+8+len(fmtChunk)+8+len(payload)+pad))
	out = append(out, "WAVE"...)
	out = append(out, "fmt "...)
	out = le.AppendUint32(out, uint32(len(fmtChunk)))
	out = append(out, fmtChunk...)
	out = append(out, "data"...)
	out = le.AppendUint32(out, claimed)
	out = append(out, payload...)
	if pad == 1 {
		out = append(out, 0)
	}
	return out
}

// pcmBytes encodes values in [-1, 1) as little-endian integer PCM.
func pcmBytes(vals []float64, bits int) []byte {
	var out []byte
	le := binary.LittleEndian
	for _, v := range vals {
		switch bits {
		case 8:
			out = append(out, byte(max(0, min(255, int(math.Round(v*128))+128))))
		case 16:
			out = le.AppendUint16(out, uint16(int16(max(-32768, min(32767, math.Round(v*32768))))))
		case 24:
			n := int32(max(-8388608, min(8388607, math.Round(v*8388608))))
			out = append(out, byte(n), byte(n>>8), byte(n>>16))
		case 32:
			n := int32(max(math.MinInt32, min(math.MaxInt32, math.Round(v*2147483648))))
			out = le.AppendUint32(out, uint32(n))
		default:
			panic("unsupported bits")
		}
	}
	return out
}

// floatBytes encodes values as little-endian IEEE float.
func floatBytes(vals []float64, bits int) []byte {
	var out []byte
	for _, v := range vals {
		switch bits {
		case 32:
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(float32(v)))
		case 64:
			out = binary.LittleEndian.AppendUint64(out, math.Float64bits(v))
		default:
			panic("unsupported bits")
		}
	}
	return out
}

// pcm16WAV builds a 16-bit PCM WAV from float samples in [-1, 1).
func pcm16WAV(rate, channels int, samples []float32) []byte {
	return buildWAV(wavSpec{tag: wavFormatPCM, channels: channels, rate: rate, bits: 16}, pcmBytes(float64s(samples), 16))
}

// allocatedBytes reports how many bytes f allocated. Tests using it must not
// run in parallel with other allocating tests.
func allocatedBytes(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}
