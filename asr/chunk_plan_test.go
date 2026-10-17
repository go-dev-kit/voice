package asr

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
	"time"
)

const testRate = 16000

func secs(s float64) int { return int(s * testRate) }

// toneWithGaps returns n samples of a loud tone with silent gaps.
func toneWithGaps(n int, gaps ...SpeechSpan) []float32 {
	samples := make([]float32, n)
	for i := range samples {
		samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/testRate))
	}
	for _, g := range gaps {
		for i := max(g.Start, 0); i < min(g.End, n); i++ {
			samples[i] = 0
		}
	}
	return samples
}

func TestChunkOptionsValidate(t *testing.T) {
	ok := DefaultChunkOptions()
	if err := ok.Validate(); err != nil {
		t.Fatalf("defaults invalid: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*ChunkOptions)
		want   string
	}{
		{"zero max chunk", func(o *ChunkOptions) { o.MaxChunk = 0 }, "must be positive"},
		{"over the cap", func(o *ChunkOptions) { o.MaxChunk = MaxDecodeDuration + time.Millisecond }, "exceeds"},
		{"negative overlap", func(o *ChunkOptions) { o.Overlap = -1 }, "must not be negative"},
		{"negative padding", func(o *ChunkOptions) { o.Padding = -1 }, "must not be negative"},
		{"negative merge gap", func(o *ChunkOptions) { o.MergeGap = -1 }, "must not be negative"},
		{"negative cut search", func(o *ChunkOptions) { o.CutSearch = -1 }, "must not be negative"},
		{"no progress", func(o *ChunkOptions) {
			o.MaxChunk = 10 * time.Second
			o.Overlap = 2 * time.Second
			o.CutSearch = 3 * time.Second
		}, "half of max chunk"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := DefaultChunkOptions()
			tt.mutate(&o)
			err := o.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() = %v, want error containing %q", err, tt.want)
			}
		})
	}
}

func TestPlanChunksRejectsBadInput(t *testing.T) {
	if _, err := PlanChunks(nil, nil, 0, DefaultChunkOptions()); err == nil {
		t.Fatal("expected error for zero sample rate")
	}
	if _, err := PlanChunks(nil, nil, testRate, ChunkOptions{}); err == nil {
		t.Fatal("expected error for invalid options")
	}
}

func TestPlanChunksShortSpansStayWhole(t *testing.T) {
	samples := make([]float32, secs(60))
	opts := DefaultChunkOptions()
	opts.Padding = 0
	opts.MergeGap = 0
	spans := []SpeechSpan{{secs(30), secs(40)}, {secs(1), secs(5)}, {secs(5), secs(4)}, {-10, 0}}
	chunks, err := PlanChunks(spans, samples, testRate, opts)
	if err != nil {
		t.Fatal(err)
	}
	want := []Chunk{
		{Start: secs(1), End: secs(5), KeepFrom: secs(1), KeepTo: secs(5)},
		{Start: secs(30), End: secs(40), KeepFrom: secs(30), KeepTo: secs(40)},
	}
	if len(chunks) != len(want) {
		t.Fatalf("got %d chunks, want %d: %+v", len(chunks), len(want), chunks)
	}
	for i := range want {
		if chunks[i] != want[i] {
			t.Errorf("chunk %d = %+v, want %+v", i, chunks[i], want[i])
		}
	}
}

func TestPlanChunksPadsClampsAndMerges(t *testing.T) {
	samples := make([]float32, secs(30))
	opts := DefaultChunkOptions() // 200 ms padding, 1 s merge gap
	spans := []SpeechSpan{
		{0, secs(2)},                 // padding clamps at 0
		{secs(2.5), secs(4)},         // 0.1 s gap after padding: overlaps, merges
		{secs(5), secs(6)},           // 0.6 s gap after padding: merges
		{secs(10), secs(11)},         // 3.6 s gap: separate
		{secs(29.9), secs(30) + 100}, // clamps at the end
	}
	chunks, err := PlanChunks(spans, samples, testRate, opts)
	if err != nil {
		t.Fatal(err)
	}
	want := []SpeechSpan{{0, secs(6.2)}, {secs(9.8), secs(11.2)}, {secs(29.7), secs(30)}}
	if len(chunks) != len(want) {
		t.Fatalf("got %+v, want spans %+v", chunks, want)
	}
	for i, w := range want {
		if chunks[i].Start != w.Start || chunks[i].End != w.End {
			t.Errorf("chunk %d = [%d,%d), want [%d,%d)", i, chunks[i].Start, chunks[i].End, w.Start, w.End)
		}
	}
}

func TestPlanChunksMergeStopsAtCap(t *testing.T) {
	samples := make([]float32, secs(60))
	opts := DefaultChunkOptions()
	opts.Padding = 0
	// Three 10 s spans 0.5 s apart: the first two fit in 25 s, the third does not.
	spans := []SpeechSpan{{0, secs(10)}, {secs(10.5), secs(20.5)}, {secs(21), secs(31)}}
	chunks, err := PlanChunks(spans, samples, testRate, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 2 || chunks[0].End != secs(20.5) || chunks[1].Start != secs(21) {
		t.Fatalf("unexpected merge: %+v", chunks)
	}
}

func TestPlanChunksSplitsAtTheQuietestPoint(t *testing.T) {
	// 60 s of tone with one silent second at 20-21 s, inside the cut search
	// window of the first chunk.
	samples := toneWithGaps(secs(60), SpeechSpan{secs(20), secs(21)})
	chunks, err := PlanChunks([]SpeechSpan{{0, len(samples)}}, samples, testRate, DefaultChunkOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected a split, got %+v", chunks)
	}
	cut := chunks[0].KeepTo
	if cut < secs(20) || cut > secs(21) {
		t.Fatalf("first cut at %.2fs, want inside the 20-21 s pause", float64(cut)/testRate)
	}
	if !chunks[1].Continues || chunks[0].Continues {
		t.Fatalf("continuation flags wrong: %+v", chunks[:2])
	}
	half := secs(1) // half of the 2 s default overlap
	if chunks[0].End != cut+half || chunks[1].Start != cut-half || chunks[1].KeepFrom != cut {
		t.Fatalf("overlap around the cut wrong: %+v", chunks[:2])
	}
}

func TestPlanChunksWithoutOverlap(t *testing.T) {
	samples := toneWithGaps(secs(80))
	opts := DefaultChunkOptions()
	opts.Overlap = 0
	chunks, err := PlanChunks([]SpeechSpan{{0, len(samples)}}, samples, testRate, opts)
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i < len(chunks); i++ {
		if chunks[i].Start != chunks[i-1].End {
			t.Fatalf("chunks %d and %d overlap without overlap configured: %+v", i-1, i, chunks)
		}
	}
	checkChunkInvariants(t, []SpeechSpan{{0, len(samples)}}, len(samples), opts, chunks)
}

func TestQuietestCutShortWindow(t *testing.T) {
	if got := quietestCut(make([]float32, 100), 10, 50, 160); got != 50 {
		t.Fatalf("window shorter than a frame: got %d, want hi", got)
	}
	// A window narrower than the 200 ms pause width still finds the quiet frame.
	samples := toneWithGaps(secs(1), SpeechSpan{secs(0.5), secs(0.52)})
	got := quietestCut(samples, secs(0.4), secs(0.6), testRate/100)
	if got < secs(0.45) || got > secs(0.57) {
		t.Fatalf("cut at %d, want near the quiet frame at %d", got, secs(0.5))
	}
}

// checkChunkInvariants asserts the planner's guarantees: no chunk exceeds the
// cap, keep ranges lie inside their chunks, and inside each merged span the
// keep ranges tile it without gaps or overlaps.
func checkChunkInvariants(t *testing.T, spans []SpeechSpan, n int, opts ChunkOptions, chunks []Chunk) {
	t.Helper()
	maxLen := int(opts.MaxChunk * testRate / time.Second)
	covered := make([]int, n)
	for i, c := range chunks {
		if c.End-c.Start > maxLen {
			t.Fatalf("chunk %d is %d samples, cap %d", i, c.End-c.Start, maxLen)
		}
		if c.Start < 0 || c.End > n || c.Start >= c.End {
			t.Fatalf("chunk %d out of range: %+v", i, c)
		}
		if c.KeepFrom < c.Start || c.KeepTo > c.End || c.KeepFrom >= c.KeepTo {
			t.Fatalf("chunk %d keep range outside chunk: %+v", i, c)
		}
		// Separate spans never touch (touching spans merge), so a chunk
		// continues exactly when its keep range starts where the last ended.
		if c.Continues != (i > 0 && chunks[i-1].KeepTo == c.KeepFrom) {
			t.Fatalf("chunk %d continuation flag inconsistent: %+v after %+v", i, c, chunks[i-1])
		}
		for s := c.KeepFrom; s < c.KeepTo; s++ {
			covered[s]++
		}
	}
	for _, s := range spans {
		for i := max(s.Start, 0); i < min(s.End, n); i++ {
			if covered[i] != 1 {
				t.Fatalf("speech sample %d covered %d times", i, covered[i])
			}
		}
	}
	for i, c := range covered {
		if c > 1 {
			t.Fatalf("sample %d owned by %d chunks", i, c)
		}
	}
}

func TestPlanChunksRandomInvariants(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for range 200 {
		n := secs(1 + rng.Float64()*240)
		samples := make([]float32, n)
		for i := range samples {
			samples[i] = float32(rng.NormFloat64() * 0.1)
		}
		spans := make([]SpeechSpan, rng.IntN(12))
		for i := range spans {
			start := rng.IntN(n)
			spans[i] = SpeechSpan{start, start + rng.IntN(secs(90))}
		}
		opts := DefaultChunkOptions()
		opts.MaxChunk = time.Duration(10+rng.IntN(16)) * time.Second
		opts.Overlap = time.Duration(rng.IntN(3)) * time.Second
		opts.CutSearch = time.Duration(rng.IntN(int(opts.MaxChunk/2-opts.Overlap)/int(time.Second))) * time.Second
		chunks, err := PlanChunks(spans, samples, testRate, opts)
		if err != nil {
			t.Fatalf("opts %+v: %v", opts, err)
		}
		checkChunkInvariants(t, spans, n, opts, chunks)
	}
}

func FuzzPlanChunks(f *testing.F) {
	f.Add(60.0, 10.0, 50.0, uint8(25), uint8(2), uint8(10))
	f.Add(3.0, 0.0, 3.0, uint8(1), uint8(0), uint8(0))
	f.Fuzz(func(t *testing.T, total, start, end float64, maxChunk, overlap, search uint8) {
		if math.IsNaN(total) || total <= 0 || total > 600 || math.IsNaN(start) || math.IsNaN(end) {
			t.Skip()
		}
		opts := DefaultChunkOptions()
		opts.MaxChunk = time.Duration(maxChunk%26) * time.Second
		opts.Overlap = time.Duration(overlap%4) * time.Second
		opts.CutSearch = time.Duration(search%11) * time.Second
		if opts.Validate() != nil {
			t.Skip()
		}
		n := secs(total)
		samples := toneWithGaps(n)
		spans := []SpeechSpan{{secs(math.Max(-1e6, math.Min(start, 1e6))), secs(math.Max(-1e6, math.Min(end, 1e6)))}}
		chunks, err := PlanChunks(spans, samples, testRate, opts)
		if err != nil {
			t.Fatal(err)
		}
		checkChunkInvariants(t, spans, n, opts, chunks)
	})
}

func BenchmarkPlanChunks(b *testing.B) {
	// One hour of audio with a speech span every 4 s plus one hour-long span.
	n := secs(3600)
	samples := toneWithGaps(n)
	spans := make([]SpeechSpan, 0, 900)
	for s := 0; s+secs(3) < n; s += secs(4) {
		spans = append(spans, SpeechSpan{s, s + secs(3)})
	}
	b.Run("vad_spans_1h", func(b *testing.B) {
		for b.Loop() {
			if _, err := PlanChunks(spans, samples, testRate, DefaultChunkOptions()); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("one_span_1h", func(b *testing.B) {
		whole := []SpeechSpan{{0, n}}
		for b.Loop() {
			if _, err := PlanChunks(whole, samples, testRate, DefaultChunkOptions()); err != nil {
				b.Fatal(err)
			}
		}
	})
}
