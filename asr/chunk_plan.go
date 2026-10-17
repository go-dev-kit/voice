package asr

import (
	"fmt"
	"slices"
	"time"
)

// MaxDecodeDuration is the hard upper bound on the audio one offline decode may
// receive on the segmented path. Sherpa offline Whisper keeps only the first
// 30 s of its input (it logs a warning and returns no error), and NeMo
// Parakeet exports fail from about 30 s, so a decoder never sees more than this.
const MaxDecodeDuration = 25 * time.Second

// SpeechSpan is a half-open sample range [Start, End) of one audio buffer.
type SpeechSpan struct {
	Start int
	End   int
}

// ChunkOptions controls how speech spans become decoder inputs.
type ChunkOptions struct {
	// MaxChunk is the hard cap on one decode. It must be positive and at most
	// MaxDecodeDuration.
	MaxChunk time.Duration
	// Overlap is the audio two neighboring chunks of one over-long span share,
	// so a word at the cut is heard whole by at least one of them. The cut
	// itself sits in the middle of the overlap.
	Overlap time.Duration
	// Padding widens every speech span on both sides; VAD trims word onsets.
	Padding time.Duration
	// MergeGap joins neighboring spans whose silence gap is at most this long,
	// as long as the joined span still fits in MaxChunk. More context per
	// decode helps accuracy and saves decoder calls.
	MergeGap time.Duration
	// CutSearch is how far before the cap the quietest cut point is searched.
	CutSearch time.Duration
}

// DefaultChunkOptions returns the options the segmented path uses unless the
// caller overrides them.
func DefaultChunkOptions() ChunkOptions {
	return ChunkOptions{
		MaxChunk:  MaxDecodeDuration,
		Overlap:   2 * time.Second,
		Padding:   200 * time.Millisecond,
		MergeGap:  time.Second,
		CutSearch: 10 * time.Second,
	}
}

// Validate reports options that could exceed the decode cap or stop a long
// span from making progress.
func (o ChunkOptions) Validate() error {
	switch {
	case o.MaxChunk <= 0:
		return fmt.Errorf("max chunk must be positive, got %s", o.MaxChunk)
	case o.MaxChunk > MaxDecodeDuration:
		return fmt.Errorf("max chunk %s exceeds the %s decode cap", o.MaxChunk, MaxDecodeDuration)
	case o.Overlap < 0, o.Padding < 0, o.MergeGap < 0, o.CutSearch < 0:
		return fmt.Errorf("overlap, padding, merge gap and cut search must not be negative")
	case o.Overlap+o.CutSearch >= o.MaxChunk/2:
		return fmt.Errorf("overlap (%s) plus cut search (%s) must stay below half of max chunk (%s)", o.Overlap, o.CutSearch, o.MaxChunk)
	}
	return nil
}

// Chunk is one decoder input planned from the speech spans.
type Chunk struct {
	// Start and End bound the samples sent to the decoder.
	Start int
	End   int
	// KeepFrom and KeepTo bound the part of the chunk this chunk owns. They
	// differ from Start and End only around a cut inside an over-long span,
	// where neighboring chunks overlap and the cut point divides ownership.
	KeepFrom int
	KeepTo   int
	// Continues is true when this chunk continues the previous chunk's span
	// through an overlapping cut.
	Continues bool
}

// PlanChunks turns speech spans into decoder inputs no longer than
// opts.MaxChunk. Spans are clamped to the audio, padded, and merged; any span
// still longer than the cap is cut at the quietest point near the cap, with
// opts.Overlap of shared audio around each cut. Nothing is dropped: every
// sample of every padded span belongs to exactly one chunk's keep range.
func PlanChunks(spans []SpeechSpan, samples []float32, sampleRate int, opts ChunkOptions) ([]Chunk, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sampleRate must be positive, got %d", sampleRate)
	}
	if err := opts.Validate(); err != nil {
		return nil, err
	}

	toSamples := func(d time.Duration) int { return int(d * time.Duration(sampleRate) / time.Second) }
	maxLen := toSamples(opts.MaxChunk)
	half := toSamples(opts.Overlap) / 2
	search := toSamples(opts.CutSearch)
	frame := max(sampleRate/100, 1)

	merged := normalizeSpans(spans, len(samples), toSamples(opts.Padding), toSamples(opts.MergeGap), maxLen)
	chunks := make([]Chunk, 0, len(merged))
	for _, span := range merged {
		pos, keepFrom := span.Start, span.Start
		for span.End-pos > maxLen {
			// The chunk ends half an overlap after the cut, so the cut can be
			// at most maxLen-half into the chunk; it must also leave the next
			// chunk starting after this one.
			hi := pos + maxLen - half
			lo := max(hi-search, pos+half+1)
			cut := quietestCut(samples, lo, hi, frame)
			chunks = append(chunks, Chunk{Start: pos, End: cut + half, KeepFrom: keepFrom, KeepTo: cut, Continues: pos != span.Start})
			keepFrom = cut
			pos = cut - half
		}
		chunks = append(chunks, Chunk{Start: pos, End: span.End, KeepFrom: keepFrom, KeepTo: span.End, Continues: pos != span.Start})
	}
	return chunks, nil
}

// normalizeSpans clamps, pads, sorts and merges spans. Spans that overlap after
// padding always merge (the result may exceed maxLen and is cut later); spans
// separated by at most mergeGap merge only while the result fits in maxLen.
func normalizeSpans(spans []SpeechSpan, n, pad, mergeGap, maxLen int) []SpeechSpan {
	out := make([]SpeechSpan, 0, len(spans))
	for _, s := range spans {
		s.Start = max(s.Start-pad, 0)
		s.End = min(s.End+pad, n)
		if s.Start < s.End {
			out = append(out, s)
		}
	}
	slices.SortFunc(out, func(a, b SpeechSpan) int { return a.Start - b.Start })

	merged := out[:0]
	for _, s := range out {
		if len(merged) == 0 {
			merged = append(merged, s)
			continue
		}
		last := &merged[len(merged)-1]
		switch {
		case s.Start <= last.End:
			last.End = max(last.End, s.End)
		case s.Start-last.End <= mergeGap && s.End-last.Start <= maxLen:
			last.End = s.End
		default:
			merged = append(merged, s)
		}
	}
	return merged
}

// pauseFrames is the width, in 10 ms frames, of the window quietestCut
// minimizes: real pauses outlast the short closures inside words.
const pauseFrames = 20

// quietestCut returns the sample position in [lo, hi] at the center of the
// quietest 200 ms window, measured in 10 ms frames. Ties go to the later
// window so chunks stay long.
func quietestCut(samples []float32, lo, hi, frame int) int {
	n := (hi - lo) / frame
	if n < 1 {
		return hi
	}
	energy := make([]float32, n)
	for i := range energy {
		for _, v := range samples[lo+i*frame : lo+(i+1)*frame] {
			energy[i] += v * v
		}
	}
	width := min(pauseFrames, n)
	var sum float32
	for _, e := range energy[:width] {
		sum += e
	}
	best, bestSum := width/2, sum
	for i := width; i < n; i++ {
		sum += energy[i] - energy[i-width]
		if sum <= bestSum {
			best, bestSum = i-width+1+width/2, sum
		}
	}
	return min(lo+best*frame+frame/2, hi)
}
