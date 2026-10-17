package asr

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode"

	"go.glpx.pro/gdk/voice/types"
)

// SegmentedTranscriber makes any offline transcriber safe for long audio. It
// finds the speech with a SpeechSegmenter, plans decoder inputs no longer than
// ChunkOptions.MaxChunk with PlanChunks, decodes them in order, and stitches
// the results. It never truncates: every planned sample reaches a decoder, and
// a decoder that fails fails the whole call.
type SegmentedTranscriber struct {
	inner     types.Transcriber
	segmenter SpeechSegmenter
	opts      ChunkOptions
}

// NewSegmentedTranscriber wraps inner. A nil segmenter treats the whole buffer
// as one speech span, so long input is still cut at quiet points under the
// cap. The transcriber owns inner and segmenter and closes both.
func NewSegmentedTranscriber(inner types.Transcriber, segmenter SpeechSegmenter, opts ChunkOptions) (*SegmentedTranscriber, error) {
	if inner == nil {
		return nil, fmt.Errorf("inner transcriber cannot be nil")
	}
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	return &SegmentedTranscriber{inner: inner, segmenter: segmenter, opts: opts}, nil
}

// Transcribe returns the stitched transcription of audio. Segments holds one
// entry per decoded chunk, with times relative to the start of audio; Tokens
// holds the timed tokens each chunk owns. Audio without speech returns an empty
// transcription, not an error.
func (t *SegmentedTranscriber) Transcribe(ctx context.Context, audio []float32, sampleRate int) (*types.Transcription, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}
	if len(audio) == 0 {
		return nil, fmt.Errorf("audio cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sampleRate must be positive, got %d", sampleRate)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	spans := []SpeechSpan{{Start: 0, End: len(audio)}}
	if t.segmenter != nil {
		var err error
		if spans, err = t.segmenter.SpeechSpans(ctx, audio, sampleRate); err != nil {
			return nil, fmt.Errorf("segment speech: %w", err)
		}
	}
	chunks, err := PlanChunks(spans, audio, sampleRate, t.opts)
	if err != nil {
		return nil, err
	}

	results := make([]*types.Transcription, len(chunks))
	for i, c := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		result, err := t.inner.Transcribe(ctx, audio[c.Start:c.End], sampleRate)
		if err != nil {
			return nil, fmt.Errorf("decode chunk %d/%d (%s-%s): %w", i+1, len(chunks),
				samplesToDuration(c.Start, sampleRate), samplesToDuration(c.End, sampleRate), err)
		}
		results[i] = result
	}
	return stitchChunks(chunks, results, len(audio), sampleRate), nil
}

// Close closes the inner transcriber and the segmenter.
func (t *SegmentedTranscriber) Close() error {
	var errs []error
	if t.segmenter != nil {
		errs = append(errs, t.segmenter.Close())
	}
	errs = append(errs, t.inner.Close())
	return errors.Join(errs...)
}

// stitchChunks joins per-chunk results into one transcription. Where chunks
// overlap around a cut, a chunk keeps only the tokens that start inside its
// keep range when its tokens are timed and spell out its text; otherwise the
// words the previous chunk already ended with are dropped from its text.
func stitchChunks(chunks []Chunk, results []*types.Transcription, sampleCount, sampleRate int) *types.Transcription {
	out := &types.Transcription{
		Timestamp: time.Now(),
		EndTime:   samplesToDuration(sampleCount, sampleRate),
		Segments:  make([]types.TranscriptSegment, 0, len(chunks)),
	}
	var text, prevText string
	for i, c := range chunks {
		r := results[i]
		if r == nil {
			r = &types.Transcription{}
		}
		if out.Language == "" {
			out.Language = r.Language
		}
		if out.Emotion == "" {
			out.Emotion = r.Emotion
		}
		if out.Event == "" {
			out.Event = r.Event
		}

		offset := samplesToDuration(c.Start, sampleRate)
		keepFrom := samplesToDuration(c.KeepFrom, sampleRate)
		keepTo := samplesToDuration(c.KeepTo, sampleRate)
		splitAfter := i+1 < len(chunks) && chunks[i+1].Continues

		var segText string
		var owned []types.Token
		if tokensSpellText(r) {
			for _, word := range groupWords(r.Tokens) {
				at := offset + word[0].StartTime
				if (c.Continues && at < keepFrom) || (splitAfter && at >= keepTo) {
					continue
				}
				for _, tok := range word {
					tok.StartTime += offset
					owned = append(owned, tok)
				}
			}
			segText = tokensText(owned)
		} else {
			segText = strings.TrimSpace(r.Text)
			if c.Continues {
				segText = dropRepeatedPrefix(prevText, segText)
			}
		}
		prevText = strings.TrimSpace(r.Text)

		out.Tokens = append(out.Tokens, owned...)
		out.Segments = append(out.Segments, types.TranscriptSegment{
			Text:      segText,
			StartTime: keepFrom,
			EndTime:   keepTo,
		})
		text = joinText(text, segText)
	}
	out.Text = text
	return out
}

// groupWords splits subword tokens into words: a word starts at a token that
// begins with a space or SentencePiece marker, and every character of a script
// written without spaces is its own word. Overlap ownership is decided per
// word, so a word cut in two at a chunk edge is kept whole from the chunk that
// heard all of it.
func groupWords(tokens []types.Token) [][]types.Token {
	var words [][]types.Token
	for i, tok := range tokens {
		if i == 0 || startsWord(tok.Text) || endsUnspaced(tokens[i-1].Text) {
			words = append(words, []types.Token{tok})
			continue
		}
		words[len(words)-1] = append(words[len(words)-1], tok)
	}
	return words
}

func startsWord(text string) bool {
	if text == "" {
		return false
	}
	first := []rune(text)[0]
	return first == ' ' || first == '▁' || unspacedScript(first)
}

func endsUnspaced(text string) bool {
	runes := []rune(text)
	return len(runes) > 0 && unspacedScript(runes[len(runes)-1])
}

// tokensSpellText reports whether every token is timed and the tokens
// concatenate to the recognizer's text, so timing can decide ownership.
func tokensSpellText(r *types.Transcription) bool {
	if len(r.Tokens) == 0 {
		return false
	}
	for _, tok := range r.Tokens {
		if !tok.HasTiming {
			return false
		}
	}
	return collapseSpace(tokensText(r.Tokens)) == collapseSpace(r.Text)
}

// tokensText detokenizes model-native tokens: SentencePiece word markers
// become spaces.
func tokensText(tokens []types.Token) string {
	var b strings.Builder
	for _, tok := range tokens {
		b.WriteString(tok.Text)
	}
	return collapseSpace(strings.ReplaceAll(b.String(), "▁", " "))
}

func collapseSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// maxRepeatedWords bounds how many words two overlapping chunks can share.
const maxRepeatedWords = 12

// dropRepeatedPrefix removes from next the longest run of words that both ends
// prev and starts next, compared case- and punctuation-insensitively.
func dropRepeatedPrefix(prev, next string) string {
	prevWords, nextWords := strings.Fields(prev), strings.Fields(next)
	limit := min(len(prevWords), len(nextWords), maxRepeatedWords)
	for k := limit; k > 0; k-- {
		match := true
		for j := range k {
			if normalizeWord(prevWords[len(prevWords)-k+j]) != normalizeWord(nextWords[j]) {
				match = false
				break
			}
		}
		if match {
			return strings.Join(nextWords[k:], " ")
		}
	}
	return next
}

func normalizeWord(w string) string {
	return strings.ToLower(strings.TrimFunc(w, func(r rune) bool { return unicode.IsPunct(r) || unicode.IsSymbol(r) }))
}

// joinText joins two pieces of text with a space, except between scripts that
// do not separate words with spaces.
func joinText(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	}
	last := []rune(a)[len([]rune(a))-1]
	first := []rune(b)[0]
	if unspacedScript(last) || unspacedScript(first) {
		return a + b
	}
	return a + " " + b
}

func unspacedScript(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Thai)
}

func samplesToDuration(samples, sampleRate int) time.Duration {
	return time.Duration(samples) * time.Second / time.Duration(sampleRate)
}
