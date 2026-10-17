package asr

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"go.glpx.pro/gdk/voice/types"
)

// recordingTranscriber records every decode and answers from a script.
type recordingTranscriber struct {
	mu      sync.Mutex
	inputs  []int // samples per call
	respond func(call int, audio []float32) (*types.Transcription, error)
	closed  int
	err     error
}

func (r *recordingTranscriber) Transcribe(ctx context.Context, audio []float32, sampleRate int) (*types.Transcription, error) {
	r.mu.Lock()
	call := len(r.inputs)
	r.inputs = append(r.inputs, len(audio))
	r.mu.Unlock()
	if r.respond != nil {
		return r.respond(call, audio)
	}
	return &types.Transcription{Text: "x"}, nil
}

func (r *recordingTranscriber) Close() error {
	r.closed++
	return r.err
}

type staticSegmenter struct {
	spans  []SpeechSpan
	err    error
	closed int
}

func (s *staticSegmenter) SpeechSpans(context.Context, []float32, int) ([]SpeechSpan, error) {
	return s.spans, s.err
}

func (s *staticSegmenter) Close() error {
	s.closed++
	return nil
}

func TestNewSegmentedTranscriberValidates(t *testing.T) {
	if _, err := NewSegmentedTranscriber(nil, nil, DefaultChunkOptions()); err == nil {
		t.Fatal("expected error for nil inner transcriber")
	}
	if _, err := NewSegmentedTranscriber(&recordingTranscriber{}, nil, ChunkOptions{}); err == nil {
		t.Fatal("expected error for invalid options")
	}
}

func TestSegmentedTranscriberRejectsBadInput(t *testing.T) {
	st, err := NewSegmentedTranscriber(&recordingTranscriber{}, nil, DefaultChunkOptions())
	if err != nil {
		t.Fatal(err)
	}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := st.Transcribe(nil, []float32{1}, testRate); err == nil {
		t.Fatal("expected error for nil context")
	}
	if _, err := st.Transcribe(context.Background(), nil, testRate); err == nil {
		t.Fatal("expected error for empty audio")
	}
	if _, err := st.Transcribe(context.Background(), []float32{1}, 0); err == nil {
		t.Fatal("expected error for zero sample rate")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := st.Transcribe(ctx, []float32{1}, testRate); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled context: got %v", err)
	}
}

func TestSegmentedTranscriberNeverExceedsTheCap(t *testing.T) {
	for _, minutes := range []float64{0.1, 1, 5, 20} {
		inner := &recordingTranscriber{}
		st, err := NewSegmentedTranscriber(inner, nil, DefaultChunkOptions())
		if err != nil {
			t.Fatal(err)
		}
		audio := toneWithGaps(secs(minutes * 60))
		result, err := st.Transcribe(context.Background(), audio, testRate)
		if err != nil {
			t.Fatal(err)
		}
		limit := secs(MaxDecodeDuration.Seconds())
		total := 0
		for i, n := range inner.inputs {
			if n > limit {
				t.Fatalf("%.1f min: decode %d got %d samples, cap %d", minutes, i, n, limit)
			}
			total += n
		}
		if total < len(audio) {
			t.Fatalf("%.1f min: decoders saw %d of %d samples", minutes, total, len(audio))
		}
		if len(result.Segments) != len(inner.inputs) {
			t.Fatalf("%d segments for %d decodes", len(result.Segments), len(inner.inputs))
		}
		if result.EndTime != samplesToDuration(len(audio), testRate) {
			t.Fatalf("EndTime %s, want audio length", result.EndTime)
		}
	}
}

func TestSegmentedTranscriberUsesSegmenterSpans(t *testing.T) {
	inner := &recordingTranscriber{respond: func(call int, _ []float32) (*types.Transcription, error) {
		return &types.Transcription{Text: []string{"eins", "zwei"}[call], Language: "de"}, nil
	}}
	seg := &staticSegmenter{spans: []SpeechSpan{{secs(1), secs(3)}, {secs(10), secs(12)}}}
	st, err := NewSegmentedTranscriber(inner, seg, DefaultChunkOptions())
	if err != nil {
		t.Fatal(err)
	}
	result, err := st.Transcribe(context.Background(), make([]float32, secs(20)), testRate)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "eins zwei" || result.Language != "de" {
		t.Fatalf("got %q (%s)", result.Text, result.Language)
	}
	if got := result.Segments[1]; got.StartTime != 9800*time.Millisecond || got.EndTime != 12200*time.Millisecond {
		t.Fatalf("second segment times %+v", got)
	}
	if err := st.Close(); err != nil {
		t.Fatal(err)
	}
	if inner.closed != 1 || seg.closed != 1 {
		t.Fatalf("close counts inner=%d segmenter=%d", inner.closed, seg.closed)
	}
}

func TestSegmentedTranscriberSilenceIsEmpty(t *testing.T) {
	inner := &recordingTranscriber{}
	st, _ := NewSegmentedTranscriber(inner, &staticSegmenter{}, DefaultChunkOptions())
	result, err := st.Transcribe(context.Background(), make([]float32, secs(5)), testRate)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "" || len(inner.inputs) != 0 {
		t.Fatalf("silence produced %q after %d decodes", result.Text, len(inner.inputs))
	}
}

func TestSegmentedTranscriberPropagatesErrors(t *testing.T) {
	st, _ := NewSegmentedTranscriber(&recordingTranscriber{}, &staticSegmenter{err: errors.New("vad broke")}, DefaultChunkOptions())
	if _, err := st.Transcribe(context.Background(), make([]float32, 10), testRate); err == nil || !strings.Contains(err.Error(), "vad broke") {
		t.Fatalf("segmenter error: got %v", err)
	}

	decodeErr := errors.New("decoder broke")
	inner := &recordingTranscriber{respond: func(call int, _ []float32) (*types.Transcription, error) {
		if call == 1 {
			return nil, decodeErr
		}
		return &types.Transcription{Text: "ok"}, nil
	}}
	st, _ = NewSegmentedTranscriber(inner, nil, DefaultChunkOptions())
	_, err := st.Transcribe(context.Background(), toneWithGaps(secs(60)), testRate)
	if !errors.Is(err, decodeErr) || !strings.Contains(err.Error(), "chunk 2/") {
		t.Fatalf("decoder error: got %v", err)
	}

	inner = &recordingTranscriber{err: errors.New("close inner")}
	seg := &staticSegmenter{}
	st, _ = NewSegmentedTranscriber(inner, seg, DefaultChunkOptions())
	if err := st.Close(); err == nil || !strings.Contains(err.Error(), "close inner") || seg.closed != 1 {
		t.Fatalf("close error: got %v (segmenter closed %d)", err, seg.closed)
	}
}

func TestSegmentedTranscriberStopsBetweenChunksOnCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	inner := &recordingTranscriber{respond: func(int, []float32) (*types.Transcription, error) {
		cancel()
		return &types.Transcription{Text: "x"}, nil
	}}
	st, _ := NewSegmentedTranscriber(inner, nil, DefaultChunkOptions())
	if _, err := st.Transcribe(ctx, toneWithGaps(secs(120)), testRate); !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if len(inner.inputs) != 1 {
		t.Fatalf("decoded %d chunks after cancel, want 1", len(inner.inputs))
	}
}

// timed builds timed tokens at the given offsets (seconds, chunk-relative).
func timed(pairs ...any) []types.Token {
	tokens := make([]types.Token, 0, len(pairs)/2)
	for i := 0; i < len(pairs); i += 2 {
		tokens = append(tokens, types.Token{
			Text:      pairs[i].(string),
			HasTiming: true,
			StartTime: time.Duration(pairs[i+1].(float64) * float64(time.Second)),
		})
	}
	return tokens
}

func TestStitchChunksTimedOwnershipKeepsWordsWhole(t *testing.T) {
	// Span 0-30 s cut at 20 s with a 2 s overlap: chunk A is 0-21 s, chunk B 19-30 s.
	chunks := []Chunk{
		{Start: 0, End: secs(21), KeepFrom: 0, KeepTo: secs(20)},
		{Start: secs(19), End: secs(30), KeepFrom: secs(20), KeepTo: secs(30), Continues: true},
	}
	a := &types.Transcription{Text: "Die Ziegen wurden domestiziert", Tokens: timed(
		" Die", 17.0, " Zie", 18.0, "gen", 18.3, " wur", 19.8, "den", 20.3, " dom", 20.8)}
	a.Text = "Die Ziegen wurden dom"
	b := &types.Transcription{Text: "Ziegen wurden domestiziert", Tokens: timed(
		" Zie", -1.0+1.0, "gen", 0.3, " wur", 0.8, "den", 1.3, " dom", 1.8, "est", 2.0, "iziert", 2.2)}
	got := stitchChunks(chunks, []*types.Transcription{a, b}, secs(30), testRate)
	if got.Text != "Die Ziegen wurden domestiziert" {
		t.Fatalf("Text = %q", got.Text)
	}
	if len(got.Segments) != 2 || got.Segments[0].Text != "Die Ziegen wurden" || got.Segments[1].Text != "domestiziert" {
		t.Fatalf("Segments = %+v", got.Segments)
	}
	if got.Segments[0].EndTime != 20*time.Second || got.Segments[1].StartTime != 20*time.Second {
		t.Fatalf("segment boundary not at the cut: %+v", got.Segments)
	}
	last := got.Tokens[len(got.Tokens)-1]
	if last.Text != "iziert" || last.StartTime != 21200*time.Millisecond {
		t.Fatalf("token timing not shifted to absolute time: %+v", last)
	}
}

func TestStitchChunksUntimedFallbackDropsRepeatedWords(t *testing.T) {
	chunks := []Chunk{
		{Start: 0, End: secs(21), KeepFrom: 0, KeepTo: secs(20)},
		{Start: secs(19), End: secs(30), KeepFrom: secs(20), KeepTo: secs(30), Continues: true},
		{Start: secs(40), End: secs(45), KeepFrom: secs(40), KeepTo: secs(45)},
	}
	results := []*types.Transcription{
		{Text: "Leider fällt das Studium schwer,"},
		{Text: "Studium schwer, da man nichts weiß."},
		{Text: "Studium schwer."}, // a new span is never deduplicated
	}
	got := stitchChunks(chunks, results, secs(45), testRate)
	want := "Leider fällt das Studium schwer, da man nichts weiß. Studium schwer."
	if got.Text != want {
		t.Fatalf("Text = %q, want %q", got.Text, want)
	}
}

func TestStitchChunksMetadataAndNilResults(t *testing.T) {
	chunks := []Chunk{{Start: 0, End: 10, KeepTo: 10}, {Start: 20, End: 30, KeepFrom: 20, KeepTo: 30}}
	got := stitchChunks(chunks, []*types.Transcription{nil, {Text: "b", Language: "de", Emotion: "happy", Event: "speech"}}, 30, testRate)
	if got.Text != "b" || got.Language != "de" || got.Emotion != "happy" || got.Event != "speech" {
		t.Fatalf("got %+v", got)
	}
}

func TestTokensSpellText(t *testing.T) {
	tests := []struct {
		name string
		r    types.Transcription
		want bool
	}{
		{"no tokens", types.Transcription{Text: "a"}, false},
		{"untimed", types.Transcription{Text: "a", Tokens: []types.Token{{Text: "a"}}}, false},
		{"spelled", types.Transcription{Text: "hallo welt", Tokens: timed("▁hal", 0.0, "lo", 0.1, "▁welt", 0.3)}, true},
		{"differs", types.Transcription{Text: "Hallo Welt.", Tokens: timed(" hallo", 0.0, " welt", 0.3)}, false},
	}
	for _, tt := range tests {
		if got := tokensSpellText(&tt.r); got != tt.want {
			t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestGroupWords(t *testing.T) {
	words := groupWords(timed(" Zie", 0.0, "gen", 0.1, " wur", 0.2, "den", 0.3, "你", 0.4, "好", 0.5, "x", 0.6))
	var got []string
	for _, w := range words {
		got = append(got, tokensText(w))
	}
	want := []string{"Ziegen", "wurden", "你", "好", "x"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("words = %q, want %q", got, want)
	}
	if startsWord("") || endsUnspaced("") {
		t.Fatal("empty tokens start or end nothing")
	}
}

func TestDropRepeatedPrefix(t *testing.T) {
	tests := []struct{ prev, next, want string }{
		{"a b c", "b c d", "d"},
		{"Das Schwimmen ist hier sicher.", "hier sicher. Es gibt", "Es gibt"},
		{"a b c", "x y", "x y"},
		{"", "a", "a"},
		{"a", "", ""},
		{"a b", "A, b! c", "c"},
	}
	for _, tt := range tests {
		if got := dropRepeatedPrefix(tt.prev, tt.next); got != tt.want {
			t.Errorf("dropRepeatedPrefix(%q, %q) = %q, want %q", tt.prev, tt.next, got, tt.want)
		}
	}
}

func TestJoinText(t *testing.T) {
	tests := []struct{ a, b, want string }{
		{"", "b", "b"},
		{"a", "", "a"},
		{"Hallo", "Welt", "Hallo Welt"},
		{"你好", "世界", "你好世界"},
		{"こんにちは", "Tokyo", "こんにちはTokyo"},
		{"안녕", "하세요", "안녕 하세요"},
	}
	for _, tt := range tests {
		if got := joinText(tt.a, tt.b); got != tt.want {
			t.Errorf("joinText(%q, %q) = %q, want %q", tt.a, tt.b, got, tt.want)
		}
	}
}

func BenchmarkSegmentedTranscriberOverhead(b *testing.B) {
	// The cost of segmentation planning and stitching around a free decoder,
	// for ten minutes of audio cut only by the cap.
	audio := toneWithGaps(secs(600))
	inner := &recordingTranscriber{respond: func(int, []float32) (*types.Transcription, error) {
		return &types.Transcription{Text: "ein Satz mit fünf Wörtern", Tokens: timed(" ein", 0.1, " Satz", 0.5, " mit", 1.0, " fünf", 1.5, " Wörtern", 2.0)}, nil
	}}
	st, err := NewSegmentedTranscriber(inner, nil, DefaultChunkOptions())
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(audio) * 4))
	for b.Loop() {
		if _, err := st.Transcribe(context.Background(), audio, testRate); err != nil {
			b.Fatal(err)
		}
	}
}

// cutChunks is a 0-30 s span cut at 20 s with a 2 s overlap.
var cutChunks = []Chunk{
	{Start: 0, End: secs(21), KeepFrom: 0, KeepTo: secs(20)},
	{Start: secs(19), End: secs(30), KeepFrom: secs(20), KeepTo: secs(30), Continues: true},
}

func TestStitchChunksTimedThenUntimedLosesNoWords(t *testing.T) {
	// Chunk A is timed and heard "wurden dom" past the cut; chunk B has no
	// timing. The cut must be stitched by text with A's whole text, or the
	// words A heard after the cut are lost.
	a := &types.Transcription{Text: "Die Ziegen wurden dom", Tokens: timed(
		" Die", 17.0, " Zie", 18.0, "gen", 18.3, " wur", 19.8, "den", 20.3, " dom", 20.8)}
	b := &types.Transcription{Text: "Ziegen wurden domestiziert im Iran"}
	got := stitchChunks(cutChunks, []*types.Transcription{a, b}, secs(30), testRate)
	if got.Text != "Die Ziegen wurden domestiziert im Iran" {
		t.Fatalf("Text = %q, want no lost or repeated words", got.Text)
	}
	if len(got.Tokens) != 0 {
		t.Fatalf("tokens from a text-stitched chunk leaked: %+v", got.Tokens)
	}
}

func TestStitchChunksUntimedThenTimedDoesNotRepeat(t *testing.T) {
	a := &types.Transcription{Text: "Leider fällt das Studium von Verkehrsflüssen schwer"}
	b := &types.Transcription{Text: "Verkehrsflüssen schwer da man nichts weiß", Tokens: timed(
		" Verkehrsflüssen", 0.2, " schwer", 0.9, " da", 1.4, " man", 1.6, " nichts", 1.8, " weiß", 2.1)}
	got := stitchChunks(cutChunks, []*types.Transcription{a, b}, secs(30), testRate)
	want := "Leider fällt das Studium von Verkehrsflüssen schwer da man nichts weiß"
	if got.Text != want {
		t.Fatalf("Text = %q, want %q", got.Text, want)
	}
}

func TestDropRepeatedPrefixSkipsGarbledWords(t *testing.T) {
	tests := []struct{ prev, next, want string }{
		// The later chunk starts mid-word: "ssen" is a fragment of "Verkehrsflüssen".
		{"das Studium von Verkehrsflüssen schwer, da man", "ssen schwer, da man Fahrerverhalten nicht", "Fahrerverhalten nicht"},
		{"a b c d", "x y c d e", "e"},
		// One shared common word after a skip is not trusted.
		{"und die", "xx die Katze", "xx die Katze"},
		// Three garbled words is beyond the skip budget.
		{"a b c d", "x y z c d e", "x y z c d e"},
		// A fragment at the end of prev is skipped too.
		{"die Ziegen wurden dom", "Ziegen wurden domestiziert", "domestiziert"},
	}
	for _, tt := range tests {
		if got := dropRepeatedPrefix(tt.prev, tt.next); got != tt.want {
			t.Errorf("dropRepeatedPrefix(%q, %q) = %q, want %q", tt.prev, tt.next, got, tt.want)
		}
	}
}

func TestMergeOverlapTrimsTheGarbledTailOfPrev(t *testing.T) {
	prev, next := mergeOverlap("Die Ziegen wurden dom", "Ziegen wurden domestiziert")
	if prev != "Die Ziegen wurden" || next != "domestiziert" {
		t.Fatalf("mergeOverlap = (%q, %q)", prev, next)
	}
	prev, next = mergeOverlap("ganz anders", "nichts gemeinsam")
	if prev != "ganz anders" || next != "nichts gemeinsam" {
		t.Fatalf("no overlap changed text: (%q, %q)", prev, next)
	}
}
