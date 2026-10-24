package evaluation

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

// Errors returned by PairedBootstrap. They are wrapped with detail, so match
// them with errors.Is.
var (
	ErrBootstrapEmpty           = errors.New("bootstrap input is empty")
	ErrBootstrapLengthMismatch  = errors.New("bootstrap systems have different utterance counts")
	ErrBootstrapPairMismatch    = errors.New("bootstrap utterances are not paired")
	ErrBootstrapNoReferenceText = errors.New("bootstrap reference word total must be positive")
	ErrBootstrapInvalidScore    = errors.New("bootstrap utterance score is invalid")
	ErrBootstrapInvalidOptions  = errors.New("bootstrap options are invalid")
)

// Bootstrap defaults applied by PairedBootstrap to zero option values.
const (
	DefaultBootstrapLevel     = 0.95
	DefaultBootstrapResamples = 10000
)

// UtteranceScore is the word-level result of one system on one utterance.
type UtteranceScore struct {
	// ID identifies the utterance. When set on both sides it must match
	// between the two systems at the same index.
	ID string
	// Group names the unit of dependence, for example a speaker or a reference
	// text shared by several recordings. When any utterance has a Group,
	// PairedBootstrap resamples whole groups; utterances with an empty Group
	// are then their own singleton groups.
	Group string
	// WordEdits is the word edit count and ReferenceWords the reference word
	// count, as in TextReport.
	WordEdits      int
	ReferenceWords int
}

// UtteranceScoreFrom builds an UtteranceScore from a TextReport.
func UtteranceScoreFrom(id string, group string, report TextReport) UtteranceScore {
	return UtteranceScore{ID: id, Group: group, WordEdits: report.WordEdits, ReferenceWords: report.ReferenceWords}
}

// BootstrapOptions configures PairedBootstrap. Zero values select defaults.
type BootstrapOptions struct {
	// Level is the confidence level of the interval, in (0,1).
	// Default DefaultBootstrapLevel.
	Level float64
	// Resamples is the number of bootstrap resamples. Default
	// DefaultBootstrapResamples.
	Resamples int
	// Seed seeds the private PCG generator. The same inputs, options and seed
	// always give the same result; the zero seed is a valid fixed seed.
	Seed uint64
}

// BootstrapResult is the outcome of PairedBootstrap.
type BootstrapResult struct {
	// WERA and WERB are the corpus-level WER of each system.
	WERA float64
	WERB float64
	// Difference is WERA - WERB, a negative value meaning A is better.
	Difference float64
	// Lower and Upper bound the percentile interval of the resampled
	// difference at Level.
	Lower float64
	Upper float64
	Level float64
	// Resamples is the number of resamples that contributed to the interval.
	Resamples int
	// Units is the number of resampled units: groups when Group is used,
	// utterances otherwise.
	Units int
	// Grouped reports whether groups were resampled.
	Grouped bool
}

// PairedBootstrap compares two systems scored on the same utterances.
//
// The statistic is the corpus-level WER difference WERA - WERB, where each WER
// is pooled edits over pooled reference words (not a mean of per-utterance
// rates). Each resample draws units with replacement, keeping the two systems'
// scores for a drawn unit together (paired), and recomputes the pooled
// difference. Units are utterances, or whole groups when Group is set.
//
// Lower and Upper are the percentile interval: the empirical (1-Level)/2 and
// (1+Level)/2 quantiles of the resampled differences. It describes how much the
// observed difference would vary if this set of utterances were redrawn from the
// same population, assuming units are independent. It is NOT a p-value, is not
// a probability that the true difference lies in the interval, and says nothing
// about unmodelled sources of variation such as other speakers, domains or
// normalizers. An interval that excludes 0 is evidence the sign of the
// difference is stable on this sample, not a significance test; with few units
// the percentile interval is also too narrow.
func PairedBootstrap(a []UtteranceScore, b []UtteranceScore, opts BootstrapOptions) (BootstrapResult, error) {
	if opts.Level == 0 {
		opts.Level = DefaultBootstrapLevel
	}
	if opts.Resamples == 0 {
		opts.Resamples = DefaultBootstrapResamples
	}
	if !(opts.Level > 0 && opts.Level < 1) {
		return BootstrapResult{}, fmt.Errorf("%w: level %v is not in (0,1)", ErrBootstrapInvalidOptions, opts.Level)
	}
	if opts.Resamples < 0 {
		return BootstrapResult{}, fmt.Errorf("%w: resamples %d is negative", ErrBootstrapInvalidOptions, opts.Resamples)
	}
	if len(a) == 0 || len(b) == 0 {
		return BootstrapResult{}, ErrBootstrapEmpty
	}
	if len(a) != len(b) {
		return BootstrapResult{}, fmt.Errorf("%w: %d vs %d", ErrBootstrapLengthMismatch, len(a), len(b))
	}

	type unit struct{ editsA, editsB, words int }
	var units []unit
	groupIndex := map[string]int{}
	grouped := false
	totalWords := 0
	for i := range a {
		x, y := a[i], b[i]
		if x.ID != "" && y.ID != "" && x.ID != y.ID {
			return BootstrapResult{}, fmt.Errorf("%w: index %d has ID %q vs %q", ErrBootstrapPairMismatch, i, x.ID, y.ID)
		}
		if x.Group != y.Group {
			return BootstrapResult{}, fmt.Errorf("%w: index %d has group %q vs %q", ErrBootstrapPairMismatch, i, x.Group, y.Group)
		}
		if x.ReferenceWords != y.ReferenceWords {
			return BootstrapResult{}, fmt.Errorf("%w: index %d has %d vs %d reference words", ErrBootstrapPairMismatch, i, x.ReferenceWords, y.ReferenceWords)
		}
		for _, s := range [...]UtteranceScore{x, y} {
			if s.WordEdits < 0 || s.ReferenceWords < 0 {
				return BootstrapResult{}, fmt.Errorf("%w: index %d has negative counts", ErrBootstrapInvalidScore, i)
			}
		}
		totalWords += x.ReferenceWords
		u := unit{x.WordEdits, y.WordEdits, x.ReferenceWords}
		if x.Group == "" {
			units = append(units, u)
			continue
		}
		grouped = true
		idx, ok := groupIndex[x.Group]
		if !ok {
			idx = len(units)
			groupIndex[x.Group] = idx
			units = append(units, unit{})
		}
		units[idx].editsA += u.editsA
		units[idx].editsB += u.editsB
		units[idx].words += u.words
	}
	if totalWords <= 0 {
		return BootstrapResult{}, ErrBootstrapNoReferenceText
	}

	var sumA, sumB int
	for _, u := range units {
		sumA += u.editsA
		sumB += u.editsB
	}
	werA := float64(sumA) / float64(totalWords)
	werB := float64(sumB) / float64(totalWords)

	// A resample whose drawn units hold no reference words has an undefined
	// WER and is skipped; it can only happen when many utterances are empty.
	// A seeded PRNG is the point: results must be reproducible. It is not used
	// for anything security relevant.
	rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15)) // #nosec G404
	diffs := make([]float64, 0, opts.Resamples)
	for r := 0; r < opts.Resamples; r++ {
		var da, db, words int
		for range units {
			u := units[rng.IntN(len(units))]
			da += u.editsA
			db += u.editsB
			words += u.words
		}
		if words == 0 {
			continue
		}
		diffs = append(diffs, float64(da-db)/float64(words))
	}
	if len(diffs) == 0 {
		return BootstrapResult{}, fmt.Errorf("%w: no resample had reference words", ErrBootstrapNoReferenceText)
	}
	slices.Sort(diffs)
	tail := (1 - opts.Level) / 2
	lo := int(math.Floor(tail * float64(len(diffs))))
	hi := int(math.Ceil((1-tail)*float64(len(diffs)))) - 1
	lo = max(0, min(lo, len(diffs)-1))
	hi = max(0, min(hi, len(diffs)-1))
	return BootstrapResult{
		WERA:       werA,
		WERB:       werB,
		Difference: werA - werB,
		Lower:      diffs[lo],
		Upper:      diffs[hi],
		Level:      opts.Level,
		Resamples:  len(diffs),
		Units:      len(units),
		Grouped:    grouped,
	}, nil
}
