package evaluation

import (
	"strings"
	"unicode"
)

// Normalizer canonicalizes text before WER/CER comparison. The zero value is
// the default normalizer used by EvaluateTranscriptText: lowercase, every
// non-letter and non-digit rune becomes a separator, whitespace collapses.
// Its output is byte-for-byte what earlier releases produced.
//
// Use GermanNormalizer for German reference text. A Normalizer is a small
// immutable value and is safe for concurrent use.
type Normalizer struct {
	profile normalizerProfile

	// SplitHyphens makes the German profile split hyphenated compounds
	// ("Wolfs-Rudel" becomes "wolfs rudel") instead of joining them
	// ("wolfsrudel"). Ignored by the default profile, which always splits.
	SplitHyphens bool
	// KeepEszett keeps "ß" instead of folding it to "ss" in the German
	// profile. Ignored by the default profile.
	KeepEszett bool
}

type normalizerProfile uint8

const (
	profileDefault normalizerProfile = iota
	profileGerman
)

// DefaultNormalizer returns the legacy normalizer (the zero Normalizer).
func DefaultNormalizer() Normalizer { return Normalizer{} }

// GermanNormalizer returns the German scoring profile:
//
//  1. compose "a", "o", "u" (either case) followed by U+0308 into the
//     precomposed umlaut. This is a partial NFC: full NFC needs
//     golang.org/x/text/unicode/norm, which this module does not depend on,
//     so other decomposed sequences (for example "e" + U+0301) are kept as
//     letter plus combining mark rather than composed;
//  2. lowercase, keeping umlauts;
//  3. fold "ß" (and "ẞ") to "ss" unless KeepEszett is set;
//  4. delete apostrophe look-alikes (' ’ ʼ ´ `) so "hab's" is one token "habs";
//  5. delete hyphens (- U+2010 U+2011) joining a compound, or turn them into
//     a space when SplitHyphens is set; en and em dashes always separate;
//  6. every other rune that is not a letter, digit or combining mark,
//     including the German quotes „ “ ” ‚ ‘ and » «, becomes a space;
//  7. whitespace collapses to single spaces and the result is trimmed.
//
// Numbers are not verbalized: "3" and "drei" stay different tokens.
func GermanNormalizer() Normalizer { return Normalizer{profile: profileGerman} }

// Normalize returns the canonical comparable form of text.
func (n Normalizer) Normalize(text string) string {
	if n.profile == profileGerman {
		return n.normalizeGerman(text)
	}
	return normalizeComparableText(text)
}

// Words returns the normalized whitespace-separated tokens of text.
func (n Normalizer) Words(text string) []string {
	text = n.Normalize(text)
	if text == "" {
		return nil
	}
	return strings.Fields(text)
}

// Runes returns the normalized text as runes, the unit used for CER.
func (n Normalizer) Runes(text string) []rune {
	text = n.Normalize(text)
	if text == "" {
		return nil
	}
	return []rune(text)
}

func (n Normalizer) normalizeGerman(text string) string {
	text = composeUmlauts(strings.TrimSpace(text))
	text = strings.ToLower(text)

	var b strings.Builder
	b.Grow(len(text))
	pendingSpace := false
	write := func(r rune) {
		if pendingSpace && b.Len() > 0 {
			b.WriteByte(' ')
		}
		pendingSpace = false
		b.WriteRune(r)
	}
	for _, r := range text {
		switch {
		case r == 'ß':
			if n.KeepEszett {
				write(r)
			} else {
				write('s')
				write('s')
			}
		case isGermanApostrophe(r):
		case isGermanHyphen(r):
			if n.SplitHyphens {
				pendingSpace = true
			}
		case unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r):
			write(r)
		default:
			pendingSpace = true
		}
	}
	return b.String()
}

func isGermanApostrophe(r rune) bool {
	switch r {
	case '\'', '’', 'ʼ', '´', '`':
		return true
	}
	return false
}

func isGermanHyphen(r rune) bool {
	switch r {
	case '-', '‐', '‑':
		return true
	}
	return false
}

// composeUmlauts folds the decomposed forms of the six German umlaut capitals
// and smalls (base letter + U+0308) into their precomposed runes.
func composeUmlauts(text string) string {
	if !strings.ContainsRune(text, '̈') {
		return text
	}
	var b strings.Builder
	b.Grow(len(text))
	var pending rune
	for _, r := range text {
		if pending != 0 {
			if r == '̈' {
				b.WriteRune(umlautOf(pending))
				pending = 0
				continue
			}
			b.WriteRune(pending)
			pending = 0
		}
		if umlautOf(r) != r {
			pending = r
			continue
		}
		b.WriteRune(r)
	}
	if pending != 0 {
		b.WriteRune(pending)
	}
	return b.String()
}

func umlautOf(r rune) rune {
	switch r {
	case 'a':
		return 'ä'
	case 'o':
		return 'ö'
	case 'u':
		return 'ü'
	case 'A':
		return 'Ä'
	case 'O':
		return 'Ö'
	case 'U':
		return 'Ü'
	}
	return r
}

// ScoreText computes WER/CER of hypothesis against reference with n.
func ScoreText(n Normalizer, reference string, hypothesis string) TextReport {
	referenceWords := n.Words(reference)
	hypothesisWords := n.Words(hypothesis)
	wordEdits := levenshteinStrings(referenceWords, hypothesisWords)

	referenceChars := n.Runes(reference)
	hypothesisChars := n.Runes(hypothesis)
	characterEdits := levenshteinRunes(referenceChars, hypothesisChars)

	return TextReport{
		WordErrorRate:      rate(wordEdits, len(referenceWords), len(hypothesisWords)),
		CharacterErrorRate: rate(characterEdits, len(referenceChars), len(hypothesisChars)),
		WordEdits:          wordEdits,
		ReferenceWords:     len(referenceWords),
		CharacterEdits:     characterEdits,
		ReferenceChars:     len(referenceChars),
	}
}
