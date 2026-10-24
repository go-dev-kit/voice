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
	// ("wolfsrudel"). It has no effect on the zero (default) Normalizer,
	// which always splits and ignores this field; use GermanNormalizer.
	SplitHyphens bool
	// KeepEszett keeps "ß" instead of folding it to "ss" in the German
	// profile. It has no effect on the zero (default) Normalizer, which
	// ignores this field; use GermanNormalizer.
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
//  6. numbers stay whole: a "," "." or ":" between digits is kept as ","
//     ("5,6" "5.6" all become "5,6"; "12:30" and "12.30" both "12,30";
//     "1.000" becomes "1,000", so thousands and decimal forms can collide),
//     a "-" or U+2212 starting a token before a digit is kept as a sign
//     ("-5"), and a hyphen between digits separates a range ("5-10" is
//     "5 10", never "510");
//  7. "%", "‰" and "°" become their own token, so "2,5%" equals "2,5 %";
//  8. every other rune that is not a letter, digit or combining mark,
//     including the German quotes „ “ ” ‚ ‘ and » «, becomes a space;
//  9. whitespace collapses to single spaces and the result is trimmed.
//
// Numbers are not verbalized: "3" and "drei" stay different tokens. Scores
// are only comparable when produced by the same Version.
func GermanNormalizer() Normalizer { return Normalizer{profile: profileGerman} }

// Version identifies the normalization rules. Record it next to every score:
// a rule change bumps it, and scores from different versions must not be
// compared.
func (n Normalizer) Version() string {
	if n.profile == profileGerman {
		return "german/1"
	}
	return "default/1"
}

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
	runes := []rune(strings.ToLower(composeUmlauts(strings.TrimSpace(text))))

	var b strings.Builder
	b.Grow(len(text))
	pendingSpace := false
	lastDigit := false // the last rune written was a digit, with no break since
	write := func(r rune) {
		if pendingSpace && b.Len() > 0 {
			b.WriteByte(' ')
		}
		pendingSpace = false
		b.WriteRune(r)
		lastDigit = unicode.IsDigit(r)
	}
	brk := func() {
		pendingSpace = true
		lastDigit = false
	}
	nextIsDigit := func(i int) bool { return i+1 < len(runes) && unicode.IsDigit(runes[i+1]) }
	for i, r := range runes {
		switch {
		case r == 'ß':
			if n.KeepEszett {
				write(r)
			} else {
				write('s')
				write('s')
			}
		case isNumberSeparator(r) && lastDigit && !pendingSpace && nextIsDigit(i):
			// "5,6", "5.6", "12:30" and "12.30" are one number token.
			write(',')
		case isMinus(r) && (b.Len() == 0 || pendingSpace) && nextIsDigit(i):
			// A sign at the start of a token: "-5" keeps its sign.
			write('-')
		case isGermanHyphen(r) && lastDigit && !pendingSpace && nextIsDigit(i):
			// A range "5-10" must not glue into "510".
			brk()
		case isGermanApostrophe(r):
		case isGermanHyphen(r):
			if n.SplitHyphens {
				brk()
			}
		case isUnitSymbol(r):
			// "2,5%" and "2,5 %" agree: the symbol is its own token.
			brk()
			write(r)
			brk()
		case unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r):
			write(r)
		default:
			brk()
		}
	}
	return b.String()
}

// isNumberSeparator reports the decimal, thousands and clock separators that
// stay inside a number token in the German profile.
func isNumberSeparator(r rune) bool { return r == ',' || r == '.' || r == ':' }

func isMinus(r rune) bool { return r == '-' || r == '−' }

// isUnitSymbol reports symbols kept as their own token in the German profile
// because they change the meaning of the number before them.
func isUnitSymbol(r rune) bool { return r == '%' || r == '‰' || r == '°' }

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

	substitutions, deletions, insertions := WordErrorBreakdown(referenceWords, hypothesisWords)

	return TextReport{
		WordErrorRate:      rate(wordEdits, len(referenceWords), len(hypothesisWords)),
		CharacterErrorRate: rate(characterEdits, len(referenceChars), len(hypothesisChars)),
		WordEdits:          wordEdits,
		ReferenceWords:     len(referenceWords),
		CharacterEdits:     characterEdits,
		ReferenceChars:     len(referenceChars),
		Substitutions:      substitutions,
		Deletions:          deletions,
		Insertions:         insertions,
	}
}
