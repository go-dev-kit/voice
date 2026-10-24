package evaluation

import (
	"math/rand"
	"strings"
	"testing"
	"unicode"
)

// legacyNormalizeComparableText is a frozen copy of the pre-Normalizer
// implementation used to prove the default behavior did not change.
func legacyNormalizeComparableText(text string) string {
	text = strings.ToLower(strings.TrimSpace(text))
	var b strings.Builder
	lastSpace := true
	for _, r := range text {
		switch {
		case unicode.IsLetter(r) || unicode.IsNumber(r):
			b.WriteRune(r)
			lastSpace = false
		case unicode.IsSpace(r):
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		default:
			if !lastSpace {
				b.WriteByte(' ')
				lastSpace = true
			}
		}
	}
	return strings.TrimSpace(b.String())
}

func TestDefaultNormalizerMatchesLegacy(t *testing.T) {
	samples := []string{
		"", "  ", "Hello, World!", "Löwenrudel agieren ähnlich wie Wolfs- oder Hunderudel",
		"ä decomposed", "Straße", "„Zitat“ – »x«", "hab's", "tab\tnew\nline", "12.400 mg/dl",
		"İstanbul ΣΑΣ", "日本語 テスト", "é", " nbsp em",
	}
	rng := rand.New(rand.NewSource(1))
	alphabet := []rune("aAbzßäÖü ,.-–„“'’\t\n1290̈́éİΣ日")
	for range 2000 {
		n := rng.Intn(24)
		rs := make([]rune, n)
		for i := range rs {
			rs[i] = alphabet[rng.Intn(len(alphabet))]
		}
		samples = append(samples, string(rs))
	}
	for _, s := range samples {
		want := legacyNormalizeComparableText(s)
		if got := normalizeComparableText(s); got != want {
			t.Fatalf("normalizeComparableText(%q) = %q, want %q", s, got, want)
		}
		if got := DefaultNormalizer().Normalize(s); got != want {
			t.Fatalf("DefaultNormalizer(%q) = %q, want %q", s, got, want)
		}
		if got := (Normalizer{SplitHyphens: true, KeepEszett: true}).Normalize(s); got != want {
			t.Fatalf("default profile must ignore German options: %q = %q, want %q", s, got, want)
		}
	}
}

func TestGermanNormalizer(t *testing.T) {
	tests := []struct {
		name string
		n    Normalizer
		in   string
		want string
	}{
		{"fleurs sample", GermanNormalizer(),
			"Löwenrudel agieren ähnlich wie Wolfs- oder Hunderudel, Tiere, die den Löwen (aber nicht anderen Großkatzen) im Verhalten überraschend ähneln.",
			"löwenrudel agieren ähnlich wie wolfs oder hunderudel tiere die den löwen aber nicht anderen grosskatzen im verhalten überraschend ähneln"},
		{"quotes and guillemets", GermanNormalizer(), "„Hallo“ sagte ‚er‘ und »sie« «ok»", "hallo sagte er und sie ok"},
		{"en em dash separate", GermanNormalizer(), "eins–zwei—drei", "eins zwei drei"},
		{"hyphen joins", GermanNormalizer(), "Wolfs-Rudel", "wolfsrudel"},
		{"hyphen splits", Normalizer{profile: profileGerman, SplitHyphens: true}, "Wolfs-Rudel e‑mail", "wolfs rudel e mail"},
		{"apostrophe joins", GermanNormalizer(), "hab's geht’s", "habs gehts"},
		{"eszett folds", GermanNormalizer(), "Straße STRASSE GROẞ", "strasse strasse gross"},
		{"eszett kept", Normalizer{profile: profileGerman, KeepEszett: true}, "Straße", "straße"},
		{"decomposed umlauts", GermanNormalizer(), "Löwen Ärger Über", "löwen ärger über"},
		{"decomposed equals precomposed", GermanNormalizer(), "Ärger", "ärger"},
		{"trailing combining kept as mark", GermanNormalizer(), "Café ü", "café ü"},
		{"whitespace collapse", GermanNormalizer(), "  a \t\n b   ", "a b"},
		{"empty", GermanNormalizer(), " \t ", ""},
		{"decimal comma is one token", GermanNormalizer(), "12,5 mg", "12,5 mg"},
		{"decimal point equals comma", GermanNormalizer(), "5.6 mg", "5,6 mg"},
		{"clock forms agree", GermanNormalizer(), "12:30 Uhr 12.30 Uhr", "12,30 uhr 12,30 uhr"},
		{"thousands separator kept", GermanNormalizer(), "1.000 Einheiten", "1,000 einheiten"},
		{"sentence end after number", GermanNormalizer(), "Dosis 5. Dann", "dosis 5 dann"},
		{"leading minus kept", GermanNormalizer(), "-5 Grad und −3", "-5 grad und -3"},
		{"minus after space kept", GermanNormalizer(), "120/80 -5", "120 80 -5"},
		{"range does not glue", GermanNormalizer(), "5-10 mg", "5 10 mg"},
		{"range does not glue when splitting", Normalizer{profile: profileGerman, SplitHyphens: true}, "5-10", "5 10"},
		{"compound with digits still joins", GermanNormalizer(), "COVID-19", "covid19"},
		{"percent is a token", GermanNormalizer(), "2,5% gleich 2,5 %", "2,5 % gleich 2,5 %"},
		{"degree is a token", GermanNormalizer(), "38,5°C", "38,5 ° c"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.n.Normalize(tt.in); got != tt.want {
				t.Fatalf("Normalize(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestScoreTextGermanMatchesAcrossVariants(t *testing.T) {
	ref := "Die Straße „Wolfs-Rudel“ ist schön."
	hyp := "die strasse wolfsrudel ist schoen"
	de := ScoreText(GermanNormalizer(), ref, hyp)
	if de.WordEdits != 1 { // schön vs schoen only
		t.Fatalf("german word edits = %d, want 1: %+v", de.WordEdits, de)
	}
	def := ScoreText(DefaultNormalizer(), ref, hyp)
	if def.WordEdits <= de.WordEdits {
		t.Fatalf("default should score worse than german here: %+v vs %+v", def, de)
	}
}

func TestDefaultScoreTextMatchesEvaluateTranscriptText(t *testing.T) {
	got := ScoreText(DefaultNormalizer(), "We need to ship analyzer", "we need ship analyzer")
	if got.WordEdits != 1 || got.ReferenceWords != 5 {
		t.Fatalf("unexpected report %+v", got)
	}
}

func TestNormalizerVersion(t *testing.T) {
	if DefaultNormalizer().Version() == GermanNormalizer().Version() {
		t.Fatal("profiles must report different versions")
	}
	if GermanNormalizer().Version() != (Normalizer{profile: profileGerman, SplitHyphens: true}).Version() {
		t.Fatal("options must not change the profile version")
	}
}

// Normalizing twice must give the same tokens, otherwise a reference that was
// already normalized would score differently from the raw text.
func TestGermanNormalizerIsIdempotent(t *testing.T) {
	alphabet := []rune("ab5,.:-−%°ßÄü 0'„“")
	rng := rand.New(rand.NewSource(7))
	g := GermanNormalizer()
	for range 20000 {
		r := make([]rune, rng.Intn(12))
		for i := range r {
			r[i] = alphabet[rng.Intn(len(alphabet))]
		}
		once := g.Normalize(string(r))
		if twice := g.Normalize(once); twice != once {
			t.Fatalf("not idempotent for %q: %q then %q", string(r), once, twice)
		}
	}
}

func TestScoreTextReportsErrorBreakdown(t *testing.T) {
	got := ScoreText(GermanNormalizer(), "nimm 5,6 mg täglich", "nimm 5,8 mg bitte täglich jetzt")
	if got.Substitutions != 1 || got.Deletions != 0 || got.Insertions != 2 || got.WordEdits != 3 {
		t.Fatalf("breakdown = S%d D%d I%d edits %d, want S1 D0 I2 edits 3", got.Substitutions, got.Deletions, got.Insertions, got.WordEdits)
	}
}
