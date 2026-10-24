package evaluation

import (
	"errors"
	"fmt"
	"testing"
)

// syntheticPair builds n utterances of 10 reference words where system A makes
// editsA(i) errors and system B makes editsB(i).
func syntheticPair(n int, editsA, editsB func(i int) int, group func(i int) string) (a, b []UtteranceScore) {
	for i := 0; i < n; i++ {
		id := fmt.Sprintf("u%04d", i)
		g := ""
		if group != nil {
			g = group(i)
		}
		a = append(a, UtteranceScore{ID: id, Group: g, WordEdits: editsA(i), ReferenceWords: 10})
		b = append(b, UtteranceScore{ID: id, Group: g, WordEdits: editsB(i), ReferenceWords: 10})
	}
	return a, b
}

func TestPairedBootstrapDeterministicForSeed(t *testing.T) {
	a, b := syntheticPair(100, func(i int) int { return i % 4 }, func(i int) int { return (i + 1) % 3 }, nil)
	opts := BootstrapOptions{Resamples: 500, Seed: 7}
	first, err := PairedBootstrap(a, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	second, err := PairedBootstrap(a, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	if first != second {
		t.Fatalf("same seed gave different results: %+v vs %+v", first, second)
	}
	opts.Seed = 8
	other, err := PairedBootstrap(a, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	if other.Lower == first.Lower && other.Upper == first.Upper {
		t.Fatalf("different seeds gave the same interval: %+v", other)
	}
	if first.Level != DefaultBootstrapLevel || first.Resamples != 500 || first.Units != 100 || first.Grouped {
		t.Fatalf("unexpected metadata: %+v", first)
	}
}

func TestPairedBootstrapIdenticalSystems(t *testing.T) {
	a, _ := syntheticPair(50, func(i int) int { return i % 5 }, func(i int) int { return 0 }, nil)
	res, err := PairedBootstrap(a, a, BootstrapOptions{Resamples: 200})
	if err != nil {
		t.Fatal(err)
	}
	if res.Difference != 0 || res.Lower != 0 || res.Upper != 0 {
		t.Fatalf("identical systems: %+v, want difference 0 and interval [0,0]", res)
	}
}

func TestPairedBootstrapClearlyBetterSystem(t *testing.T) {
	a, b := syntheticPair(200, func(i int) int { return 2 + i%2 }, func(i int) int { return i % 2 }, nil)
	res, err := PairedBootstrap(a, b, BootstrapOptions{Seed: 3})
	if err != nil {
		t.Fatal(err)
	}
	if res.Difference <= 0 || res.Lower <= 0 {
		t.Fatalf("A is clearly worse than B, interval should exclude 0: %+v", res)
	}
	if res.Lower > res.Difference || res.Upper < res.Difference {
		t.Fatalf("interval %v..%v should contain the observed difference %v", res.Lower, res.Upper, res.Difference)
	}
	if res.WERA != 0.25 || res.WERB != 0.05 {
		t.Fatalf("pooled WER = %v, %v, want 0.25, 0.05", res.WERA, res.WERB)
	}
}

func TestPairedBootstrapGroupsWidenCorrelatedInterval(t *testing.T) {
	// 20 groups of 5 utterances; A's extra errors hit whole groups at once, so
	// utterances inside a group are strongly correlated.
	group := func(i int) string { return fmt.Sprintf("g%02d", i/5) }
	editsA := func(i int) int {
		if (i/5)%2 == 0 {
			return 3
		}
		return 0
	}
	a, b := syntheticPair(100, editsA, func(i int) int { return 1 }, group)
	opts := BootstrapOptions{Resamples: 2000, Seed: 11}
	grouped, err := PairedBootstrap(a, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a {
		a[i].Group, b[i].Group = "", ""
	}
	flat, err := PairedBootstrap(a, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	if !grouped.Grouped || grouped.Units != 20 || flat.Units != 100 {
		t.Fatalf("unexpected units: grouped=%+v flat=%+v", grouped, flat)
	}
	gw, fw := grouped.Upper-grouped.Lower, flat.Upper-flat.Lower
	if gw < fw {
		t.Fatalf("grouped width %v < utterance width %v", gw, fw)
	}
	t.Logf("grouped width %.4f, utterance width %.4f", gw, fw)
}

func TestPairedBootstrapValidation(t *testing.T) {
	ok, _ := syntheticPair(3, func(i int) int { return 1 }, func(i int) int { return 0 }, nil)
	mutate := func(f func(s []UtteranceScore)) []UtteranceScore {
		out := append([]UtteranceScore(nil), ok...)
		f(out)
		return out
	}
	tests := []struct {
		name string
		a, b []UtteranceScore
		opts BootstrapOptions
		want error
	}{
		{"empty", nil, nil, BootstrapOptions{}, ErrBootstrapEmpty},
		{"length", ok, ok[:2], BootstrapOptions{}, ErrBootstrapLengthMismatch},
		{"id", ok, mutate(func(s []UtteranceScore) { s[1].ID = "other" }), BootstrapOptions{}, ErrBootstrapPairMismatch},
		{"group", ok, mutate(func(s []UtteranceScore) { s[1].Group = "g" }), BootstrapOptions{}, ErrBootstrapPairMismatch},
		{"reference words", ok, mutate(func(s []UtteranceScore) { s[1].ReferenceWords = 9 }), BootstrapOptions{}, ErrBootstrapPairMismatch},
		{"zero reference", mutate(func(s []UtteranceScore) { s[0].ReferenceWords, s[1].ReferenceWords, s[2].ReferenceWords = 0, 0, 0 }),
			mutate(func(s []UtteranceScore) { s[0].ReferenceWords, s[1].ReferenceWords, s[2].ReferenceWords = 0, 0, 0 }), BootstrapOptions{}, ErrBootstrapNoReferenceText},
		{"negative", mutate(func(s []UtteranceScore) { s[0].WordEdits = -1 }), ok, BootstrapOptions{}, ErrBootstrapInvalidScore},
		{"level", ok, ok, BootstrapOptions{Level: 1}, ErrBootstrapInvalidOptions},
		{"resamples", ok, ok, BootstrapOptions{Resamples: -1}, ErrBootstrapInvalidOptions},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := PairedBootstrap(tt.a, tt.b, tt.opts); !errors.Is(err, tt.want) {
				t.Fatalf("err = %v, want %v", err, tt.want)
			}
		})
	}
}

func BenchmarkPairedBootstrap500x10000(b *testing.B) {
	x, y := syntheticPair(500, func(i int) int { return i % 4 }, func(i int) int { return (i + 1) % 3 }, nil)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := PairedBootstrap(x, y, BootstrapOptions{Resamples: 10000, Seed: 1}); err != nil {
			b.Fatal(err)
		}
	}
}
