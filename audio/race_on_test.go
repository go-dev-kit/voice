//go:build race

package audio

// raceEnabled is true when the race detector is on, which slows the
// sample-by-sample test loops by more than an order of magnitude.
const raceEnabled = true
