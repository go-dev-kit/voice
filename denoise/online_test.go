package denoise

import (
	"context"
	"errors"
	"testing"
)

// fakeOnlineDenoiser is a mockable seam implementation used to prove
// SherpaStreamingDenoiser forwards samples, drains on Flush, and clears
// state on Reset, without loading real model weights. Run/Flush halve each
// sample so tests can assert the transform round-trips.
type fakeOnlineDenoiser struct {
	sampleRate int

	runSamples []float32
	runRate    int
	runErr     error

	flushCalls int
	flushErr   error

	resetCalls int

	closeCalls int
	closeErr   error
}

func (f *fakeOnlineDenoiser) Run(samples []float32, sampleRate int) (*denoisedAudio, error) {
	f.runSamples = samples
	f.runRate = sampleRate
	if f.runErr != nil {
		return nil, f.runErr
	}
	out := make([]float32, len(samples))
	for i, s := range samples {
		out[i] = s / 2
	}
	return &denoisedAudio{Samples: out, SampleRate: sampleRate}, nil
}

func (f *fakeOnlineDenoiser) Flush() (*denoisedAudio, error) {
	f.flushCalls++
	if f.flushErr != nil {
		return nil, f.flushErr
	}
	return &denoisedAudio{Samples: []float32{9, 9}, SampleRate: f.sampleRate}, nil
}

func (f *fakeOnlineDenoiser) Reset() {
	f.resetCalls++
}

func (f *fakeOnlineDenoiser) SampleRate() int {
	return f.sampleRate
}

func (f *fakeOnlineDenoiser) Close() error {
	f.closeCalls++
	return f.closeErr
}

func TestSherpaStreamingDenoiser_Accept_ForwardsAndTransforms(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}

	chunk := []float32{2, 4, 6}
	got, err := d.Accept(context.Background(), chunk)
	if err != nil {
		t.Fatalf("Accept returned error: %v", err)
	}
	want := []float32{1, 2, 3}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if fake.runRate != 16000 {
		t.Fatalf("Run called with sample rate %d, want 16000 (model-fixed)", fake.runRate)
	}
}

func TestSherpaStreamingDenoiser_Accept_NilDenoiser(t *testing.T) {
	var d *SherpaStreamingDenoiser
	if _, err := d.Accept(context.Background(), []float32{1}); err == nil {
		t.Fatal("expected error for nil denoiser")
	}
}

func TestSherpaStreamingDenoiser_Accept_NilContext(t *testing.T) {
	d := &SherpaStreamingDenoiser{denoiser: &fakeOnlineDenoiser{sampleRate: 16000}, sampleRate: 16000}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := d.Accept(nil, []float32{1}); err == nil { //nolint:staticcheck // intentional nil-context rejection test
		t.Fatal("expected error for nil context")
	}
}

func TestSherpaStreamingDenoiser_Accept_EmptyChunk(t *testing.T) {
	d := &SherpaStreamingDenoiser{denoiser: &fakeOnlineDenoiser{sampleRate: 16000}, sampleRate: 16000}
	if _, err := d.Accept(context.Background(), nil); err == nil {
		t.Fatal("expected error for nil/empty chunk")
	}
	if _, err := d.Accept(context.Background(), []float32{}); err == nil {
		t.Fatal("expected error for empty chunk")
	}
}

func TestSherpaStreamingDenoiser_Accept_RunError(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000, runErr: errors.New("boom")}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}
	if _, err := d.Accept(context.Background(), []float32{1}); err == nil {
		t.Fatal("expected error from Run failure")
	}
}

func TestSherpaStreamingDenoiser_Flush_Drains(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}

	got, err := d.Flush()
	if err != nil {
		t.Fatalf("Flush returned error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %v samples, want 2", got)
	}
	if fake.flushCalls != 1 {
		t.Fatalf("native Flush called %d times, want 1", fake.flushCalls)
	}
}

func TestSherpaStreamingDenoiser_Flush_Error(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000, flushErr: errors.New("boom")}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}
	if _, err := d.Flush(); err == nil {
		t.Fatal("expected error from Flush failure")
	}
}

func TestSherpaStreamingDenoiser_Reset_ClearsState(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}

	d.Reset()
	if fake.resetCalls != 1 {
		t.Fatalf("native Reset called %d times, want 1", fake.resetCalls)
	}
}

func TestSherpaStreamingDenoiser_Reset_NilSafe(t *testing.T) {
	var d *SherpaStreamingDenoiser
	d.Reset() // must not panic
}

func TestSherpaStreamingDenoiser_Close_IdempotentAndFreesOnce(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}

	if err := d.Close(); err != nil {
		t.Fatalf("first Close returned error: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("second Close returned error: %v", err)
	}
	if fake.closeCalls != 1 {
		t.Fatalf("native Close called %d times, want 1", fake.closeCalls)
	}
}

func TestSherpaStreamingDenoiser_Accept_AfterClose(t *testing.T) {
	fake := &fakeOnlineDenoiser{sampleRate: 16000}
	d := &SherpaStreamingDenoiser{denoiser: fake, sampleRate: fake.sampleRate}
	if err := d.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if _, err := d.Accept(context.Background(), []float32{1}); err == nil {
		t.Fatal("expected error after Close")
	}
}
