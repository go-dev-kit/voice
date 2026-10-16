package denoise

import (
	"context"
	"errors"
	"testing"
)

// fakeOfflineDenoiser is a mockable seam implementation used to prove
// SherpaSpeechDenoiser forwards samples/sample rate and returns the
// transformed output, without loading real model weights. Run halves each
// sample so tests can assert the transform round-trips through Denoise.
type fakeOfflineDenoiser struct {
	runSamples []float32
	runRate    int
	runErr     error
	closeCalls int
	closeErr   error
}

func (f *fakeOfflineDenoiser) Run(samples []float32, sampleRate int) (*denoisedAudio, error) {
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

func (f *fakeOfflineDenoiser) Close() error {
	f.closeCalls++
	return f.closeErr
}

func TestSherpaSpeechDenoiser_Denoise_ForwardsAndTransforms(t *testing.T) {
	fake := &fakeOfflineDenoiser{}
	d := &SherpaSpeechDenoiser{denoiser: fake}

	samples := []float32{1, 2, 3, 4}
	got, err := d.Denoise(context.Background(), samples, 16000)
	if err != nil {
		t.Fatalf("Denoise returned error: %v", err)
	}
	want := []float32{0.5, 1, 1.5, 2}
	if len(got) != len(want) {
		t.Fatalf("got %v samples, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if fake.runRate != 16000 {
		t.Fatalf("Run called with sample rate %d, want 16000", fake.runRate)
	}
	if len(fake.runSamples) != len(samples) {
		t.Fatalf("Run called with %d samples, want %d", len(fake.runSamples), len(samples))
	}
}

func TestSherpaSpeechDenoiser_Denoise_NilDenoiser(t *testing.T) {
	var d *SherpaSpeechDenoiser
	if _, err := d.Denoise(context.Background(), []float32{1}, 16000); err == nil {
		t.Fatal("expected error for nil denoiser")
	}
}

func TestSherpaSpeechDenoiser_Denoise_NilContext(t *testing.T) {
	d := &SherpaSpeechDenoiser{denoiser: &fakeOfflineDenoiser{}}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := d.Denoise(nil, []float32{1}, 16000); err == nil { //nolint:staticcheck // intentional nil-context rejection test
		t.Fatal("expected error for nil context")
	}
}

func TestSherpaSpeechDenoiser_Denoise_EmptySamples(t *testing.T) {
	d := &SherpaSpeechDenoiser{denoiser: &fakeOfflineDenoiser{}}
	if _, err := d.Denoise(context.Background(), nil, 16000); err == nil {
		t.Fatal("expected error for nil/empty samples")
	}
	if _, err := d.Denoise(context.Background(), []float32{}, 16000); err == nil {
		t.Fatal("expected error for empty samples")
	}
}

func TestSherpaSpeechDenoiser_Denoise_InvalidSampleRate(t *testing.T) {
	d := &SherpaSpeechDenoiser{denoiser: &fakeOfflineDenoiser{}}
	if _, err := d.Denoise(context.Background(), []float32{1}, 0); err == nil {
		t.Fatal("expected error for non-positive sample rate")
	}
}

func TestSherpaSpeechDenoiser_Denoise_RunError(t *testing.T) {
	fake := &fakeOfflineDenoiser{runErr: errors.New("boom")}
	d := &SherpaSpeechDenoiser{denoiser: fake}
	if _, err := d.Denoise(context.Background(), []float32{1}, 16000); err == nil {
		t.Fatal("expected error from Run failure")
	}
}

func TestSherpaSpeechDenoiser_Denoise_AfterClose(t *testing.T) {
	fake := &fakeOfflineDenoiser{}
	d := &SherpaSpeechDenoiser{denoiser: fake}
	if err := d.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if _, err := d.Denoise(context.Background(), []float32{1}, 16000); err == nil {
		t.Fatal("expected error after Close")
	}
}

func TestSherpaSpeechDenoiser_Close_IdempotentAndFreesOnce(t *testing.T) {
	fake := &fakeOfflineDenoiser{}
	d := &SherpaSpeechDenoiser{denoiser: fake}
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
