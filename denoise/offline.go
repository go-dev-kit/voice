package denoise

import (
	"context"
	"fmt"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/types"
)

// SherpaSpeechDenoiser implements types.SpeechDenoiser with Sherpa ONNX's
// offline speech-enhancement model (GTCRN or DPDFNet), denoising a complete
// audio buffer in one call.
type SherpaSpeechDenoiser struct {
	config   *DenoiserConfig
	denoiser offlineDenoiser
	mu       sync.Mutex
	closed   bool
}

// offlineDenoiser is the mockable seam over the native Sherpa
// OfflineSpeechDenoiser binding, so tests can inject a fake without loading
// real model weights.
type offlineDenoiser interface {
	Run(samples []float32, sampleRate int) (*denoisedAudio, error)
	Close() error
}

// denoisedAudio mirrors sherpa.DenoisedAudio, decoupling the exported API
// from the vendored binding's result type.
type denoisedAudio struct {
	Samples    []float32
	SampleRate int
}

// NewSherpaSpeechDenoiser validates config and constructs an offline speech
// denoiser backed by a native Sherpa OfflineSpeechDenoiser instance.
func NewSherpaSpeechDenoiser(config *DenoiserConfig) (*SherpaSpeechDenoiser, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	config.ApplyDefaults()
	if err := config.Validate(); err != nil {
		return nil, err
	}

	nativeConfig := buildOfflineDenoiserConfig(config)
	native := sherpa.NewOfflineSpeechDenoiser(nativeConfig)
	if native == nil {
		return nil, fmt.Errorf("failed to create Sherpa offline speech denoiser")
	}

	return &SherpaSpeechDenoiser{
		config:   config,
		denoiser: &nativeOfflineDenoiser{sd: native},
	}, nil
}

func buildOfflineDenoiserConfig(config *DenoiserConfig) *sherpa.OfflineSpeechDenoiserConfig {
	model := sherpa.OfflineSpeechDenoiserModelConfig{
		NumThreads: int32(config.NumThreads), // #nosec G115 -- Validate bounds NumThreads to the native int32 range.
		Debug:      boolToInt32(config.Debug),
		Provider:   config.Provider,
	}
	if config.GtcrnModel != "" {
		model.Gtcrn = sherpa.OfflineSpeechDenoiserGtcrnModelConfig{Model: config.GtcrnModel}
	}
	if config.DpdfNetModel != "" {
		model.DpdfNet = sherpa.OfflineSpeechDenoiserDpdfNetModelConfig{Model: config.DpdfNetModel}
	}

	return &sherpa.OfflineSpeechDenoiserConfig{Model: model}
}

// Denoise suppresses noise in a complete audio buffer, returning denoised
// PCM samples.
func (d *SherpaSpeechDenoiser) Denoise(ctx context.Context, samples []float32, sampleRate int) ([]float32, error) {
	if d == nil {
		return nil, fmt.Errorf("denoiser cannot be nil")
	}
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}
	if len(samples) == 0 {
		return nil, fmt.Errorf("samples cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sampleRate must be positive, got %d", sampleRate)
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed || d.denoiser == nil {
		return nil, fmt.Errorf("sherpa speech denoiser is closed")
	}

	audio, err := d.denoiser.Run(samples, sampleRate)
	if err != nil {
		return nil, err
	}
	if audio == nil {
		return nil, fmt.Errorf("sherpa speech denoiser run failed")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	return audio.Samples, nil
}

// Close releases the native offline speech denoiser. It is safe to call
// more than once.
func (d *SherpaSpeechDenoiser) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	if d.denoiser != nil {
		if err := d.denoiser.Close(); err != nil {
			return err
		}
		d.denoiser = nil
	}
	d.closed = true
	return nil
}

var _ types.SpeechDenoiser = (*SherpaSpeechDenoiser)(nil)

type nativeOfflineDenoiser struct {
	sd *sherpa.OfflineSpeechDenoiser
}

func (n *nativeOfflineDenoiser) Run(samples []float32, sampleRate int) (*denoisedAudio, error) {
	if n == nil || n.sd == nil {
		return nil, fmt.Errorf("sherpa offline speech denoiser is closed")
	}
	audio := n.sd.Run(samples, sampleRate)
	if audio == nil {
		return nil, fmt.Errorf("sherpa offline speech denoiser run failed")
	}
	return &denoisedAudio{Samples: audio.Samples, SampleRate: audio.SampleRate}, nil
}

func (n *nativeOfflineDenoiser) Close() error {
	if n == nil || n.sd == nil {
		return nil
	}
	sherpa.DeleteOfflineSpeechDenoiser(n.sd)
	n.sd = nil
	return nil
}

func boolToInt32(v bool) int32 {
	if v {
		return 1
	}
	return 0
}
