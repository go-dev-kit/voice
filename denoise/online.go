package denoise

import (
	"context"
	"fmt"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/types"
)

// SherpaStreamingDenoiser implements types.StreamingDenoiser with Sherpa
// ONNX's online (streaming) speech-enhancement model (GTCRN or DPDFNet).
// The stream's sample rate is fixed by the model at construction time
// (see sherpa.OnlineSpeechDenoiser.SampleRate); Accept expects chunks at
// that rate.
type SherpaStreamingDenoiser struct {
	config     *DenoiserConfig
	denoiser   onlineDenoiser
	sampleRate int
	mu         sync.Mutex
	closed     bool
}

// onlineDenoiser is the mockable seam over the native Sherpa
// OnlineSpeechDenoiser binding, so tests can inject a fake without loading
// real model weights.
type onlineDenoiser interface {
	Run(samples []float32, sampleRate int) (*denoisedAudio, error)
	Flush() (*denoisedAudio, error)
	Reset()
	SampleRate() int
	Close() error
}

// NewSherpaStreamingDenoiser validates config and constructs a streaming
// speech denoiser backed by a native Sherpa OnlineSpeechDenoiser instance.
func NewSherpaStreamingDenoiser(config *DenoiserConfig) (*SherpaStreamingDenoiser, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	config.ApplyDefaults()
	if err := config.Validate(); err != nil {
		return nil, err
	}

	nativeConfig := buildOnlineDenoiserConfig(config)
	native := sherpa.NewOnlineSpeechDenoiser(nativeConfig)
	if native == nil {
		return nil, fmt.Errorf("failed to create Sherpa online speech denoiser")
	}

	wrapped := &nativeOnlineDenoiser{sd: native}
	return &SherpaStreamingDenoiser{
		config:     config,
		denoiser:   wrapped,
		sampleRate: wrapped.SampleRate(),
	}, nil
}

func buildOnlineDenoiserConfig(config *DenoiserConfig) *sherpa.OnlineSpeechDenoiserConfig {
	// OnlineSpeechDenoiserConfig reuses OfflineSpeechDenoiserModelConfig for
	// its Model field (sherpa_onnx.go ~:2592), so this shares the same
	// model-config shape as the offline denoiser.
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

	return &sherpa.OnlineSpeechDenoiserConfig{Model: model}
}

// Accept feeds a chunk of audio at the stream's model-fixed sample rate and
// returns whatever denoised samples are ready.
func (d *SherpaStreamingDenoiser) Accept(ctx context.Context, chunk []float32) ([]float32, error) {
	if d == nil {
		return nil, fmt.Errorf("denoiser cannot be nil")
	}
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}
	if len(chunk) == 0 {
		return nil, fmt.Errorf("chunk cannot be empty")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed || d.denoiser == nil {
		return nil, fmt.Errorf("sherpa streaming denoiser is closed")
	}

	audio, err := d.denoiser.Run(chunk, d.sampleRate)
	if err != nil {
		return nil, err
	}
	if audio == nil {
		return nil, fmt.Errorf("sherpa streaming denoiser run failed")
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	return audio.Samples, nil
}

// Flush drains any remaining buffered audio at end of stream.
func (d *SherpaStreamingDenoiser) Flush() ([]float32, error) {
	if d == nil {
		return nil, fmt.Errorf("denoiser cannot be nil")
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed || d.denoiser == nil {
		return nil, fmt.Errorf("sherpa streaming denoiser is closed")
	}

	audio, err := d.denoiser.Flush()
	if err != nil {
		return nil, err
	}
	if audio == nil {
		return nil, fmt.Errorf("sherpa streaming denoiser flush failed")
	}
	return audio.Samples, nil
}

// Reset clears stream state to begin a new stream on the same instance.
func (d *SherpaStreamingDenoiser) Reset() {
	if d == nil {
		return
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed || d.denoiser == nil {
		return
	}
	d.denoiser.Reset()
}

// Close releases the native online speech denoiser. It is safe to call more
// than once.
func (d *SherpaStreamingDenoiser) Close() error {
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

var _ types.StreamingDenoiser = (*SherpaStreamingDenoiser)(nil)

type nativeOnlineDenoiser struct {
	sd *sherpa.OnlineSpeechDenoiser
}

func (n *nativeOnlineDenoiser) Run(samples []float32, sampleRate int) (*denoisedAudio, error) {
	if n == nil || n.sd == nil {
		return nil, fmt.Errorf("sherpa online speech denoiser is closed")
	}
	audio := n.sd.Run(samples, sampleRate)
	if audio == nil {
		return nil, fmt.Errorf("sherpa online speech denoiser run failed")
	}
	return &denoisedAudio{Samples: audio.Samples, SampleRate: audio.SampleRate}, nil
}

func (n *nativeOnlineDenoiser) Flush() (*denoisedAudio, error) {
	if n == nil || n.sd == nil {
		return nil, fmt.Errorf("sherpa online speech denoiser is closed")
	}
	audio := n.sd.Flush()
	if audio == nil {
		return nil, fmt.Errorf("sherpa online speech denoiser flush failed")
	}
	return &denoisedAudio{Samples: audio.Samples, SampleRate: audio.SampleRate}, nil
}

func (n *nativeOnlineDenoiser) Reset() {
	if n == nil || n.sd == nil {
		return
	}
	n.sd.Reset()
}

func (n *nativeOnlineDenoiser) SampleRate() int {
	if n == nil || n.sd == nil {
		return 0
	}
	return n.sd.SampleRate()
}

func (n *nativeOnlineDenoiser) Close() error {
	if n == nil || n.sd == nil {
		return nil
	}
	sherpa.DeleteOnlineSpeechDenoiser(n.sd)
	n.sd = nil
	return nil
}
