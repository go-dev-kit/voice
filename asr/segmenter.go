package asr

import (
	"context"
	"fmt"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/audio"
)

// SpeechSegmenter finds the speech in a complete audio buffer.
type SpeechSegmenter interface {
	// SpeechSpans returns the speech spans of audio, in samples at sampleRate.
	SpeechSpans(ctx context.Context, audio []float32, sampleRate int) ([]SpeechSpan, error)
	Close() error
}

// segmenterFeed is how much audio the segmenter hands the VAD at once; the
// context is checked between feeds.
const segmenterFeed = 0.5 // seconds

// minSegmenterBufferSeconds keeps the native VAD ring buffer larger than the
// longest segment Silero emits in practice. Its max_speech_duration is not a
// hard cap: 43-50 s segments were measured with it set to 30.
const minSegmenterBufferSeconds = 120

// SherpaSegmenter segments complete audio with a Sherpa Silero or TEN VAD.
// Each call runs its own native detector, so calls may run concurrently.
type SherpaSegmenter struct {
	config      VADConfig
	modelConfig sherpa.VadModelConfig
	resampler   *audio.Resampler
}

// NewSpeechSegmenter returns the segmenter for config.Provider: a
// SherpaSegmenter for silero_vad and ten_vad, or nil for none and energy, which
// means the whole buffer is one span. It loads the VAD model once to fail fast
// on a bad model path.
func NewSpeechSegmenter(config *VADConfig) (SpeechSegmenter, error) {
	if config == nil {
		return nil, nil
	}
	cfg := *config
	applyVADDefaults(&cfg)
	cfg.Provider = normalizeVADProvider(cfg.Provider)
	switch cfg.Provider {
	case VADProviderNone, VADProviderEnergy:
		return nil, nil
	case VADProviderSilero, VADProviderTen:
	default:
		return nil, fmt.Errorf("unsupported VAD provider %q", cfg.Provider)
	}
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("VAD model path is required for provider %s", cfg.Provider)
	}
	// Check the file before the native loader sees it; a missing model there
	// is a log line and a nil detector at best.
	if err := requireFile(cfg.ModelPath); err != nil {
		return nil, fmt.Errorf("VAD model path: %w", err)
	}

	s := &SherpaSegmenter{
		config:      cfg,
		modelConfig: sherpaVADModelConfig(&cfg),
		resampler:   audio.NewResampler(&audio.ResampleConfig{Method: audio.MethodSinc, FilterLength: 16}),
	}
	vad, err := s.newDetector()
	if err != nil {
		return nil, err
	}
	sherpa.DeleteVoiceActivityDetector(vad)
	return s, nil
}

func (s *SherpaSegmenter) newDetector() (*sherpa.VoiceActivityDetector, error) {
	buffer := max(s.config.BufferSizeSeconds, minSegmenterBufferSeconds)
	vad := sherpa.NewVoiceActivityDetector(&s.modelConfig, buffer)
	if vad == nil {
		return nil, fmt.Errorf("failed to create sherpa VAD provider %s", s.config.Provider)
	}
	return vad, nil
}

// SpeechSpans runs the VAD over samples and returns the speech spans. Audio at
// another rate than the VAD's is resampled for detection, and the spans are
// mapped back to the caller's rate.
func (s *SherpaSegmenter) SpeechSpans(ctx context.Context, samples []float32, sampleRate int) ([]SpeechSpan, error) {
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sampleRate must be positive, got %d", sampleRate)
	}
	if len(samples) == 0 {
		return nil, nil
	}
	vadRate := s.config.SampleRate
	input := samples
	if sampleRate != vadRate {
		resampled, err := s.resampler.Resample(samples, sampleRate, vadRate)
		if err != nil {
			return nil, fmt.Errorf("resample for VAD: %w", err)
		}
		input = resampled
	}

	vad, err := s.newDetector()
	if err != nil {
		return nil, err
	}
	defer sherpa.DeleteVoiceActivityDetector(vad)

	var spans []SpeechSpan
	drain := func() {
		for !vad.IsEmpty() {
			segment := vad.Front()
			vad.Pop()
			if segment != nil && len(segment.Samples) > 0 {
				spans = append(spans, SpeechSpan{Start: segment.Start, End: segment.Start + len(segment.Samples)})
			}
		}
	}
	feed := max(int(segmenterFeed*float64(vadRate)), 1)
	for start := 0; start < len(input); start += feed {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vad.AcceptWaveform(input[start:min(start+feed, len(input))])
		drain()
	}
	vad.Flush()
	drain()

	if sampleRate != vadRate {
		for i := range spans {
			spans[i].Start = min(spans[i].Start*sampleRate/vadRate, len(samples))
			spans[i].End = min((spans[i].End*sampleRate+vadRate-1)/vadRate, len(samples))
		}
	}
	return spans, nil
}

// Close releases nothing: detectors live only for one SpeechSpans call.
func (s *SherpaSegmenter) Close() error {
	return nil
}
