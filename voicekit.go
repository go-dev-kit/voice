// Package voicekit provides a unified interface for voice processing operations
// including speaker recognition, diarization, and audio processing.
package voicekit

import (
	"context"
	"fmt"
	"sync"
	"time"

	voiceasr "go.glpx.pro/gdk/voice/asr"
	"go.glpx.pro/gdk/voice/audio"
	"go.glpx.pro/gdk/voice/denoise"
	"go.glpx.pro/gdk/voice/diarization"
	"go.glpx.pro/gdk/voice/speaker"
	voicetts "go.glpx.pro/gdk/voice/tts"
	"go.glpx.pro/gdk/voice/types"
)

// VoiceKit provides a unified interface for all voice processing operations
type VoiceKit struct {
	config         *Config
	speakerManager *speaker.Manager
	audioConverter *audio.Converter
	audioResampler *audio.Resampler
	diarizationMgr *diarization.Manager
	asrService     ASRService
	// asrMu guards asrService, which has a public setter that may run
	// concurrently with ASR and closeComponents.
	asrMu       sync.RWMutex
	transcriber types.Transcriber
	synthesizer types.SpeechSynthesizer
	denoiser    types.SpeechDenoiser
	metrics     *MetricsCollector

	closeOnce sync.Once
	closeErr  error
}

// newDiarizationManager is the diarization constructor; a test replaces it to
// fail a component that starts after the speaker manager.
var newDiarizationManager = diarization.NewManagerWithBackend

// NewVoiceKit creates a new VoiceKit instance with the provided configuration.
// If a component fails to start, the ones already started are closed again.
func NewVoiceKit(config *Config) (_ *VoiceKit, err error) {
	if config == nil {
		config = DefaultConfig()
	}
	config.ApplyDefaults()

	// Validate configuration at initialization
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	vk := &VoiceKit{config: config, metrics: NewMetricsCollector()}
	defer func() {
		if err != nil {
			_ = vk.Close()
		}
	}()

	// Initialize speaker manager if configured
	if config.Speaker.ModelPath != "" {
		speakerConfig := &speaker.Config{
			ModelPath:       config.Speaker.ModelPath,
			NumThreads:      config.Speaker.NumThreads,
			Provider:        config.Speaker.Provider,
			Threshold:       config.Speaker.Threshold,
			DataDir:         config.Speaker.DataDir,
			MaxSpeakers:     config.Speaker.MaxSpeakers,
			RetentionPolicy: config.Speaker.RetentionPolicy,
			MaxAge:          config.Speaker.MaxAge,
			Logger:          config.Speaker.Logger,
		}
		if vk.speakerManager, err = speaker.NewManager(speakerConfig); err != nil {
			return nil, err
		}
	}

	// Initialize audio converter
	audioConfig := &audio.ConverterConfig{
		EnableResampling:    true,
		EnableNormalization: true,
		TargetRMS:           0.1,
		TempBufferSize:      8192, // Default buffer size
		NormalizeFactor:     float32(config.Audio.NormalizeFactor),
	}
	if vk.audioConverter, err = audio.NewConverter(audioConfig); err != nil {
		return nil, fmt.Errorf("failed to create audio converter: %w", err)
	}
	vk.audioResampler = audio.NewResampler(&audio.ResampleConfig{
		Method:       audio.MethodCubic,
		Quality:      audio.QualityMedium,
		FilterLength: 16,
		UseSIMD:      false,
	})

	// Optional speech-denoiser preprocessing stage for the full pipeline.
	if config.Denoiser.GtcrnModel != "" || config.Denoiser.DpdfNetModel != "" {
		d, err := denoise.NewSherpaSpeechDenoiser(&config.Denoiser)
		if err != nil {
			return nil, fmt.Errorf("failed to create speech denoiser: %w", err)
		}
		vk.denoiser = d
	}

	if config.Diarization.Enabled {
		var speakerDB diarization.SpeakerDatabase
		var extractor diarization.EmbeddingExtractor
		if vk.speakerManager != nil {
			adapter := &speakerDatabaseAdapter{manager: vk.speakerManager}
			speakerDB = adapter
			extractor = adapter
		}
		if vk.diarizationMgr, err = newDiarizationManager(&config.Diarization, speakerDB, extractor); err != nil {
			return nil, fmt.Errorf("failed to create diarization manager: %w", err)
		}
	}

	// Initialize ASR service if enabled
	if config.ASR.Enabled {
		switch config.ASR.Backend {
		case voiceasr.BackendSherpaOnline:
			service, err := voiceasr.NewService(&config.ASR)
			if err != nil {
				return nil, fmt.Errorf("failed to create ASR service: %w", err)
			}
			vk.asrService = service
		case voiceasr.BackendSherpaOffline:
			offline, err := voiceasr.NewOfflineTranscriber(&config.ASR)
			if err != nil {
				return nil, fmt.Errorf("failed to create ASR transcriber: %w", err)
			}
			vk.transcriber = offline
		default:
			return nil, fmt.Errorf("unsupported ASR backend %q", config.ASR.Backend)
		}
	}

	if config.TTS.Enabled {
		synth, err := voicetts.NewSherpaOfflineSynthesizer(&config.TTS)
		if err != nil {
			return nil, fmt.Errorf("failed to create TTS synthesizer: %w", err)
		}
		vk.synthesizer = synth
	}

	return vk, nil
}

// ASRService defines the interface for ASR services.
type ASRService = types.ASRService

// ASRModel represents an ASR model interface.
type ASRModel = types.ASRModel

// Transcriber represents a batch/offline ASR transcriber.
type Transcriber = types.Transcriber

// SpeechSynthesizer represents a text-to-speech synthesizer.
type SpeechSynthesizer = types.SpeechSynthesizer

// SynthesisRequest represents one text-to-speech generation request.
type SynthesisRequest = types.SynthesisRequest

// SynthesizedSpeech contains generated normalized PCM samples.
type SynthesizedSpeech = types.SynthesizedSpeech

// StreamingSynthesizer synthesizes speech incrementally over an audio-chunk sink.
type StreamingSynthesizer = types.StreamingSynthesizer

// AudioChunk is a contiguous span of freshly generated PCM audio.
type AudioChunk = types.AudioChunk

// AudioChunkFunc receives generated audio chunks; returning false stops synthesis.
type AudioChunkFunc = types.AudioChunkFunc

// Punctuation restores punctuation and casing in raw ASR transcript text.
type Punctuation = types.Punctuation

// KeywordSpotter detects configured keywords/wake-words in streaming audio.
type KeywordSpotter = types.KeywordSpotter

// KeywordMatch is a keyword detected in a stream.
type KeywordMatch = types.KeywordMatch

// LanguageIdentifier identifies the spoken language of an audio segment.
type LanguageIdentifier = types.LanguageIdentifier

// LanguageResult is the detected spoken language of an audio segment.
type LanguageResult = types.LanguageResult

// SpeechDenoiser suppresses noise in a complete audio buffer.
type SpeechDenoiser = types.SpeechDenoiser

// StreamingDenoiser suppresses noise incrementally for realtime pipelines.
type StreamingDenoiser = types.StreamingDenoiser

// Close releases all resources held by VoiceKit
func (vk *VoiceKit) Close() error {
	vk.closeOnce.Do(func() { vk.closeErr = vk.closeComponents() })
	return vk.closeErr
}

func (vk *VoiceKit) closeComponents() error {
	var errs []error
	if vk.speakerManager != nil {
		vk.speakerManager.Close()
	}
	if svc := vk.asrSnapshot(); svc != nil {
		if err := svc.Close(); err != nil {
			errs = append(errs, fmt.Errorf("ASR service close failed: %w", err))
		}
	}
	if vk.transcriber != nil {
		if err := vk.transcriber.Close(); err != nil {
			errs = append(errs, fmt.Errorf("ASR transcriber close failed: %w", err))
		}
	}
	if vk.synthesizer != nil {
		if err := vk.synthesizer.Close(); err != nil {
			errs = append(errs, fmt.Errorf("TTS synthesizer close failed: %w", err))
		}
	}
	if vk.denoiser != nil {
		if err := vk.denoiser.Close(); err != nil {
			errs = append(errs, fmt.Errorf("speech denoiser close failed: %w", err))
		}
	}
	if vk.diarizationMgr != nil {
		if err := vk.diarizationMgr.Close(); err != nil {
			errs = append(errs, fmt.Errorf("diarization close failed: %w", err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("voicekit close failed: %v", errs)
	}
	return nil
}

// Speaker returns the speaker recognition manager
func (vk *VoiceKit) Speaker() *speaker.Manager {
	return vk.speakerManager
}

// Audio returns audio processing utilities
func (vk *VoiceKit) Audio() *audio.Converter {
	return vk.audioConverter
}

// Diarization returns the diarization manager
func (vk *VoiceKit) Diarization() *diarization.Manager {
	return vk.diarizationMgr
}

// Metrics returns the metrics collector
func (vk *VoiceKit) Metrics() *MetricsCollector {
	return vk.metrics
}

// ASR returns the ASR service
func (vk *VoiceKit) ASR() ASRService {
	return vk.asrSnapshot()
}

// asrSnapshot reads asrService under its lock. The returned service is
// shared: a SetASRService after this call swaps the stored value but does
// not close the returned one.
func (vk *VoiceKit) asrSnapshot() ASRService {
	vk.asrMu.RLock()
	defer vk.asrMu.RUnlock()
	return vk.asrService
}

// Transcriber returns the batch/offline ASR transcriber.
func (vk *VoiceKit) Transcriber() types.Transcriber {
	return vk.transcriber
}

// Synthesizer returns the text-to-speech synthesizer.
func (vk *VoiceKit) Synthesizer() types.SpeechSynthesizer {
	return vk.synthesizer
}

// SetASRService sets the ASR service (used to avoid circular imports).
// Safe to call concurrently with ASR and Close: readers observe either the
// old or the new service, never a torn value. Closing a replaced service
// remains the caller's responsibility.
func (vk *VoiceKit) SetASRService(service ASRService) {
	vk.asrMu.Lock()
	defer vk.asrMu.Unlock()
	vk.asrService = service
}

// ProcessAudio processes audio data through the full voice pipeline
// This is a convenience method that combines audio conversion, speaker recognition, and diarization.
//
// The method takes no context.Context, so a call cannot be cancelled once
// started: internal stages run against context.Background and run to
// completion (or first error). Keep inputs bounded or use the stage APIs
// (Speaker.Identify, diarization.Manager) directly when cancellation matters.
func (vk *VoiceKit) ProcessAudio(audioData []byte, inputConfig *audio.AudioConfig, sessionID string) (*speaker.IdentifyResult, *diarization.DiarizationResult, error) {
	if len(audioData) == 0 {
		return nil, nil, fmt.Errorf("audioData cannot be empty")
	}
	if inputConfig == nil {
		return nil, nil, fmt.Errorf("inputConfig cannot be nil")
	}
	if sessionID == "" {
		return nil, nil, fmt.Errorf("sessionID cannot be empty")
	}

	startTime := time.Now()

	// Convert audio to float32
	float32Data, err := vk.audioConverter.ConvertToFloat32(audioData, inputConfig)
	audioProcessTime := time.Since(startTime)
	vk.metrics.RecordAudioProcessing(audioProcessTime, err == nil)

	if err != nil {
		return nil, nil, err
	}

	processingAudio, processingSampleRate, err := vk.prepareProcessingAudio(float32Data, inputConfig.SampleRate)
	if err != nil {
		return nil, nil, err
	}
	if len(processingAudio) == 0 {
		return nil, nil, fmt.Errorf("audio conversion produced empty sample data")
	}

	// Speaker identification
	var speakerResult *speaker.IdentifyResult
	if vk.speakerManager != nil {
		speakerStartTime := time.Now()
		speakerResult, err = vk.speakerManager.IdentifySpeaker(processingAudio, processingSampleRate)
		speakerProcessTime := time.Since(speakerStartTime)
		vk.metrics.RecordSpeakerIdentify(speakerProcessTime, err == nil)

		if err != nil {
			return nil, nil, err
		}
	}

	var diarizationResult *diarization.DiarizationResult
	if vk.diarizationMgr != nil {
		diarizationStartTime := time.Now()
		diarizationResult, err = vk.diarizationMgr.ProcessAudio(processingAudio, processingSampleRate, sessionID)
		diarizationProcessTime := time.Since(diarizationStartTime)
		vk.metrics.RecordDiarization(diarizationProcessTime, err == nil)

		if err != nil {
			return nil, nil, err
		}
	}

	return speakerResult, diarizationResult, nil
}

func (vk *VoiceKit) prepareProcessingAudio(samples []float32, sourceRate int) ([]float32, int, error) {
	targetRate := vk.config.Audio.SampleRate
	rate := sourceRate
	if sourceRate != targetRate {
		if vk.audioResampler == nil {
			return nil, 0, fmt.Errorf("audio resampler is not initialized")
		}
		resampled, err := vk.audioResampler.Resample(samples, sourceRate, targetRate)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to normalize audio sample rate from %d Hz to %d Hz: %w", sourceRate, targetRate, err)
		}
		samples, rate = resampled, targetRate
	}

	// Optional denoise stage, applied after resampling and before speaker/diarization.
	if vk.denoiser != nil {
		denoised, err := vk.denoiser.Denoise(context.Background(), samples, rate)
		if err != nil {
			return nil, 0, fmt.Errorf("speech denoise failed: %w", err)
		}
		samples = denoised
	}

	return samples, rate, nil
}

// NewSpeakerManager creates a standalone speaker recognition manager
func NewSpeakerManager(config *speaker.Config) (*speaker.Manager, error) {
	return speaker.NewManager(config)
}

// NewAudioConverter creates a standalone audio converter
func NewAudioConverter(config *audio.ConverterConfig) (*audio.Converter, error) {
	return audio.NewConverter(config)
}

// NewDiarizationManager creates a standalone diarization manager
func NewDiarizationManager(config *diarization.DiarizationConfig, speakerDB diarization.SpeakerDatabase) (*diarization.Manager, error) {
	return diarization.NewManager(config, speakerDB)
}

// NewAudioResampler creates a standalone audio resampler
func NewAudioResampler(config *audio.ResampleConfig) *audio.Resampler {
	return audio.NewResampler(config)
}

// NewDiarizationSegmenter creates a standalone diarization segmenter
func NewDiarizationSegmenter(config *diarization.DiarizationConfig) *diarization.Segmenter {
	return diarization.NewSegmenter(config)
}

// NewDiarizationIntegrator creates a standalone diarization integrator
func NewDiarizationIntegrator(config *diarization.DiarizationConfig, manager *diarization.Manager) *diarization.Integrator {
	return diarization.NewIntegrator(config, manager)
}

// NewSpeakerParser creates a standalone speaker audio parser
func NewSpeakerParser() *speaker.AudioParser {
	return speaker.NewAudioParser()
}

// NewStreamingSynthesizer creates a standalone streaming (chunked) TTS synthesizer.
func NewStreamingSynthesizer(config *voicetts.Config) (*voicetts.SherpaStreamingSynthesizer, error) {
	return voicetts.NewSherpaStreamingSynthesizer(config)
}

// NewPunctuation creates a standalone post-ASR punctuation restorer.
func NewPunctuation(config *voiceasr.PunctuationConfig) (*voiceasr.SherpaPunctuation, error) {
	return voiceasr.NewSherpaPunctuation(config)
}

// NewKeywordSpotter creates a standalone streaming keyword/wake-word spotter.
func NewKeywordSpotter(config *voiceasr.KeywordSpotterConfig) (*voiceasr.SherpaKeywordSpotter, error) {
	return voiceasr.NewSherpaKeywordSpotter(config)
}

// NewLanguageIdentifier creates a standalone spoken-language identifier.
func NewLanguageIdentifier(config *voiceasr.LanguageIDConfig) (*voiceasr.SherpaLanguageIdentifier, error) {
	return voiceasr.NewSherpaLanguageIdentifier(config)
}

// NewSpeechDenoiser creates a standalone offline (whole-buffer) speech denoiser.
func NewSpeechDenoiser(config *denoise.DenoiserConfig) (*denoise.SherpaSpeechDenoiser, error) {
	return denoise.NewSherpaSpeechDenoiser(config)
}

// NewStreamingDenoiser creates a standalone online (per-chunk) speech denoiser.
func NewStreamingDenoiser(config *denoise.DenoiserConfig) (*denoise.SherpaStreamingDenoiser, error) {
	return denoise.NewSherpaStreamingDenoiser(config)
}

// speakerDatabaseAdapter adapts speaker.Manager to diarization.SpeakerDatabase interface
type speakerDatabaseAdapter struct {
	manager *speaker.Manager
}

func (a *speakerDatabaseAdapter) GetSpeakerEmbedding(speakerID string) ([]float32, error) {
	return a.manager.GetSpeakerEmbedding(speakerID)
}

func (a *speakerDatabaseAdapter) GetAllSpeakers() map[string][]float32 {
	return a.manager.GetAllSpeakersEmbeddings()
}

func (a *speakerDatabaseAdapter) CalculateSimilarity(embedding1, embedding2 []float32) float32 {
	return speaker.CosineSimilarity(embedding1, embedding2)
}

func (a *speakerDatabaseAdapter) RegisterSpeakerEmbedding(speakerID string, embedding []float32) error {
	return a.manager.RegisterSpeakerEmbedding(speakerID, embedding)
}

func (a *speakerDatabaseAdapter) ExtractEmbedding(ctx context.Context, audioData []float32, sampleRate int) ([]float32, error) {
	return a.manager.ExtractEmbedding(ctx, audioData, sampleRate)
}
