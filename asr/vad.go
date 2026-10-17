package asr

import (
	"fmt"
	"strings"
	"sync"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/types"
)

// VADDetector is the provider seam for voice activity detection.
type VADDetector interface {
	Process(audio []float32, state any) (*types.VADResult, error)
	Close() error
}

// VADService delegates VAD processing to the configured detector.
type VADService struct {
	config   *VADConfig
	detector VADDetector
}

// VADConfig holds VAD configuration.
type VADConfig struct {
	Provider           string  `json:"provider"`
	ModelPath          string  `json:"model_path"`
	Threshold          float32 `json:"threshold"`
	WindowSize         int     `json:"window_size"`
	SampleRate         int     `json:"sample_rate"`
	NumThreads         int     `json:"num_threads"`
	RuntimeProvider    string  `json:"runtime_provider"`
	Debug              bool    `json:"debug"`
	MinSilenceDuration float32 `json:"min_silence_duration"`
	MinSpeechDuration  float32 `json:"min_speech_duration"`
	MaxSpeechDuration  float32 `json:"max_speech_duration"`
	BufferSizeSeconds  float32 `json:"buffer_size_seconds"`
}

func vadConfigFromASR(config *Config) *VADConfig {
	return &VADConfig{
		Provider:           config.VADProvider,
		ModelPath:          config.VADModelPath,
		Threshold:          config.VADThreshold,
		WindowSize:         config.VADWindowSize,
		SampleRate:         config.SampleRate,
		NumThreads:         config.NumThreads,
		RuntimeProvider:    config.Provider,
		Debug:              config.Debug,
		MinSilenceDuration: config.VADMinSilenceDuration,
		MinSpeechDuration:  config.VADMinSpeechDuration,
		MaxSpeechDuration:  config.VADMaxSpeechDuration,
		BufferSizeSeconds:  config.VADBufferSizeSeconds,
	}
}

func defaultVADConfig() *VADConfig {
	return &VADConfig{
		Provider:           VADProviderEnergy,
		Threshold:          0.5,
		WindowSize:         512,
		SampleRate:         16000,
		NumThreads:         1,
		RuntimeProvider:    "cpu",
		MinSilenceDuration: 0.5,
		MinSpeechDuration:  0.25,
		MaxSpeechDuration:  20,
		BufferSizeSeconds:  10,
	}
}

// normalizeVADProvider resolves common short provider names to their canonical
// VAD provider constants, case-insensitively (mirrors the online model-type
// aliasing in sherpa_online.go). Unrecognized values are returned unchanged so
// the caller's switch/default still hard-errors on truly unsupported
// providers, with the original (un-normalized) value in the error message.
func normalizeVADProvider(provider string) string {
	switch strings.ToLower(provider) {
	case "silero":
		return VADProviderSilero
	case "ten":
		return VADProviderTen
	default:
		return provider
	}
}

// NewVADService creates a VAD service for the named provider. This is an
// exported constructor that callers can use directly, bypassing
// asr.Config.Validate() — provider aliasing is applied here too so both paths
// accept the same short names.
func NewVADService(config *VADConfig) (*VADService, error) {
	if config == nil {
		config = defaultVADConfig()
	}
	applyVADDefaults(config)
	config.Provider = normalizeVADProvider(config.Provider)

	var detector VADDetector
	var err error
	switch config.Provider {
	case VADProviderNone:
		detector = passThroughVAD{}
	case VADProviderEnergy:
		detector = &energyVAD{threshold: config.Threshold}
	case VADProviderSilero, VADProviderTen:
		detector, err = newSherpaVAD(config)
	default:
		return nil, fmt.Errorf("unsupported VAD provider %q", config.Provider)
	}
	if err != nil {
		return nil, err
	}

	return &VADService{config: config, detector: detector}, nil
}

func applyVADDefaults(config *VADConfig) {
	defaults := defaultVADConfig()
	if config.Provider == "" {
		config.Provider = defaults.Provider
	}
	if config.Threshold <= 0 {
		config.Threshold = defaults.Threshold
	}
	if config.WindowSize <= 0 {
		config.WindowSize = defaults.WindowSize
	}
	if config.SampleRate <= 0 {
		config.SampleRate = defaults.SampleRate
	}
	if config.NumThreads <= 0 {
		config.NumThreads = defaults.NumThreads
	}
	if config.RuntimeProvider == "" {
		config.RuntimeProvider = defaults.RuntimeProvider
	}
	if config.MinSilenceDuration <= 0 {
		config.MinSilenceDuration = defaults.MinSilenceDuration
	}
	if config.MinSpeechDuration <= 0 {
		config.MinSpeechDuration = defaults.MinSpeechDuration
	}
	if config.MaxSpeechDuration <= 0 {
		config.MaxSpeechDuration = defaults.MaxSpeechDuration
	}
	if config.BufferSizeSeconds <= 0 {
		config.BufferSizeSeconds = defaults.BufferSizeSeconds
	}
}

// Process processes audio for voice activity detection.
func (v *VADService) Process(audio []float32, state any) (*types.VADResult, error) {
	if v == nil || v.detector == nil {
		return nil, fmt.Errorf("VAD service is not initialized")
	}
	return v.detector.Process(audio, state)
}

// Close closes the VAD service.
func (v *VADService) Close() error {
	if v == nil || v.detector == nil {
		return nil
	}
	return v.detector.Close()
}

type passThroughVAD struct{}

func (passThroughVAD) Process(audio []float32, state any) (*types.VADResult, error) {
	return &types.VADResult{
		IsSpeech:   len(audio) > 0,
		IsEndpoint: false,
		Confidence: 0,
		State:      state,
	}, nil
}

func (passThroughVAD) Close() error {
	return nil
}

type energyVAD struct {
	threshold float32
}

func (v *energyVAD) Process(audio []float32, state any) (*types.VADResult, error) {
	if len(audio) == 0 {
		return &types.VADResult{IsSpeech: false, IsEndpoint: false, Confidence: 0, State: state}, nil
	}

	// meanSquare is the average per-sample energy (not RMS: no square root is
	// taken). The threshold is compared against this mean-square energy directly.
	var sum float32
	for _, sample := range audio {
		sum += sample * sample
	}
	meanSquare := sum / float32(len(audio))

	return &types.VADResult{
		IsSpeech:   meanSquare > v.threshold,
		IsEndpoint: false,
		Confidence: 0,
		State:      state,
	}, nil
}

func (v *energyVAD) Close() error {
	return nil
}

type sherpaVAD struct {
	provider string
	vad      *sherpa.VoiceActivityDetector
	mu       sync.Mutex
}

func newSherpaVAD(config *VADConfig) (*sherpaVAD, error) {
	if config.ModelPath == "" {
		return nil, fmt.Errorf("VAD model path is required for provider %s", config.Provider)
	}

	switch config.Provider {
	case VADProviderSilero, VADProviderTen:
	default:
		return nil, fmt.Errorf("unsupported sherpa VAD provider %q", config.Provider)
	}
	modelConfig := sherpaVADModelConfig(config)
	vad := sherpa.NewVoiceActivityDetector(&modelConfig, config.BufferSizeSeconds)
	if vad == nil {
		return nil, fmt.Errorf("failed to create sherpa VAD provider %s", config.Provider)
	}
	return &sherpaVAD{provider: config.Provider, vad: vad}, nil
}

// sherpaVADModelConfig maps a Silero or TEN VAD config to the native config.
func sherpaVADModelConfig(config *VADConfig) sherpa.VadModelConfig {
	modelConfig := sherpa.VadModelConfig{
		SampleRate: config.SampleRate,
		NumThreads: config.NumThreads,
		Provider:   config.RuntimeProvider,
		Debug:      boolToInt(config.Debug),
	}
	model := sherpa.SileroVadModelConfig{
		Model:              config.ModelPath,
		Threshold:          config.Threshold,
		MinSilenceDuration: config.MinSilenceDuration,
		MinSpeechDuration:  config.MinSpeechDuration,
		WindowSize:         config.WindowSize,
		MaxSpeechDuration:  config.MaxSpeechDuration,
	}
	if config.Provider == VADProviderTen {
		modelConfig.TenVad = sherpa.TenVadModelConfig(model)
	} else {
		modelConfig.SileroVad = model
	}
	return modelConfig
}

func (v *sherpaVAD) Process(audio []float32, state any) (*types.VADResult, error) {
	if len(audio) == 0 {
		return &types.VADResult{IsSpeech: false, IsEndpoint: false, Confidence: 0, State: state}, nil
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if v.vad == nil {
		return nil, fmt.Errorf("sherpa VAD provider %s is closed", v.provider)
	}

	v.vad.AcceptWaveform(audio)
	isSpeech := v.vad.IsSpeech()
	isEndpoint := false
	for !v.vad.IsEmpty() {
		segment := v.vad.Front()
		v.vad.Pop()
		if segment != nil && len(segment.Samples) > 0 {
			isEndpoint = true
		}
	}

	return &types.VADResult{
		IsSpeech:   isSpeech || isEndpoint,
		IsEndpoint: isEndpoint,
		Confidence: 0,
		State:      state,
	}, nil
}

func (v *sherpaVAD) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.vad != nil {
		sherpa.DeleteVoiceActivityDetector(v.vad)
		v.vad = nil
	}
	return nil
}
