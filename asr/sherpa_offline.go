package asr

import (
	"context"
	"fmt"
	"sync"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/types"
)

// SherpaOfflineModel implements batch ASR with Sherpa ONNX OfflineRecognizer.
// One Transcribe call is one decode: input longer than the model handles is
// refused (see OfflineConfig.MaxDecodeSeconds), never truncated. Use
// NewOfflineTranscriber for audio of any length.
type SherpaOfflineModel struct {
	name         string
	language     string
	quantization string
	recognizer   offlineRecognizer
	// maxDecode is the longest input one decode accepts; zero means no limit.
	maxDecode time.Duration

	// slots bounds concurrent decodes on the shared native recognizer; its
	// capacity is OfflineConfig.MaxConcurrentDecodes (one when unset).
	slots     chan struct{}
	slotsOnce sync.Once

	// mu guards recognizer and closed: decodes hold the read lock, Close the
	// write lock, so Close waits for in-flight decodes.
	mu     sync.RWMutex
	closed bool
}

// decodeSlots returns the decode semaphore, defaulting to one slot.
func (m *SherpaOfflineModel) decodeSlots() chan struct{} {
	m.slotsOnce.Do(func() {
		if m.slots == nil {
			m.slots = make(chan struct{}, 1)
		}
	})
	return m.slots
}

type offlineRecognizer interface {
	NewStream() (offlineStream, error)
	Decode(stream offlineStream) error
	Close() error
}

type offlineStream interface {
	AcceptWaveform(sampleRate int, samples []float32) error
	Result() *sherpa.OfflineRecognizerResult
	Close() error
}

func NewSherpaOfflineModel(config *Config) (*SherpaOfflineModel, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	config.ApplyDefaults()
	if config.Backend != BackendSherpaOffline {
		return nil, fmt.Errorf("sherpa offline recognizer requires backend %q, got %q", BackendSherpaOffline, config.Backend)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}

	recognizerConfig, err := buildOfflineRecognizerConfig(config)
	if err != nil {
		return nil, err
	}
	recognizer := sherpa.NewOfflineRecognizer(recognizerConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create Sherpa offline recognizer")
	}

	return &SherpaOfflineModel{
		name:         config.DefaultModel,
		language:     config.Offline.Language,
		quantization: config.Quantization,
		recognizer:   &nativeOfflineRecognizer{recognizer: recognizer},
		maxDecode:    secondsToDuration(config.Offline.maxDecodeSeconds()),
		slots:        make(chan struct{}, max(config.Offline.MaxConcurrentDecodes, 1)),
	}, nil
}

// NewOfflineTranscriber returns the long-audio-safe offline transcriber for
// config: a SherpaOfflineModel wrapped in a SegmentedTranscriber that cuts the
// input into decodes of at most config.Offline.ChunkOptions().MaxChunk. With a
// Silero or TEN VAD configured (VADProvider, VADModelPath) it decodes only the
// speech; otherwise the whole input is cut at quiet points.
func NewOfflineTranscriber(config *Config) (*SegmentedTranscriber, error) {
	model, err := NewSherpaOfflineModel(config)
	if err != nil {
		return nil, err
	}
	segmenter, err := offlineSegmenter(config)
	if err != nil {
		_ = model.Close()
		return nil, fmt.Errorf("create speech segmenter: %w", err)
	}
	transcriber, err := NewSegmentedTranscriber(model, segmenter, config.Offline.ChunkOptions())
	if err != nil {
		_ = model.Close()
		if segmenter != nil {
			_ = segmenter.Close()
		}
		return nil, err
	}
	return transcriber, nil
}

// offlineSegmenter returns the speech segmenter config asks for: a Sherpa VAD
// for silero_vad/ten_vad (short names accepted), nil for an empty provider,
// none and energy. An empty provider means the Config default (none) even on a
// config that never had defaults applied.
func offlineSegmenter(config *Config) (SpeechSegmenter, error) {
	switch provider := normalizeVADProvider(config.VADProvider); provider {
	case "", VADProviderNone, VADProviderEnergy:
		return nil, nil
	default:
		vad := vadConfigFromASR(config)
		vad.Provider = provider
		return NewSpeechSegmenter(vad)
	}
}

func buildOfflineRecognizerConfig(config *Config) (*sherpa.OfflineRecognizerConfig, error) {
	offline := config.Offline
	modelType := offline.ModelType
	if modelType == "" && hintsSherpaModelType(offline.ModelFamily) {
		modelType = offline.ModelFamily
	}

	modelConfig := sherpa.OfflineModelConfig{
		Tokens:     offline.TokensPath,
		NumThreads: config.NumThreads,
		Debug:      boolToInt(config.Debug),
		Provider:   config.Provider,
		ModelType:  modelType,
	}

	switch offline.ModelFamily {
	case OfflineFamilyTransducer:
		modelConfig.Transducer = sherpa.OfflineTransducerModelConfig{
			Encoder: offline.EncoderPath,
			Decoder: offline.DecoderPath,
			Joiner:  offline.JoinerPath,
		}
	case OfflineFamilyParaformer:
		modelConfig.Paraformer = sherpa.OfflineParaformerModelConfig{Model: offline.ModelPath}
	case OfflineFamilyZipformerCTC:
		modelConfig.ZipformerCtc = sherpa.OfflineZipformerCtcModelConfig{Model: offline.ModelPath}
	case OfflineFamilyNemoCTC:
		modelConfig.NemoCTC = sherpa.OfflineNemoEncDecCtcModelConfig{Model: offline.ModelPath}
	case OfflineFamilySenseVoice:
		modelConfig.SenseVoice = sherpa.OfflineSenseVoiceModelConfig{
			Model:                       offline.ModelPath,
			Language:                    offline.Language,
			UseInverseTextNormalization: boolToInt(offline.UseInverseTextNormalization),
		}
	case OfflineFamilyWhisper:
		modelConfig.Whisper = sherpa.OfflineWhisperModelConfig{
			Encoder:                 offline.EncoderPath,
			Decoder:                 offline.DecoderPath,
			Language:                offline.Language,
			Task:                    offline.Task,
			TailPaddings:            offline.TailPaddings,
			EnableTokenTimestamps:   boolToInt(offline.EnableTokenTimestamps),
			EnableSegmentTimestamps: boolToInt(offline.EnableSegmentTimestamps),
		}
	default:
		if err := applyExtendedOfflineFamily(&modelConfig, offline); err != nil {
			return nil, err
		}
	}

	return &sherpa.OfflineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{
			SampleRate: config.SampleRate,
			FeatureDim: config.FeatureDim,
		},
		ModelConfig:    modelConfig,
		DecodingMethod: config.DecodingMethod,
		MaxActivePaths: config.MaxActivePaths,
	}, nil
}

// hintsSherpaModelType reports whether the family name doubles as the sherpa
// model_type hint when OfflineConfig.ModelType is empty. Only the original six
// families do; the extended families leave the hint empty so sherpa detects the
// model from its metadata, which is what the serving app did before this
// package owned the mapping.
func hintsSherpaModelType(family string) bool {
	switch family {
	case OfflineFamilyTransducer, OfflineFamilyParaformer, OfflineFamilyZipformerCTC,
		OfflineFamilyNemoCTC, OfflineFamilySenseVoice, OfflineFamilyWhisper:
		return true
	}
	return false
}

// applyExtendedOfflineFamily maps the offline families added after the original
// six onto modelConfig. It returns an error for a family it does not know.
//
// Tokens (already set from OfflineConfig.TokensPath by the caller) is cleared
// for funasr_nano and qwen3_asr: sherpa reads their vocabulary from the
// tokenizer directory and never uses tokens.txt.
func applyExtendedOfflineFamily(modelConfig *sherpa.OfflineModelConfig, offline OfflineConfig) error {
	switch offline.ModelFamily {
	case OfflineFamilyFunASRNano:
		modelConfig.Tokens = ""
		modelConfig.FunAsrNano = sherpa.OfflineFunASRNanoModelConfig{
			EncoderAdaptor:              offline.EncoderAdaptorPath,
			LLM:                         offline.LLMPath,
			Embedding:                   offline.EmbeddingPath,
			Tokenizer:                   offline.TokenizerPath,
			SystemPrompt:                offline.SystemPrompt,
			UserPrompt:                  offline.UserPrompt,
			MaxNewTokens:                offline.MaxNewTokens,
			Temperature:                 offline.Temperature,
			TopP:                        offline.TopP,
			Seed:                        offline.Seed,
			Language:                    offline.Language,
			UseInverseTextNormalization: boolToInt(offline.UseInverseTextNormalization),
			Hotwords:                    offline.Hotwords,
		}
	case OfflineFamilyMoonshine:
		modelConfig.Moonshine = sherpa.OfflineMoonshineModelConfig{
			Preprocessor:    offline.PreprocessorPath,
			Encoder:         offline.EncoderPath,
			UncachedDecoder: offline.UncachedDecoderPath,
			CachedDecoder:   offline.CachedDecoderPath,
			MergedDecoder:   offline.MergedDecoderPath,
		}
	case OfflineFamilyQwen3ASR:
		modelConfig.Tokens = ""
		modelConfig.Qwen3ASR = sherpa.OfflineQwen3ASRModelConfig{
			ConvFrontend: offline.ConvFrontendPath,
			Encoder:      offline.EncoderPath,
			Decoder:      offline.DecoderPath,
			Tokenizer:    offline.TokenizerPath,
			MaxTotalLen:  offline.MaxTotalLen,
			MaxNewTokens: offline.MaxNewTokens,
			Temperature:  offline.Temperature,
			TopP:         offline.TopP,
			Seed:         offline.Seed,
			Hotwords:     offline.Hotwords,
		}
	case OfflineFamilyCanary:
		modelConfig.Canary = sherpa.OfflineCanaryModelConfig{
			Encoder: offline.EncoderPath,
			Decoder: offline.DecoderPath,
			SrcLang: offline.SrcLang,
			TgtLang: offline.TgtLang,
			UsePnc:  boolToInt(offline.UsePunctuation),
		}
	case OfflineFamilyCohereTranscribe:
		modelConfig.CohereTranscribe = sherpa.OfflineCohereTranscribeModelConfig{
			Encoder:                     offline.EncoderPath,
			Decoder:                     offline.DecoderPath,
			Language:                    offline.Language,
			UsePunct:                    boolToInt(offline.UsePunctuation),
			UseInverseTextNormalization: boolToInt(offline.UseInverseTextNormalization),
		}
	case OfflineFamilyWenetCTC:
		modelConfig.WenetCtc = sherpa.OfflineWenetCtcModelConfig{Model: offline.ModelPath}
	case OfflineFamilyOmnilingualCTC:
		modelConfig.Omnilingual = sherpa.OfflineOmnilingualAsrCtcModelConfig{Model: offline.ModelPath}
	case OfflineFamilyMedASRCTC:
		modelConfig.MedAsr = sherpa.OfflineMedAsrCtcModelConfig{Model: offline.ModelPath}
	case OfflineFamilyFireRedASRCTC:
		modelConfig.FireRedAsrCtc = sherpa.OfflineFireRedAsrCtcModelConfig{Model: offline.ModelPath}
	case OfflineFamilyDolphin:
		modelConfig.Dolphin = sherpa.OfflineDolphinModelConfig{Model: offline.ModelPath}
	case OfflineFamilyTDNN:
		modelConfig.Tdnn = sherpa.OfflineTdnnModelConfig{Model: offline.ModelPath}
	case OfflineFamilyFireRedASR:
		modelConfig.FireRedAsr = sherpa.OfflineFireRedAsrModelConfig{
			Encoder: offline.EncoderPath,
			Decoder: offline.DecoderPath,
		}
	default:
		return fmt.Errorf("unsupported offline model family %q", offline.ModelFamily)
	}
	return nil
}

func (m *SherpaOfflineModel) Transcribe(ctx context.Context, audio []float32, sampleRate int) (transcription *types.Transcription, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("context cannot be nil")
	}
	if len(audio) == 0 {
		return nil, fmt.Errorf("audio cannot be empty")
	}
	if sampleRate <= 0 {
		return nil, fmt.Errorf("sampleRate must be positive, got %d", sampleRate)
	}
	if duration := samplesToDuration(len(audio), sampleRate); m.maxDecode > 0 && duration > m.maxDecode {
		return nil, &AudioTooLongError{Duration: duration, Limit: m.maxDecode}
	}

	// Wait for a decode slot; a caller that gives up while queued leaves
	// without costing a decode.
	slots := m.decodeSlots()
	select {
	case slots <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-slots }()
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.closed || m.recognizer == nil {
		return nil, fmt.Errorf("sherpa offline recognizer is closed")
	}

	stream, err := m.recognizer.NewStream()
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := stream.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}()

	if err := stream.AcceptWaveform(sampleRate, audio); err != nil {
		return nil, err
	}
	if err := m.recognizer.Decode(stream); err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	result := stream.Result()
	return transcriptionFromOfflineResult(result, m.language, len(audio), sampleRate), nil
}

func (m *SherpaOfflineModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	if m.recognizer != nil {
		if err := m.recognizer.Close(); err != nil {
			return err
		}
		m.recognizer = nil
	}
	m.closed = true
	return nil
}

func transcriptionFromOfflineResult(result *sherpa.OfflineRecognizerResult, fallbackLanguage string, sampleCount, sampleRate int) *types.Transcription {
	language := fallbackLanguage
	text := ""
	emotion := ""
	event := ""
	if result != nil {
		text = result.Text
		emotion = result.Emotion
		event = result.Event
		if result.Lang != "" {
			language = result.Lang
		}
	}

	return &types.Transcription{
		Text:       text,
		IsPartial:  false,
		Confidence: 0,
		Language:   language,
		Emotion:    emotion,
		Event:      event,
		Timestamp:  time.Now(),
		StartTime:  0,
		EndTime:    time.Duration(sampleCount) * time.Second / time.Duration(sampleRate),
		Tokens:     tokensFromOfflineResult(result),
	}
}

func tokensFromOfflineResult(result *sherpa.OfflineRecognizerResult) []types.Token {
	if result == nil || len(result.Tokens) == 0 {
		return nil
	}

	tokens := make([]types.Token, 0, len(result.Tokens))
	for i, text := range result.Tokens {
		token := types.Token{Text: text}
		if i < len(result.Timestamps) {
			token.HasTiming = true
			token.StartTime = time.Duration(result.Timestamps[i] * float32(time.Second))
		}
		if i < len(result.Durations) && result.Durations[i] > 0 {
			token.Duration = time.Duration(result.Durations[i] * float32(time.Second))
		}
		tokens = append(tokens, token)
	}
	return tokens
}

type nativeOfflineRecognizer struct {
	recognizer *sherpa.OfflineRecognizer
}

func (r *nativeOfflineRecognizer) NewStream() (offlineStream, error) {
	if r == nil || r.recognizer == nil {
		return nil, fmt.Errorf("sherpa offline recognizer is closed")
	}
	stream := sherpa.NewOfflineStream(r.recognizer)
	if stream == nil {
		return nil, fmt.Errorf("failed to create Sherpa offline stream")
	}
	return &nativeOfflineStream{stream: stream}, nil
}

func (r *nativeOfflineRecognizer) Decode(stream offlineStream) error {
	if r == nil || r.recognizer == nil {
		return fmt.Errorf("sherpa offline recognizer is closed")
	}
	native, ok := stream.(*nativeOfflineStream)
	if !ok || native.stream == nil {
		return fmt.Errorf("unsupported offline stream implementation")
	}
	r.recognizer.Decode(native.stream)
	return nil
}

func (r *nativeOfflineRecognizer) Close() error {
	if r == nil || r.recognizer == nil {
		return nil
	}
	sherpa.DeleteOfflineRecognizer(r.recognizer)
	r.recognizer = nil
	return nil
}

type nativeOfflineStream struct {
	stream *sherpa.OfflineStream
}

func (s *nativeOfflineStream) AcceptWaveform(sampleRate int, samples []float32) error {
	if s == nil || s.stream == nil {
		return fmt.Errorf("sherpa offline stream is closed")
	}
	if sampleRate <= 0 {
		return fmt.Errorf("sampleRate must be positive, got %d", sampleRate)
	}
	if len(samples) == 0 {
		return fmt.Errorf("audio cannot be empty")
	}
	s.stream.AcceptWaveform(sampleRate, samples)
	return nil
}

func (s *nativeOfflineStream) Result() *sherpa.OfflineRecognizerResult {
	if s == nil || s.stream == nil {
		return nil
	}
	return s.stream.GetResult()
}

func (s *nativeOfflineStream) Close() error {
	if s == nil || s.stream == nil {
		return nil
	}
	sherpa.DeleteOfflineStream(s.stream)
	s.stream = nil
	return nil
}
