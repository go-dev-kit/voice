package asr

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/types"
)

// SherpaOnlineModel implements streaming ASR with Sherpa ONNX OnlineRecognizer.
type SherpaOnlineModel struct {
	name         string
	language     string
	quantization string
	sampleRate   int
	recognizer   onlineRecognizer
	sessions     map[*onlineSession]struct{}

	mu     sync.Mutex
	closed bool
}

type onlineRecognizer interface {
	NewStream() (onlineStream, error)
	Decode(stream onlineStream) error
	IsReady(stream onlineStream) bool
	IsEndpoint(stream onlineStream) bool
	Reset(stream onlineStream) error
	Result(stream onlineStream) (*sherpa.OnlineRecognizerResult, error)
	Close() error
}

type onlineStream interface {
	AcceptWaveform(sampleRate int, samples []float32) error
	InputFinished() error
	Close() error
}

type sherpaOnlineRecognizer struct {
	recognizer *sherpa.OnlineRecognizer
}

type sherpaOnlineStream struct {
	stream *sherpa.OnlineStream
}

type onlineSession struct {
	stream          onlineStream
	acceptedSamples int64
	closed          bool
}

// NewSherpaOnlineModel builds a SherpaOnlineModel from the legacy single
// Config.Online field, deriving the model's name from config.DefaultModel and
// its language from config.Language (both unaffected by Config.OnlineModels).
// Multi-model hosting builds models via newSherpaOnlineModelFromOnline
// instead; see Config.resolvedOnlineModelConfigs and NewService.
func NewSherpaOnlineModel(config *Config) (*SherpaOnlineModel, error) {
	if config == nil {
		return nil, fmt.Errorf("config cannot be nil")
	}
	config.ApplyDefaults()
	if config.Backend != BackendSherpaOnline {
		return nil, fmt.Errorf("sherpa online recognizer requires backend %q, got %q", BackendSherpaOnline, config.Backend)
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}

	oc := config.Online
	if oc.Name == "" {
		oc.Name = config.DefaultModel
	}
	if oc.Language == "" {
		oc.Language = config.Language
	}
	return newSherpaOnlineModelFromOnline(config, oc)
}

// newSherpaOnlineModelFromOnline builds a SherpaOnlineModel for a single
// OnlineConfig entry, sharing base for runtime settings (sample rate,
// threads, provider, decoding method, quantization). Callers must ensure base
// has already been defaulted (ApplyDefaults) and that oc.Name/oc.Language are
// resolved to non-empty values.
func newSherpaOnlineModelFromOnline(base *Config, oc OnlineConfig) (*SherpaOnlineModel, error) {
	recognizerConfig, err := buildOnlineRecognizerConfigFor(base, oc)
	if err != nil {
		return nil, err
	}
	recognizer := sherpa.NewOnlineRecognizer(recognizerConfig)
	if recognizer == nil {
		return nil, fmt.Errorf("failed to create Sherpa online recognizer %q", oc.Name)
	}

	return &SherpaOnlineModel{
		name:         oc.Name,
		language:     oc.Language,
		quantization: base.Quantization,
		sampleRate:   base.SampleRate,
		recognizer:   &sherpaOnlineRecognizer{recognizer: recognizer},
		sessions:     make(map[*onlineSession]struct{}),
	}, nil
}

// buildOnlineRecognizerConfig maps the legacy single Config.Online field onto
// the native Sherpa recognizer config.
func buildOnlineRecognizerConfig(config *Config) (*sherpa.OnlineRecognizerConfig, error) {
	return buildOnlineRecognizerConfigFor(config, config.Online)
}

// buildOnlineRecognizerConfigFor maps an arbitrary OnlineConfig entry (either
// the legacy Config.Online or one Config.OnlineModels item) onto the native
// Sherpa recognizer config, using config for the runtime settings shared
// across all online models (sample rate, threads, provider, decoding method).
func buildOnlineRecognizerConfigFor(config *Config, online OnlineConfig) (*sherpa.OnlineRecognizerConfig, error) {
	modelConfig := sherpa.OnlineModelConfig{
		Tokens:        online.TokensPath,
		NumThreads:    config.NumThreads,
		Provider:      config.Provider,
		Debug:         boolToInt(config.Debug),
		ModelType:     online.ModelType,
		TokensBufSize: 0,
	}
	if online.HotwordsFile != "" || online.HotwordsText != "" {
		modelConfig.ModelingUnit = online.ModelingUnit
		modelConfig.BpeVocab = online.BpeVocab
	}

	if online.EncoderPath != "" || online.DecoderPath != "" || online.JoinerPath != "" {
		if online.EncoderPath == "" || online.DecoderPath == "" || online.JoinerPath == "" {
			return nil, fmt.Errorf("transducer encoder, decoder, and joiner paths must be provided together")
		}
		modelConfig.Transducer = sherpa.OnlineTransducerModelConfig{
			Encoder: online.EncoderPath,
			Decoder: online.DecoderPath,
			Joiner:  online.JoinerPath,
		}
	} else {
		switch strings.ToLower(online.ModelType) {
		case "", "zipformer2_ctc", "zipformer_ctc", "ctc":
			modelConfig.Zipformer2Ctc.Model = online.ModelPath
		case "nemo_ctc", "nemo":
			modelConfig.NemoCtc.Model = online.ModelPath
		case "tone_ctc", "tone":
			modelConfig.ToneCtc.Model = online.ModelPath
		default:
			return nil, fmt.Errorf("unsupported single-file online Sherpa model type %q", online.ModelType)
		}
	}

	return &sherpa.OnlineRecognizerConfig{
		FeatConfig: sherpa.FeatureConfig{
			SampleRate: config.SampleRate,
			FeatureDim: config.FeatureDim,
		},
		ModelConfig:     modelConfig,
		DecodingMethod:  config.DecodingMethod,
		MaxActivePaths:  config.MaxActivePaths,
		HotwordsFile:    online.HotwordsFile,
		HotwordsScore:   online.HotwordsScore,
		HotwordsBuf:     online.HotwordsText,
		HotwordsBufSize: len(online.HotwordsText),
	}, nil
}

func (m *SherpaOnlineModel) Name() string {
	return m.name
}

func (m *SherpaOnlineModel) Language() string {
	return m.language
}

func (m *SherpaOnlineModel) Quantization() string {
	return m.quantization
}

func (m *SherpaOnlineModel) ProcessAudio(ctx context.Context, audio []float32, state *types.StreamingState) (*types.Transcription, error) {
	return m.processAudio(ctx, audio, state, false)
}

func (m *SherpaOnlineModel) FinishAudio(ctx context.Context, audio []float32, state *types.StreamingState) (*types.Transcription, error) {
	return m.processAudio(ctx, audio, state, true)
}

func (m *SherpaOnlineModel) processAudio(ctx context.Context, audio []float32, state *types.StreamingState, finish bool) (*types.Transcription, error) {
	state, closeAfter, err := m.prepareProcessState(ctx, audio, state, finish)
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	return m.processAudioLocked(ctx, audio, state, finish, closeAfter)
}

func (m *SherpaOnlineModel) prepareProcessState(ctx context.Context, audio []float32, state *types.StreamingState, finish bool) (*types.StreamingState, bool, error) {
	if ctx == nil {
		return nil, false, fmt.Errorf("context cannot be nil")
	}
	if len(audio) == 0 && !finish {
		return nil, false, fmt.Errorf("audio cannot be empty")
	}
	localState := state == nil
	if localState {
		state = &types.StreamingState{Language: m.language}
	}
	return state, localState, nil
}

func (m *SherpaOnlineModel) processAudioLocked(ctx context.Context, audio []float32, state *types.StreamingState, finish bool, closeAfter bool) (*types.Transcription, error) {
	if m.closed || m.recognizer == nil {
		return nil, fmt.Errorf("sherpa online recognizer is closed")
	}

	if err := contextError(ctx); err != nil {
		return nil, err
	}

	session, err := m.sessionForStateLocked(state)
	if err != nil {
		return nil, err
	}
	if finish || closeAfter {
		defer func() {
			_ = m.closeSessionLocked(session)
			if state.ASRState == session {
				state.ASRState = nil
			}
		}()
	}

	if len(audio) > 0 {
		if err := session.stream.AcceptWaveform(m.sampleRate, audio); err != nil {
			return nil, err
		}
		session.acceptedSamples += int64(len(audio))
	}
	if finish {
		if err := session.stream.InputFinished(); err != nil {
			return nil, err
		}
	}

	result, isEndpoint, err := m.decodeSessionLocked(ctx, session)
	if err != nil {
		return nil, err
	}

	transcription := m.transcriptionFromOnlineResult(result, session.acceptedSamples, finish, isEndpoint)
	if isEndpoint && !finish {
		if err := m.recognizer.Reset(session.stream); err != nil {
			return nil, err
		}
		session.acceptedSamples = 0
	}

	return transcription, nil
}

func (m *SherpaOnlineModel) decodeSessionLocked(ctx context.Context, session *onlineSession) (*sherpa.OnlineRecognizerResult, bool, error) {
	for m.recognizer.IsReady(session.stream) {
		if err := contextError(ctx); err != nil {
			return nil, false, err
		}
		if err := m.recognizer.Decode(session.stream); err != nil {
			return nil, false, err
		}
	}

	result, err := m.recognizer.Result(session.stream)
	if err != nil {
		return nil, false, err
	}
	if result == nil {
		return nil, false, fmt.Errorf("sherpa online recognizer returned nil result")
	}
	return result, m.recognizer.IsEndpoint(session.stream), nil
}

func (m *SherpaOnlineModel) transcriptionFromOnlineResult(result *sherpa.OnlineRecognizerResult, acceptedSamples int64, finish bool, isEndpoint bool) *types.Transcription {
	return &types.Transcription{
		Text:       result.Text,
		IsPartial:  !finish && !isEndpoint,
		Confidence: 0,
		Language:   m.language,
		Timestamp:  time.Now(),
		StartTime:  0,
		EndTime:    time.Duration(acceptedSamples) * time.Second / time.Duration(m.sampleRate),
		Tokens:     tokensFromOnlineResult(result),
	}
}

func (m *SherpaOnlineModel) sessionForStateLocked(state *types.StreamingState) (*onlineSession, error) {
	if session, ok := state.ASRState.(*onlineSession); ok && !session.closed {
		return session, nil
	}

	if closer, ok := state.ASRState.(interface{ Close() error }); ok {
		_ = closer.Close()
	}

	stream, err := m.recognizer.NewStream()
	if err != nil {
		return nil, err
	}
	session := &onlineSession{stream: stream}
	if m.sessions == nil {
		m.sessions = make(map[*onlineSession]struct{})
	}
	m.sessions[session] = struct{}{}
	state.ASRState = session
	return session, nil
}

func (m *SherpaOnlineModel) closeSessionLocked(session *onlineSession) error {
	if session == nil {
		return nil
	}
	err := session.Close()
	delete(m.sessions, session)
	return err
}

// discardState closes and deregisters a state-owned native stream after the
// service aborts an utterance. It complements FinishAudio, which performs the
// same cleanup on the normal finalization path.
func (m *SherpaOnlineModel) discardState(state *types.StreamingState) error {
	if state == nil || state.ASRState == nil {
		return nil
	}
	session, ok := state.ASRState.(*onlineSession)
	if !ok {
		return closeASRState(state)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	err := m.closeSessionLocked(session)
	if state.ASRState == session {
		state.ASRState = nil
	}
	return err
}

func (m *SherpaOnlineModel) SupportsLanguage(lang string) bool {
	if lang == "" || m.language == "" || m.language == "auto" || m.language == "multi" || m.language == "all" {
		return true
	}
	return strings.EqualFold(m.language, lang)
}

func (m *SherpaOnlineModel) Latency() time.Duration {
	return 200 * time.Millisecond
}

func (m *SherpaOnlineModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return nil
	}
	var errs []error
	for session := range m.sessions {
		if err := session.Close(); err != nil {
			errs = append(errs, err)
		}
		delete(m.sessions, session)
	}
	if m.recognizer != nil {
		if err := m.recognizer.Close(); err != nil {
			errs = append(errs, err)
		}
		m.recognizer = nil
	}
	m.closed = true
	return errors.Join(errs...)
}

func (r *sherpaOnlineRecognizer) NewStream() (onlineStream, error) {
	stream := sherpa.NewOnlineStream(r.recognizer)
	if stream == nil {
		return nil, fmt.Errorf("failed to create sherpa online stream")
	}
	return &sherpaOnlineStream{stream: stream}, nil
}

func (r *sherpaOnlineRecognizer) Decode(stream onlineStream) error {
	sherpaStream, err := nativeOnlineStream(stream)
	if err != nil {
		return err
	}
	r.recognizer.Decode(sherpaStream)
	return nil
}

func (r *sherpaOnlineRecognizer) IsReady(stream onlineStream) bool {
	sherpaStream, err := nativeOnlineStream(stream)
	if err != nil {
		return false
	}
	return r.recognizer.IsReady(sherpaStream)
}

func (r *sherpaOnlineRecognizer) IsEndpoint(stream onlineStream) bool {
	sherpaStream, err := nativeOnlineStream(stream)
	if err != nil {
		return false
	}
	return r.recognizer.IsEndpoint(sherpaStream)
}

func (r *sherpaOnlineRecognizer) Reset(stream onlineStream) error {
	sherpaStream, err := nativeOnlineStream(stream)
	if err != nil {
		return err
	}
	r.recognizer.Reset(sherpaStream)
	return nil
}

func (r *sherpaOnlineRecognizer) Result(stream onlineStream) (*sherpa.OnlineRecognizerResult, error) {
	sherpaStream, err := nativeOnlineStream(stream)
	if err != nil {
		return nil, err
	}
	return r.recognizer.GetResult(sherpaStream), nil
}

func (r *sherpaOnlineRecognizer) Close() error {
	if r.recognizer != nil {
		sherpa.DeleteOnlineRecognizer(r.recognizer)
		r.recognizer = nil
	}
	return nil
}

func (s *sherpaOnlineStream) AcceptWaveform(sampleRate int, samples []float32) error {
	if s.stream == nil {
		return fmt.Errorf("sherpa online stream is closed")
	}
	s.stream.AcceptWaveform(sampleRate, samples)
	return nil
}

func (s *sherpaOnlineStream) InputFinished() error {
	if s.stream == nil {
		return fmt.Errorf("sherpa online stream is closed")
	}
	s.stream.InputFinished()
	return nil
}

func (s *sherpaOnlineStream) Close() error {
	if s.stream != nil {
		sherpa.DeleteOnlineStream(s.stream)
		s.stream = nil
	}
	return nil
}

func (s *onlineSession) Close() error {
	if s == nil || s.closed {
		return nil
	}
	s.closed = true
	if s.stream != nil {
		return s.stream.Close()
	}
	return nil
}

func nativeOnlineStream(stream onlineStream) (*sherpa.OnlineStream, error) {
	sherpaStream, ok := stream.(*sherpaOnlineStream)
	if !ok || sherpaStream.stream == nil {
		return nil, fmt.Errorf("unexpected online stream implementation")
	}
	return sherpaStream.stream, nil
}

func tokensFromOnlineResult(result *sherpa.OnlineRecognizerResult) []types.Token {
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
		tokens = append(tokens, token)
	}
	return tokens
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func contextError(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}
