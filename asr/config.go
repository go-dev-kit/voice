package asr

import (
	"fmt"
	"maps"
	"math"
	"os"
	"slices"
	"time"
)

const (
	BackendSherpaOnline  = "sherpa_online"
	BackendSherpaOffline = "sherpa_offline"

	OfflineFamilyTransducer   = "transducer"
	OfflineFamilyParaformer   = "paraformer"
	OfflineFamilyZipformerCTC = "zipformer_ctc"
	OfflineFamilyNemoCTC      = "nemo_ctc"
	OfflineFamilySenseVoice   = "sense_voice"
	OfflineFamilyWhisper      = "whisper"

	// Families added after the original six. Each maps onto the Sherpa
	// offline model config of the same name; see applyExtendedOfflineFamily.
	OfflineFamilyFunASRNano       = "funasr_nano"
	OfflineFamilyMoonshine        = "moonshine"
	OfflineFamilyQwen3ASR         = "qwen3_asr"
	OfflineFamilyCanary           = "canary"
	OfflineFamilyCohereTranscribe = "cohere_transcribe"
	OfflineFamilyWenetCTC         = "wenet_ctc"
	OfflineFamilyOmnilingualCTC   = "omnilingual_ctc"
	OfflineFamilyMedASRCTC        = "med_asr_ctc"
	OfflineFamilyFireRedASRCTC    = "firered_asr_ctc"
	OfflineFamilyDolphin          = "dolphin"
	OfflineFamilyTDNN             = "tdnn"
	OfflineFamilyFireRedASR       = "firered_asr"

	// VAD providers. "none" and "energy" are dependency-free; the neural
	// providers require a model path (see validateVAD). Recommended default for
	// production is Silero VAD (v6.x, MIT-licensed, ~6000 languages). TEN VAD can
	// have lower turn-detection latency but ships under a modified Apache-2.0
	// license, so it is opt-in and left to the caller after a license review.
	VADProviderNone   = "none"
	VADProviderEnergy = "energy"
	VADProviderSilero = "silero_vad"
	VADProviderTen    = "ten_vad"
)

// Logger is the logging contract used by ASR without importing the root package.
type Logger interface {
	Infof(format string, args ...any)
	Warnf(format string, args ...any)
	Errorf(format string, args ...any)
}

// OnlineConfig owns Sherpa online/streaming model paths. Name and Language are
// only required when this entry is one of several in Config.OnlineModels; the
// legacy single-model Config.Online path derives them from
// Config.DefaultModel and Config.Language when left empty.
type OnlineConfig struct {
	Name        string `json:"name,omitempty"`
	Language    string `json:"language,omitempty"`
	TokensPath  string `json:"tokens_path"`
	EncoderPath string `json:"encoder_path"`
	DecoderPath string `json:"decoder_path"`
	JoinerPath  string `json:"joiner_path"`
	ModelPath   string `json:"model_path"`
	ModelType   string `json:"model_type"`
}

// OfflineConfig owns Sherpa offline/batch model paths and family-specific options.
type OfflineConfig struct {
	ModelFamily string `json:"model_family"`
	ModelType   string `json:"model_type,omitempty"`

	TokensPath  string `json:"tokens_path,omitempty"`
	EncoderPath string `json:"encoder_path,omitempty"`
	DecoderPath string `json:"decoder_path,omitempty"`
	JoinerPath  string `json:"joiner_path,omitempty"`
	ModelPath   string `json:"model_path,omitempty"`

	// Multi-file layouts of the extended families. Moonshine v1 uses
	// PreprocessorPath, EncoderPath, UncachedDecoderPath and CachedDecoderPath;
	// v2 uses EncoderPath and MergedDecoderPath.
	PreprocessorPath    string `json:"preprocessor_path,omitempty"`
	UncachedDecoderPath string `json:"uncached_decoder_path,omitempty"`
	CachedDecoderPath   string `json:"cached_decoder_path,omitempty"`
	MergedDecoderPath   string `json:"merged_decoder_path,omitempty"`
	ConvFrontendPath    string `json:"conv_frontend_path,omitempty"`
	EncoderAdaptorPath  string `json:"encoder_adaptor_path,omitempty"`
	LLMPath             string `json:"llm_path,omitempty"`
	EmbeddingPath       string `json:"embedding_path,omitempty"`
	// TokenizerPath is a directory (vocab.json, merges.txt, ...) for
	// funasr_nano and qwen3_asr, not a file.
	TokenizerPath string `json:"tokenizer_path,omitempty"`

	Language                    string `json:"language,omitempty"`
	Task                        string `json:"task,omitempty"`
	TailPaddings                int    `json:"tail_paddings,omitempty"`
	EnableTokenTimestamps       bool   `json:"enable_token_timestamps,omitempty"`
	EnableSegmentTimestamps     bool   `json:"enable_segment_timestamps,omitempty"`
	UseInverseTextNormalization bool   `json:"use_inverse_text_normalization,omitempty"`

	// Canary translation languages and the punctuation flag shared by canary
	// and cohere_transcribe.
	SrcLang        string `json:"src_lang,omitempty"`
	TgtLang        string `json:"tgt_lang,omitempty"`
	UsePunctuation bool   `json:"use_punctuation,omitempty"`

	// LLM-decoder options used by funasr_nano and qwen3_asr (MaxTotalLen only
	// by qwen3_asr; the prompts only by funasr_nano). Zero values keep the
	// Sherpa defaults.
	SystemPrompt string  `json:"system_prompt,omitempty"`
	UserPrompt   string  `json:"user_prompt,omitempty"`
	MaxTotalLen  int     `json:"max_total_len,omitempty"`
	MaxNewTokens int     `json:"max_new_tokens,omitempty"`
	Temperature  float32 `json:"temperature,omitempty"`
	TopP         float32 `json:"top_p,omitempty"`
	Seed         int     `json:"seed,omitempty"`
	Hotwords     string  `json:"hotwords,omitempty"`

	// MaxDecodeSeconds makes one SherpaOfflineModel decode refuse longer input
	// with an *AudioTooLongError instead of letting the model truncate it.
	// Zero means the family default: 30 for whisper, whose fixed window keeps
	// only the first 30 s, and no limit otherwise.
	MaxDecodeSeconds float32 `json:"max_decode_seconds,omitempty"`

	// MaxConcurrentDecodes is how many decodes may share the native
	// recognizer at once; zero or one serializes them. Each decode still uses
	// NumThreads threads, so size the product to the cores available.
	// Concurrent decodes are verified to match a lone decode for the
	// transducer (Parakeet TDT) and whisper families. qwen3_asr and
	// funasr_nano sample with an LLM decoder that may keep sampler state in
	// the recognizer, so they always decode one at a time.
	MaxConcurrentDecodes int `json:"max_concurrent_decodes,omitempty"`

	// Chunking used by NewOfflineTranscriber; zero means the
	// DefaultChunkOptions value. MaxChunkSeconds cannot exceed
	// MaxDecodeDuration.
	MaxChunkSeconds       float32 `json:"max_chunk_seconds,omitempty"`
	ChunkOverlapSeconds   float32 `json:"chunk_overlap_seconds,omitempty"`
	SpeechPaddingSeconds  float32 `json:"speech_padding_seconds,omitempty"`
	SpeechMergeGapSeconds float32 `json:"speech_merge_gap_seconds,omitempty"`
}

// effectiveConcurrentDecodes is MaxConcurrentDecodes clamped to the families
// that can share a recognizer: the LLM-decoder families stay serialized until
// a gated equivalence test shows concurrent decodes give a lone decode's text.
func (o OfflineConfig) effectiveConcurrentDecodes() int {
	switch o.ModelFamily {
	case OfflineFamilyQwen3ASR, OfflineFamilyFunASRNano:
		return 1
	}
	return max(o.MaxConcurrentDecodes, 1)
}

// whisperWindowSeconds is the fixed input window of Whisper models.
const whisperWindowSeconds = 30

// maxDecodeSeconds resolves MaxDecodeSeconds against the family default.
func (o OfflineConfig) maxDecodeSeconds() float32 {
	if o.MaxDecodeSeconds > 0 {
		return o.MaxDecodeSeconds
	}
	if o.ModelFamily == OfflineFamilyWhisper {
		return whisperWindowSeconds
	}
	return 0
}

// secondsToDuration converts a float32 config value in seconds to a duration
// rounded to the millisecond, so 0.1 becomes 100ms rather than 100.000001ms.
func secondsToDuration(v float32) time.Duration {
	return time.Duration(math.Round(float64(v)*1000)) * time.Millisecond
}

// ChunkOptions resolves the chunking fields against DefaultChunkOptions.
func (o OfflineConfig) ChunkOptions() ChunkOptions {
	opts := DefaultChunkOptions()
	seconds := secondsToDuration
	if o.MaxChunkSeconds > 0 {
		opts.MaxChunk = seconds(o.MaxChunkSeconds)
	}
	if o.ChunkOverlapSeconds > 0 {
		opts.Overlap = seconds(o.ChunkOverlapSeconds)
	}
	if o.SpeechPaddingSeconds > 0 {
		opts.Padding = seconds(o.SpeechPaddingSeconds)
	}
	if o.SpeechMergeGapSeconds > 0 {
		opts.MergeGap = seconds(o.SpeechMergeGapSeconds)
	}
	return opts
}

// Config owns ASR runtime configuration.
type Config struct {
	Enabled bool `json:"enabled"`

	DefaultModel string `json:"default_model"`
	Language     string `json:"language"`
	Quantization string `json:"quantization"`

	MaxConcurrentStreams int `json:"max_concurrent_streams"`
	StreamTimeout        int `json:"stream_timeout"` // seconds
	ChunkSize            int `json:"chunk_size"`
	SampleRate           int `json:"sample_rate"`
	FeatureDim           int `json:"feature_dim"`

	Backend        string `json:"backend"`
	Provider       string `json:"provider"`
	NumThreads     int    `json:"num_threads"`
	Debug          bool   `json:"debug"`
	DecodingMethod string `json:"decoding_method"`
	MaxActivePaths int    `json:"max_active_paths"`

	Online  OnlineConfig  `json:"online"`
	Offline OfflineConfig `json:"offline"`

	// OnlineModels hosts multiple online/streaming models in one Service, each
	// routed by StreamingState.Language via SelectModel (see
	// Service.SetSessionLanguage). Each entry requires a distinct non-empty
	// Name. When empty, NewService falls back to the single legacy Online
	// model for backward compatibility.
	OnlineModels []OnlineConfig `json:"online_models,omitempty"`

	VADProvider           string  `json:"vad_provider"`
	VADModelPath          string  `json:"vad_model_path"`
	VADThreshold          float32 `json:"vad_threshold"`
	VADMinSilenceDuration float32 `json:"vad_min_silence_duration"`
	VADMinSpeechDuration  float32 `json:"vad_min_speech_duration"`
	VADMaxSpeechDuration  float32 `json:"vad_max_speech_duration"`
	VADWindowSize         int     `json:"vad_window_size"`
	VADBufferSizeSeconds  float32 `json:"vad_buffer_size_seconds"`

	Logger Logger `json:"-"`
}

// DefaultConfig returns audio-only ASR defaults. It does not enable ASR.
func DefaultConfig() Config {
	return Config{
		Enabled:               false,
		DefaultModel:          BackendSherpaOffline,
		Language:              "en",
		Quantization:          "float32",
		MaxConcurrentStreams:  10,
		StreamTimeout:         300,
		ChunkSize:             16000,
		SampleRate:            16000,
		FeatureDim:            80,
		Backend:               BackendSherpaOffline,
		Provider:              "cpu",
		NumThreads:            1,
		DecodingMethod:        "greedy_search",
		MaxActivePaths:        4,
		VADProvider:           VADProviderNone,
		VADThreshold:          0.5,
		VADMinSilenceDuration: 0.5,
		VADMinSpeechDuration:  0.25,
		VADMaxSpeechDuration:  20,
		VADWindowSize:         512,
		VADBufferSizeSeconds:  10,
		Offline: OfflineConfig{
			ModelFamily:  OfflineFamilySenseVoice,
			Task:         "transcribe",
			TailPaddings: -1,
		},
	}
}

// ApplyDefaults merges zero-value runtime settings with conservative defaults.
func (c *Config) ApplyDefaults() {
	defaults := DefaultConfig()
	c.applyModelDefaults(defaults)
	c.applyRuntimeDefaults(defaults)
	c.applyVADDefaults(defaults)
	c.applyOfflineDefaults(defaults)
	c.applyOnlineModelsDefaults()
}

func (c *Config) applyModelDefaults(defaults Config) {
	if c.Backend == "" {
		c.Backend = defaults.Backend
	}
	if c.DefaultModel == "" {
		c.DefaultModel = c.Backend
	}
	if c.Language == "" {
		c.Language = defaults.Language
	}
	if c.Quantization == "" {
		c.Quantization = defaults.Quantization
	}
	if c.Provider == "" {
		c.Provider = defaults.Provider
	}
	if c.DecodingMethod == "" {
		c.DecodingMethod = defaults.DecodingMethod
	}
}

func (c *Config) applyRuntimeDefaults(defaults Config) {
	if c.MaxConcurrentStreams <= 0 {
		c.MaxConcurrentStreams = defaults.MaxConcurrentStreams
	}
	if c.StreamTimeout <= 0 {
		c.StreamTimeout = defaults.StreamTimeout
	}
	if c.ChunkSize <= 0 {
		c.ChunkSize = defaults.ChunkSize
	}
	if c.SampleRate <= 0 {
		c.SampleRate = defaults.SampleRate
	}
	if c.FeatureDim <= 0 {
		c.FeatureDim = defaults.FeatureDim
	}
	if c.NumThreads <= 0 {
		c.NumThreads = defaults.NumThreads
	}
	if c.MaxActivePaths <= 0 {
		c.MaxActivePaths = defaults.MaxActivePaths
	}
}

func (c *Config) applyVADDefaults(defaults Config) {
	if c.VADProvider == "" {
		c.VADProvider = defaults.VADProvider
	}
	if c.VADThreshold <= 0 {
		c.VADThreshold = defaults.VADThreshold
	}
	if c.VADMinSilenceDuration <= 0 {
		c.VADMinSilenceDuration = defaults.VADMinSilenceDuration
	}
	if c.VADMinSpeechDuration <= 0 {
		c.VADMinSpeechDuration = defaults.VADMinSpeechDuration
	}
	if c.VADMaxSpeechDuration <= 0 {
		c.VADMaxSpeechDuration = defaults.VADMaxSpeechDuration
	}
	if c.VADWindowSize <= 0 {
		c.VADWindowSize = defaults.VADWindowSize
	}
	if c.VADBufferSizeSeconds <= 0 {
		c.VADBufferSizeSeconds = defaults.VADBufferSizeSeconds
	}
}

// applyOnlineModelsDefaults defaults each OnlineModels entry's Language from
// the top-level Language when left unset. Name is intentionally left as-is:
// Validate requires it to be explicit and non-empty for every entry.
func (c *Config) applyOnlineModelsDefaults() {
	for i := range c.OnlineModels {
		if c.OnlineModels[i].Language == "" {
			c.OnlineModels[i].Language = c.Language
		}
	}
}

func (c *Config) applyOfflineDefaults(defaults Config) {
	if c.Offline.ModelFamily == "" {
		c.Offline.ModelFamily = defaults.Offline.ModelFamily
	}
	if c.Offline.Language == "" {
		c.Offline.Language = c.Language
	}
	if c.Offline.Task == "" {
		c.Offline.Task = defaults.Offline.Task
	}
	if c.Offline.TailPaddings == 0 {
		c.Offline.TailPaddings = defaults.Offline.TailPaddings
	}
}

// Validate validates ASR configuration. Disabled ASR skips runtime model checks.
func (c *Config) Validate() error {
	c.ApplyDefaults()
	if !c.Enabled {
		return nil
	}

	var errs []error
	errs = append(errs, c.validateCommon()...)
	errs = append(errs, c.validateVAD()...)

	switch c.Backend {
	case BackendSherpaOnline:
		if len(c.OnlineModels) > 0 {
			errs = append(errs, c.validateOnlineModels()...)
		} else {
			errs = append(errs, c.validateOnline()...)
		}
	case BackendSherpaOffline:
		errs = append(errs, c.validateOffline()...)
	default:
		errs = append(errs, fmt.Errorf("unsupported ASR backend %q", c.Backend))
	}

	if len(errs) > 0 {
		return fmt.Errorf("ASR config validation failed: %v", errs)
	}
	return nil
}

func (c *Config) validateCommon() []error {
	var errs []error
	if c.DefaultModel == "" {
		errs = append(errs, fmt.Errorf("default model cannot be empty"))
	}
	if c.Language == "" {
		errs = append(errs, fmt.Errorf("language cannot be empty"))
	}
	if c.MaxConcurrentStreams <= 0 {
		errs = append(errs, fmt.Errorf("max concurrent streams must be positive, got %d", c.MaxConcurrentStreams))
	}
	if c.StreamTimeout <= 0 {
		errs = append(errs, fmt.Errorf("stream timeout must be positive, got %d", c.StreamTimeout))
	}
	if c.ChunkSize <= 0 {
		errs = append(errs, fmt.Errorf("chunk size must be positive, got %d", c.ChunkSize))
	}
	if c.SampleRate <= 0 {
		errs = append(errs, fmt.Errorf("sample rate must be positive, got %d", c.SampleRate))
	}
	if c.FeatureDim <= 0 {
		errs = append(errs, fmt.Errorf("feature dim must be positive, got %d", c.FeatureDim))
	}
	if c.NumThreads <= 0 {
		errs = append(errs, fmt.Errorf("num threads must be positive, got %d", c.NumThreads))
	}
	return errs
}

func (c *Config) validateVAD() []error {
	var errs []error
	c.VADProvider = normalizeVADProvider(c.VADProvider)
	switch c.VADProvider {
	case VADProviderNone, VADProviderEnergy:
	case VADProviderSilero, VADProviderTen:
		if c.VADModelPath == "" {
			errs = append(errs, fmt.Errorf("VAD model path is required for provider %s", c.VADProvider))
		} else if err := requireFile(c.VADModelPath); err != nil {
			errs = append(errs, fmt.Errorf("VAD model path: %w", err))
		}
	default:
		errs = append(errs, fmt.Errorf("unsupported VAD provider %q", c.VADProvider))
	}
	return errs
}

func (c *Config) validateOnline() []error {
	return validateOnlineConfigPaths("online", c.Online)
}

// validateOnlineModels validates every Config.OnlineModels entry: each must
// have a distinct non-empty Name and a valid path set (same shape required of
// the legacy single Online config).
func (c *Config) validateOnlineModels() []error {
	var errs []error
	seen := make(map[string]struct{}, len(c.OnlineModels))
	for i, oc := range c.OnlineModels {
		label := fmt.Sprintf("online model[%d]", i)
		if oc.Name == "" {
			errs = append(errs, fmt.Errorf("%s: name is required", label))
		} else {
			if _, dup := seen[oc.Name]; dup {
				errs = append(errs, fmt.Errorf("%s: duplicate model name %q", label, oc.Name))
			}
			seen[oc.Name] = struct{}{}
			label = fmt.Sprintf("online model %q", oc.Name)
		}
		errs = append(errs, validateOnlineConfigPaths(label, oc)...)
	}
	return errs
}

// validateOnlineConfigPaths validates the model-path shape shared by the
// legacy single Online config and every OnlineModels entry.
func validateOnlineConfigPaths(label string, online OnlineConfig) []error {
	var errs []error
	if online.TokensPath == "" {
		errs = append(errs, fmt.Errorf("%s tokens path is required", label))
	} else if err := requireFile(online.TokensPath); err != nil {
		errs = append(errs, fmt.Errorf("%s tokens path: %w", label, err))
	}

	hasTransducer := online.EncoderPath != "" || online.DecoderPath != "" || online.JoinerPath != ""
	switch {
	case hasTransducer:
		errs = append(errs, requirePathSet(label+" transducer", map[string]string{
			"encoder": online.EncoderPath,
			"decoder": online.DecoderPath,
			"joiner":  online.JoinerPath,
		})...)
	case online.ModelPath != "":
		if err := requireFile(online.ModelPath); err != nil {
			errs = append(errs, fmt.Errorf("%s model path: %w", label, err))
		}
	default:
		errs = append(errs, fmt.Errorf("%s Sherpa ASR requires transducer encoder/decoder/joiner paths or a single online model path", label))
	}
	return errs
}

// resolvedOnlineModelConfigs returns the per-model OnlineConfig entries
// NewService should build for backend BackendSherpaOnline: one entry per
// OnlineModels item when set, or a single entry expanded from the legacy
// Online field (with Name/Language defaulted from DefaultModel/Language)
// otherwise. Callers must call ApplyDefaults first.
func (c *Config) resolvedOnlineModelConfigs() []OnlineConfig {
	if len(c.OnlineModels) > 0 {
		return c.OnlineModels
	}
	oc := c.Online
	if oc.Name == "" {
		oc.Name = c.DefaultModel
	}
	if oc.Language == "" {
		oc.Language = c.Language
	}
	return []OnlineConfig{oc}
}

func (c *Config) validateOffline() []error {
	var errs []error
	offline := c.Offline
	if offline.ModelFamily == "" {
		return append(errs, fmt.Errorf("offline model family is required"))
	}

	errs = append(errs, validateOfflineFamilyPaths(offline)...)
	if offline.MaxConcurrentDecodes < 0 {
		errs = append(errs, fmt.Errorf("offline max concurrent decodes must not be negative, got %d", offline.MaxConcurrentDecodes))
	}
	if offline.MaxDecodeSeconds < 0 {
		errs = append(errs, fmt.Errorf("offline max decode seconds must not be negative, got %g", offline.MaxDecodeSeconds))
	}
	if err := offline.ChunkOptions().Validate(); err != nil {
		errs = append(errs, fmt.Errorf("offline chunking: %w", err))
	}
	return errs
}

// validateOfflineFamilyPaths checks the path set the offline family needs.
// Sherpa checks tokens for every family except funasr_nano and qwen3_asr (it
// refuses the config before loading any model), so tokens are required for all
// others, including sense_voice and whisper.
func validateOfflineFamilyPaths(offline OfflineConfig) []error {
	label := "offline " + offline.ModelFamily
	switch offline.ModelFamily {
	case OfflineFamilyTransducer:
		return requirePathSet("offline transducer", map[string]string{
			"tokens":  offline.TokensPath,
			"encoder": offline.EncoderPath,
			"decoder": offline.DecoderPath,
			"joiner":  offline.JoinerPath,
		})
	case OfflineFamilyParaformer, OfflineFamilyZipformerCTC, OfflineFamilyNemoCTC,
		OfflineFamilyWenetCTC, OfflineFamilyOmnilingualCTC, OfflineFamilyMedASRCTC,
		OfflineFamilyFireRedASRCTC, OfflineFamilyDolphin, OfflineFamilyTDNN:
		return requirePathSet(label, map[string]string{
			"tokens": offline.TokensPath,
			"model":  offline.ModelPath,
		})
	case OfflineFamilySenseVoice:
		return requirePathSet("offline sense voice", map[string]string{
			"tokens": offline.TokensPath,
			"model":  offline.ModelPath,
		})
	case OfflineFamilyWhisper, OfflineFamilyCanary, OfflineFamilyCohereTranscribe, OfflineFamilyFireRedASR:
		return requirePathSet(label, map[string]string{
			"tokens":  offline.TokensPath,
			"encoder": offline.EncoderPath,
			"decoder": offline.DecoderPath,
		})
	case OfflineFamilyMoonshine:
		return validateMoonshinePaths(offline)
	case OfflineFamilyQwen3ASR:
		return append(requirePathSet(label, map[string]string{
			"conv frontend": offline.ConvFrontendPath,
			"encoder":       offline.EncoderPath,
			"decoder":       offline.DecoderPath,
		}), requireDirPath(label, "tokenizer", offline.TokenizerPath)...)
	case OfflineFamilyFunASRNano:
		return append(requirePathSet(label, map[string]string{
			"encoder adaptor": offline.EncoderAdaptorPath,
			"llm":             offline.LLMPath,
			"embedding":       offline.EmbeddingPath,
		}), requireDirPath(label, "tokenizer", offline.TokenizerPath)...)
	default:
		return []error{fmt.Errorf("unsupported offline model family %q", offline.ModelFamily)}
	}
}

// validateMoonshinePaths accepts either Moonshine layout: v1 (preprocessor,
// encoder, uncached and cached decoder) or v2 (encoder and merged decoder).
// Every path that is set must exist, so a stale path next to a complete set is
// still reported.
func validateMoonshinePaths(offline OfflineConfig) []error {
	const label = "offline moonshine"
	hasV1 := offline.PreprocessorPath != "" && offline.EncoderPath != "" &&
		offline.UncachedDecoderPath != "" && offline.CachedDecoderPath != ""
	hasV2 := offline.EncoderPath != "" && offline.MergedDecoderPath != ""

	paths := map[string]string{"tokens": offline.TokensPath}
	for _, field := range []struct{ name, path string }{
		{"preprocessor", offline.PreprocessorPath},
		{"encoder", offline.EncoderPath},
		{"uncached decoder", offline.UncachedDecoderPath},
		{"cached decoder", offline.CachedDecoderPath},
		{"merged decoder", offline.MergedDecoderPath},
	} {
		if field.path != "" {
			paths[field.name] = field.path
		}
	}
	errs := requirePathSet(label, paths)
	if !hasV1 && !hasV2 {
		errs = append(errs, fmt.Errorf("%s requires either the v1 paths (preprocessor, encoder, uncached decoder, cached decoder) or the v2 paths (encoder, merged decoder)", label))
	}
	return errs
}

// requireDirPath requires a non-empty path that names an existing directory.
func requireDirPath(label, name, path string) []error {
	if path == "" {
		return []error{fmt.Errorf("%s %s path is required", label, name)}
	}
	if err := requireDir(path); err != nil {
		return []error{fmt.Errorf("%s %s path: %w", label, name, err)}
	}
	return nil
}

func requirePathSet(label string, paths map[string]string) []error {
	errs := make([]error, 0, len(paths))
	// Sorted, so the joined validation error reads the same every time.
	for _, name := range slices.Sorted(maps.Keys(paths)) {
		path := paths[name]
		if path == "" {
			errs = append(errs, fmt.Errorf("%s %s path is required", label, name))
			continue
		}
		if err := requireFile(path); err != nil {
			errs = append(errs, fmt.Errorf("%s %s path: %w", label, name, err))
		}
	}
	return errs
}

// requireDir is requireFile's counterpart for model assets that Sherpa reads
// as a directory, such as the funasr_nano and qwen3_asr tokenizers.
func requireDir(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is a file, expected directory", path)
	}
	return nil
}

func requireFile(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.IsDir() {
		return fmt.Errorf("%s is a directory, expected file", path)
	}
	return nil
}
