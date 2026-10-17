package asr

import (
	"strings"
	"testing"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// These tests prove that every supported ASR model family maps its configuration
// onto the correct Sherpa native config fields, and that the required-path
// validation matrix is enforced per family. They are hermetic: no model weights
// and no native recognizer construction are involved, so they run everywhere.

func offlineBaseConfig() Config {
	c := DefaultConfig()
	c.Enabled = true
	c.Backend = BackendSherpaOffline
	c.SampleRate = 16000
	c.FeatureDim = 80
	c.NumThreads = 2
	c.Provider = "cpu"
	c.DecodingMethod = "greedy_search"
	c.MaxActivePaths = 4
	return c
}

func assertCommonOffline(t *testing.T, got *sherpa.OfflineRecognizerConfig, c *Config) {
	t.Helper()
	if got.FeatConfig.SampleRate != c.SampleRate {
		t.Errorf("FeatConfig.SampleRate = %d, want %d", got.FeatConfig.SampleRate, c.SampleRate)
	}
	if got.FeatConfig.FeatureDim != c.FeatureDim {
		t.Errorf("FeatConfig.FeatureDim = %d, want %d", got.FeatConfig.FeatureDim, c.FeatureDim)
	}
	if got.ModelConfig.NumThreads != c.NumThreads {
		t.Errorf("ModelConfig.NumThreads = %d, want %d", got.ModelConfig.NumThreads, c.NumThreads)
	}
	if got.ModelConfig.Provider != c.Provider {
		t.Errorf("ModelConfig.Provider = %q, want %q", got.ModelConfig.Provider, c.Provider)
	}
	if got.DecodingMethod != c.DecodingMethod {
		t.Errorf("DecodingMethod = %q, want %q", got.DecodingMethod, c.DecodingMethod)
	}
	if got.MaxActivePaths != c.MaxActivePaths {
		t.Errorf("MaxActivePaths = %d, want %d", got.MaxActivePaths, c.MaxActivePaths)
	}
}

func TestBuildOfflineRecognizerConfigMapsTransducer(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{
		ModelFamily: OfflineFamilyTransducer,
		TokensPath:  "tok", EncoderPath: "enc", DecoderPath: "dec", JoinerPath: "join",
	}
	got, err := buildOfflineRecognizerConfig(&c)
	if err != nil {
		t.Fatalf("buildOfflineRecognizerConfig: %v", err)
	}
	tr := got.ModelConfig.Transducer
	if tr.Encoder != "enc" || tr.Decoder != "dec" || tr.Joiner != "join" {
		t.Errorf("transducer paths not mapped: %+v", tr)
	}
	if got.ModelConfig.Tokens != "tok" {
		t.Errorf("Tokens = %q, want tok", got.ModelConfig.Tokens)
	}
	if got.ModelConfig.SenseVoice.Model != "" || got.ModelConfig.Paraformer.Model != "" || got.ModelConfig.Whisper.Encoder != "" {
		t.Error("unexpected cross-family field population for transducer")
	}
	assertCommonOffline(t, got, &c)
}

func TestBuildOfflineRecognizerConfigMapsParaformer(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{ModelFamily: OfflineFamilyParaformer, TokensPath: "tok", ModelPath: "model"}
	got, err := buildOfflineRecognizerConfig(&c)
	if err != nil {
		t.Fatalf("buildOfflineRecognizerConfig: %v", err)
	}
	if got.ModelConfig.Paraformer.Model != "model" {
		t.Errorf("Paraformer.Model = %q, want model", got.ModelConfig.Paraformer.Model)
	}
	if got.ModelConfig.Transducer.Encoder != "" || got.ModelConfig.ZipformerCtc.Model != "" {
		t.Error("unexpected cross-family field population for paraformer")
	}
	assertCommonOffline(t, got, &c)
}

func TestBuildOfflineRecognizerConfigMapsZipformerCTC(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{ModelFamily: OfflineFamilyZipformerCTC, TokensPath: "tok", ModelPath: "model"}
	got, err := buildOfflineRecognizerConfig(&c)
	if err != nil {
		t.Fatalf("buildOfflineRecognizerConfig: %v", err)
	}
	if got.ModelConfig.ZipformerCtc.Model != "model" {
		t.Errorf("ZipformerCtc.Model = %q, want model", got.ModelConfig.ZipformerCtc.Model)
	}
	if got.ModelConfig.Paraformer.Model != "" || got.ModelConfig.NemoCTC.Model != "" {
		t.Error("unexpected cross-family field population for zipformer_ctc")
	}
	assertCommonOffline(t, got, &c)
}

func TestBuildOfflineRecognizerConfigMapsNemoCTC(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{ModelFamily: OfflineFamilyNemoCTC, TokensPath: "tok", ModelPath: "model"}
	got, err := buildOfflineRecognizerConfig(&c)
	if err != nil {
		t.Fatalf("buildOfflineRecognizerConfig: %v", err)
	}
	if got.ModelConfig.NemoCTC.Model != "model" {
		t.Errorf("NemoCTC.Model = %q, want model", got.ModelConfig.NemoCTC.Model)
	}
	if got.ModelConfig.ZipformerCtc.Model != "" || got.ModelConfig.SenseVoice.Model != "" {
		t.Error("unexpected cross-family field population for nemo_ctc")
	}
	assertCommonOffline(t, got, &c)
}

func TestBuildOfflineRecognizerConfigMapsSenseVoice(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{
		ModelFamily: OfflineFamilySenseVoice, ModelPath: "model",
		Language: "zh", UseInverseTextNormalization: true,
	}
	got, err := buildOfflineRecognizerConfig(&c)
	if err != nil {
		t.Fatalf("buildOfflineRecognizerConfig: %v", err)
	}
	sv := got.ModelConfig.SenseVoice
	if sv.Model != "model" || sv.Language != "zh" || sv.UseInverseTextNormalization != 1 {
		t.Errorf("sense_voice not mapped: %+v", sv)
	}
	if got.ModelConfig.Whisper.Encoder != "" || got.ModelConfig.Transducer.Encoder != "" {
		t.Error("unexpected cross-family field population for sense_voice")
	}
	assertCommonOffline(t, got, &c)
}

// Whisper offline mapping is covered by TestBuildOfflineRecognizerConfigMapsWhisper
// in service_test.go; the families below are the previously-untested ones.

func TestBuildOfflineRecognizerConfigRejectsUnknownFamily(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{ModelFamily: "does_not_exist", ModelPath: "model"}
	if _, err := buildOfflineRecognizerConfig(&c); err == nil {
		t.Fatal("expected error for unknown offline family, got nil")
	}
}

func onlineBaseConfig() Config {
	c := DefaultConfig()
	c.Enabled = true
	c.Backend = BackendSherpaOnline
	c.SampleRate = 16000
	c.FeatureDim = 80
	c.NumThreads = 2
	c.Provider = "cpu"
	return c
}

func TestBuildOnlineRecognizerConfigMapsTransducer(t *testing.T) {
	c := onlineBaseConfig()
	c.Online = OnlineConfig{TokensPath: "tok", EncoderPath: "enc", DecoderPath: "dec", JoinerPath: "join"}
	got, err := buildOnlineRecognizerConfig(&c)
	if err != nil {
		t.Fatalf("buildOnlineRecognizerConfig: %v", err)
	}
	tr := got.ModelConfig.Transducer
	if tr.Encoder != "enc" || tr.Decoder != "dec" || tr.Joiner != "join" {
		t.Errorf("online transducer paths not mapped: %+v", tr)
	}
	if got.ModelConfig.Zipformer2Ctc.Model != "" || got.ModelConfig.NemoCtc.Model != "" {
		t.Error("unexpected CTC field population for online transducer")
	}
	if got.FeatConfig.SampleRate != c.SampleRate || got.ModelConfig.NumThreads != c.NumThreads {
		t.Error("online common fields not mapped")
	}
}

func TestBuildOnlineRecognizerConfigMapsSingleFileCTC(t *testing.T) {
	cases := []struct {
		modelType string
		field     func(*sherpa.OnlineModelConfig) string
		other     func(*sherpa.OnlineModelConfig) bool
	}{
		{"zipformer2_ctc", func(m *sherpa.OnlineModelConfig) string { return m.Zipformer2Ctc.Model }, func(m *sherpa.OnlineModelConfig) bool { return m.NemoCtc.Model != "" || m.ToneCtc.Model != "" }},
		{"nemo_ctc", func(m *sherpa.OnlineModelConfig) string { return m.NemoCtc.Model }, func(m *sherpa.OnlineModelConfig) bool { return m.Zipformer2Ctc.Model != "" || m.ToneCtc.Model != "" }},
		{"tone_ctc", func(m *sherpa.OnlineModelConfig) string { return m.ToneCtc.Model }, func(m *sherpa.OnlineModelConfig) bool { return m.Zipformer2Ctc.Model != "" || m.NemoCtc.Model != "" }},
	}
	for _, tc := range cases {
		t.Run(tc.modelType, func(t *testing.T) {
			c := onlineBaseConfig()
			c.Online = OnlineConfig{TokensPath: "tok", ModelPath: "model", ModelType: tc.modelType}
			got, err := buildOnlineRecognizerConfig(&c)
			if err != nil {
				t.Fatalf("buildOnlineRecognizerConfig: %v", err)
			}
			if tc.field(&got.ModelConfig) != "model" {
				t.Errorf("%s model path not mapped", tc.modelType)
			}
			if tc.other(&got.ModelConfig) {
				t.Errorf("%s populated a sibling CTC field", tc.modelType)
			}
			if got.ModelConfig.Transducer.Encoder != "" {
				t.Errorf("%s populated the transducer field", tc.modelType)
			}
		})
	}
}

func TestBuildOnlineRecognizerConfigRejectsPartialTransducer(t *testing.T) {
	c := onlineBaseConfig()
	c.Online = OnlineConfig{TokensPath: "tok", EncoderPath: "enc"} // decoder+joiner missing
	if _, err := buildOnlineRecognizerConfig(&c); err == nil {
		t.Fatal("expected error for partial online transducer paths, got nil")
	}
}

func TestBuildOnlineRecognizerConfigRejectsUnknownModelType(t *testing.T) {
	c := onlineBaseConfig()
	c.Online = OnlineConfig{TokensPath: "tok", ModelPath: "model", ModelType: "not_a_real_type"}
	if _, err := buildOnlineRecognizerConfig(&c); err == nil {
		t.Fatal("expected error for unknown online model type, got nil")
	}
}

// TestValidateOfflineRequiredPathMatrix asserts that each offline family reports
// its required-but-missing paths. Uses empty paths so no filesystem access occurs.
func TestValidateOfflineRequiredPathMatrix(t *testing.T) {
	cases := []struct {
		family   string
		wantSubs []string
	}{
		{OfflineFamilyTransducer, []string{"tokens", "encoder", "decoder", "joiner"}},
		{OfflineFamilyParaformer, []string{"tokens", "model"}},
		{OfflineFamilyZipformerCTC, []string{"tokens", "model"}},
		{OfflineFamilyNemoCTC, []string{"tokens", "model"}},
		{OfflineFamilySenseVoice, []string{"tokens", "model"}},
		{OfflineFamilyWhisper, []string{"tokens", "encoder", "decoder"}},
		{OfflineFamilyFunASRNano, []string{"encoder adaptor", "llm", "embedding", "tokenizer"}},
		{OfflineFamilyMoonshine, []string{"tokens", "requires either the v1 paths"}},
		{OfflineFamilyQwen3ASR, []string{"conv frontend", "encoder", "decoder", "tokenizer"}},
		{OfflineFamilyCanary, []string{"tokens", "encoder", "decoder"}},
		{OfflineFamilyCohereTranscribe, []string{"tokens", "encoder", "decoder"}},
		{OfflineFamilyWenetCTC, []string{"tokens", "model"}},
		{OfflineFamilyOmnilingualCTC, []string{"tokens", "model"}},
		{OfflineFamilyMedASRCTC, []string{"tokens", "model"}},
		{OfflineFamilyFireRedASRCTC, []string{"tokens", "model"}},
		{OfflineFamilyDolphin, []string{"tokens", "model"}},
		{OfflineFamilyTDNN, []string{"tokens", "model"}},
		{OfflineFamilyFireRedASR, []string{"tokens", "encoder", "decoder"}},
	}
	for _, tc := range cases {
		t.Run(tc.family, func(t *testing.T) {
			c := offlineBaseConfig()
			c.Offline = OfflineConfig{ModelFamily: tc.family}
			err := c.Validate()
			if err == nil {
				t.Fatalf("expected validation errors for %s with no paths", tc.family)
			}
			msg := err.Error()
			for _, sub := range tc.wantSubs {
				if !strings.Contains(msg, sub) {
					t.Errorf("validation error for %s missing mention of %q; got: %s", tc.family, sub, msg)
				}
			}
		})
	}
}

// TestValidateOnlineRequiresTokensAndModel asserts the online backend rejects a
// config with neither transducer paths nor a single model path.
func TestValidateOnlineRequiresModel(t *testing.T) {
	c := onlineBaseConfig()
	c.Online = OnlineConfig{} // nothing set
	err := c.Validate()
	if err == nil {
		t.Fatal("expected validation error for empty online config")
	}
	if !strings.Contains(err.Error(), "tokens") {
		t.Errorf("expected online tokens requirement; got: %s", err.Error())
	}
}

// TestValidateVADAcceptsProviderAliases proves that validateVAD (the
// Config.Validate() path) normalizes short/aliased VAD provider names
// case-insensitively to their canonical constants. It uses an empty
// VADModelPath so the assertion is on the "VAD model path is required for
// provider <canonical>" error (which only fires once the provider has been
// recognized as a sherpa provider) rather than on the "unsupported VAD
// provider" default-case error an unrecognized alias would produce; this
// keeps the test hermetic (no native sherpa model construction).
func TestValidateVADAcceptsProviderAliases(t *testing.T) {
	cases := []struct {
		alias string
		want  string
	}{
		{"silero", VADProviderSilero},
		{"SILERO", VADProviderSilero},
		{"ten", VADProviderTen},
		{"TEN", VADProviderTen},
	}
	for _, tc := range cases {
		t.Run(tc.alias, func(t *testing.T) {
			c := offlineBaseConfig()
			c.Offline = OfflineConfig{ModelFamily: OfflineFamilySenseVoice, ModelPath: "model"}
			c.VADProvider = tc.alias
			c.VADModelPath = ""
			err := c.Validate()
			if err == nil {
				t.Fatalf("expected validation error for missing VAD model path with alias %q", tc.alias)
			}
			if strings.Contains(err.Error(), "unsupported VAD provider") {
				t.Fatalf("alias %q was not normalized before dispatch: %v", tc.alias, err)
			}
			if !strings.Contains(err.Error(), "VAD model path is required for provider "+tc.want) {
				t.Errorf("expected normalized provider %q in error; got: %s", tc.want, err.Error())
			}
			if c.VADProvider != tc.want {
				t.Errorf("VADProvider after validate = %q, want normalized %q", c.VADProvider, tc.want)
			}
		})
	}
}

// TestValidateVADRejectsUnknownProvider proves validateVAD still hard-errors
// on unsupported VAD providers after alias normalization.
func TestValidateVADRejectsUnknownProvider(t *testing.T) {
	c := offlineBaseConfig()
	c.Offline = OfflineConfig{ModelFamily: OfflineFamilySenseVoice, ModelPath: "model"}
	c.VADProvider = "bogus"
	err := c.Validate()
	if err == nil {
		t.Fatal("expected validation error for unsupported VAD provider")
	}
	if !strings.Contains(err.Error(), `unsupported VAD provider "bogus"`) {
		t.Errorf("expected unsupported-provider error mentioning %q; got: %s", "bogus", err.Error())
	}
}

// TestNewVADServiceAcceptsProviderAliases proves the exported NewVADService
// constructor (a public bypass of Config.Validate()) also normalizes aliases
// case-insensitively before dispatch. It uses an empty ModelPath so the
// assertion is on newSherpaVAD's "VAD model path is required for provider
// <canonical>" error rather than on native sherpa VAD construction, keeping
// the test hermetic.
func TestNewVADServiceAcceptsProviderAliases(t *testing.T) {
	cases := []struct {
		alias string
		want  string
	}{
		{"silero", VADProviderSilero},
		{"SILERO", VADProviderSilero},
		{"ten", VADProviderTen},
		{"TEN", VADProviderTen},
	}
	for _, tc := range cases {
		t.Run(tc.alias, func(t *testing.T) {
			cfg := &VADConfig{Provider: tc.alias}
			_, err := NewVADService(cfg)
			if err == nil {
				t.Fatalf("expected error for alias %q with no VAD model path", tc.alias)
			}
			if strings.Contains(err.Error(), "unsupported VAD provider") {
				t.Fatalf("alias %q was not normalized before dispatch: %v", tc.alias, err)
			}
			if !strings.Contains(err.Error(), "VAD model path is required for provider "+tc.want) {
				t.Errorf("expected normalized provider %q in error; got: %s", tc.want, err.Error())
			}
			if cfg.Provider != tc.want {
				t.Errorf("Provider after NewVADService = %q, want normalized %q", cfg.Provider, tc.want)
			}
		})
	}
}

// TestNewVADServiceRejectsUnknownProvider proves NewVADService still
// hard-errors on unsupported VAD providers after alias normalization.
func TestNewVADServiceRejectsUnknownProvider(t *testing.T) {
	cfg := &VADConfig{Provider: "bogus"}
	_, err := NewVADService(cfg)
	if err == nil {
		t.Fatal("expected error for unsupported VAD provider")
	}
	if !strings.Contains(err.Error(), `unsupported VAD provider "bogus"`) {
		t.Errorf("expected unsupported-provider error mentioning %q; got: %s", "bogus", err.Error())
	}
}
