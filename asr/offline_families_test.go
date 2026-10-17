package asr

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

// These tests cover the offline families added after the original six
// (transducer, paraformer, zipformer_ctc, nemo_ctc, sense_voice, whisper):
// the exact Sherpa fields each one sets, and the required-path matrix. They
// are hermetic: files are empty placeholders and no native recognizer is built.

// familyFiles creates placeholder model files under one temp directory.
type familyFiles struct {
	t   *testing.T
	dir string
}

func newFamilyFiles(t *testing.T) *familyFiles {
	t.Helper()
	return &familyFiles{t: t, dir: t.TempDir()}
}

func (f *familyFiles) file(name string) string {
	f.t.Helper()
	path := filepath.Join(f.dir, name)
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		f.t.Fatalf("write %s: %v", name, err)
	}
	return path
}

func (f *familyFiles) subdir(name string) string {
	f.t.Helper()
	path := filepath.Join(f.dir, name)
	if err := os.Mkdir(path, 0o755); err != nil {
		f.t.Fatalf("mkdir %s: %v", name, err)
	}
	return path
}

// expectedOfflineConfig is the sherpa config buildOfflineRecognizerConfig must
// produce for base plus the given family model config. Comparing whole structs
// also proves no sibling family field is populated.
func expectedOfflineConfig(c *Config, model sherpa.OfflineModelConfig) *sherpa.OfflineRecognizerConfig {
	model.NumThreads = c.NumThreads
	model.Provider = c.Provider
	return &sherpa.OfflineRecognizerConfig{
		FeatConfig:     sherpa.FeatureConfig{SampleRate: c.SampleRate, FeatureDim: c.FeatureDim},
		ModelConfig:    model,
		DecodingMethod: c.DecodingMethod,
		MaxActivePaths: c.MaxActivePaths,
	}
}

func TestBuildOfflineRecognizerConfigExtendedFamilies(t *testing.T) {
	cases := []struct {
		name    string
		offline OfflineConfig
		want    sherpa.OfflineModelConfig
	}{
		{
			name: OfflineFamilyFunASRNano,
			offline: OfflineConfig{
				ModelFamily:                 OfflineFamilyFunASRNano,
				EncoderAdaptorPath:          "adaptor.onnx",
				LLMPath:                     "llm.onnx",
				EmbeddingPath:               "embedding.onnx",
				TokenizerPath:               "tokenizer-dir",
				SystemPrompt:                "system",
				UserPrompt:                  "user",
				MaxNewTokens:                256,
				Temperature:                 0.3,
				TopP:                        0.8,
				Seed:                        42,
				Language:                    "zh",
				UseInverseTextNormalization: true,
				Hotwords:                    "alpha,beta",
				// Ignored fields: funasr_nano never reads tokens.txt, and
				// MaxTotalLen belongs to qwen3_asr.
				TokensPath:  "tokens.txt",
				MaxTotalLen: 999,
			},
			want: sherpa.OfflineModelConfig{
				FunAsrNano: sherpa.OfflineFunASRNanoModelConfig{
					EncoderAdaptor:              "adaptor.onnx",
					LLM:                         "llm.onnx",
					Embedding:                   "embedding.onnx",
					Tokenizer:                   "tokenizer-dir",
					SystemPrompt:                "system",
					UserPrompt:                  "user",
					MaxNewTokens:                256,
					Temperature:                 0.3,
					TopP:                        0.8,
					Seed:                        42,
					Language:                    "zh",
					UseInverseTextNormalization: 1,
					Hotwords:                    "alpha,beta",
				},
			},
		},
		{
			name: "moonshine v1",
			offline: OfflineConfig{
				ModelFamily:         OfflineFamilyMoonshine,
				TokensPath:          "tokens.txt",
				PreprocessorPath:    "pre.onnx",
				EncoderPath:         "enc.onnx",
				UncachedDecoderPath: "uncached.onnx",
				CachedDecoderPath:   "cached.onnx",
			},
			want: sherpa.OfflineModelConfig{
				Tokens: "tokens.txt",
				Moonshine: sherpa.OfflineMoonshineModelConfig{
					Preprocessor:    "pre.onnx",
					Encoder:         "enc.onnx",
					UncachedDecoder: "uncached.onnx",
					CachedDecoder:   "cached.onnx",
				},
			},
		},
		{
			name: "moonshine v2",
			offline: OfflineConfig{
				ModelFamily:       OfflineFamilyMoonshine,
				TokensPath:        "tokens.txt",
				EncoderPath:       "enc.onnx",
				MergedDecoderPath: "merged.onnx",
			},
			want: sherpa.OfflineModelConfig{
				Tokens: "tokens.txt",
				Moonshine: sherpa.OfflineMoonshineModelConfig{
					Encoder:       "enc.onnx",
					MergedDecoder: "merged.onnx",
				},
			},
		},
		{
			name: OfflineFamilyQwen3ASR,
			offline: OfflineConfig{
				ModelFamily:      OfflineFamilyQwen3ASR,
				ConvFrontendPath: "conv.onnx",
				EncoderPath:      "enc.onnx",
				DecoderPath:      "dec.onnx",
				TokenizerPath:    "tokenizer-dir",
				MaxTotalLen:      2048,
				MaxNewTokens:     512,
				Temperature:      0.1,
				TopP:             0.9,
				Seed:             7,
				Hotwords:         "gamma",
				// Ignored fields: qwen3_asr takes no language, ITN or prompts
				// and never reads tokens.txt.
				TokensPath:                  "tokens.txt",
				Language:                    "en",
				UseInverseTextNormalization: true,
				SystemPrompt:                "ignored",
			},
			want: sherpa.OfflineModelConfig{
				Qwen3ASR: sherpa.OfflineQwen3ASRModelConfig{
					ConvFrontend: "conv.onnx",
					Encoder:      "enc.onnx",
					Decoder:      "dec.onnx",
					Tokenizer:    "tokenizer-dir",
					MaxTotalLen:  2048,
					MaxNewTokens: 512,
					Temperature:  0.1,
					TopP:         0.9,
					Seed:         7,
					Hotwords:     "gamma",
				},
			},
		},
		{
			name: OfflineFamilyCanary,
			offline: OfflineConfig{
				ModelFamily:    OfflineFamilyCanary,
				TokensPath:     "tokens.txt",
				EncoderPath:    "enc.onnx",
				DecoderPath:    "dec.onnx",
				SrcLang:        "en",
				TgtLang:        "de",
				UsePunctuation: true,
				// Canary has its own language pair; Language is not mapped.
				Language: "fr",
			},
			want: sherpa.OfflineModelConfig{
				Tokens: "tokens.txt",
				Canary: sherpa.OfflineCanaryModelConfig{
					Encoder: "enc.onnx",
					Decoder: "dec.onnx",
					SrcLang: "en",
					TgtLang: "de",
					UsePnc:  1,
				},
			},
		},
		{
			name: OfflineFamilyCohereTranscribe,
			offline: OfflineConfig{
				ModelFamily:                 OfflineFamilyCohereTranscribe,
				TokensPath:                  "tokens.txt",
				EncoderPath:                 "enc.onnx",
				DecoderPath:                 "dec.onnx",
				Language:                    "de",
				UsePunctuation:              true,
				UseInverseTextNormalization: true,
			},
			want: sherpa.OfflineModelConfig{
				Tokens: "tokens.txt",
				CohereTranscribe: sherpa.OfflineCohereTranscribeModelConfig{
					Encoder:                     "enc.onnx",
					Decoder:                     "dec.onnx",
					Language:                    "de",
					UsePunct:                    1,
					UseInverseTextNormalization: 1,
				},
			},
		},
		{
			name:    OfflineFamilyWenetCTC,
			offline: OfflineConfig{ModelFamily: OfflineFamilyWenetCTC, TokensPath: "tokens.txt", ModelPath: "m.onnx"},
			want: sherpa.OfflineModelConfig{
				Tokens:   "tokens.txt",
				WenetCtc: sherpa.OfflineWenetCtcModelConfig{Model: "m.onnx"},
			},
		},
		{
			name:    OfflineFamilyOmnilingualCTC,
			offline: OfflineConfig{ModelFamily: OfflineFamilyOmnilingualCTC, TokensPath: "tokens.txt", ModelPath: "m.onnx"},
			want: sherpa.OfflineModelConfig{
				Tokens:      "tokens.txt",
				Omnilingual: sherpa.OfflineOmnilingualAsrCtcModelConfig{Model: "m.onnx"},
			},
		},
		{
			name:    OfflineFamilyMedASRCTC,
			offline: OfflineConfig{ModelFamily: OfflineFamilyMedASRCTC, TokensPath: "tokens.txt", ModelPath: "m.onnx"},
			want: sherpa.OfflineModelConfig{
				Tokens: "tokens.txt",
				MedAsr: sherpa.OfflineMedAsrCtcModelConfig{Model: "m.onnx"},
			},
		},
		{
			name:    OfflineFamilyFireRedASRCTC,
			offline: OfflineConfig{ModelFamily: OfflineFamilyFireRedASRCTC, TokensPath: "tokens.txt", ModelPath: "m.onnx"},
			want: sherpa.OfflineModelConfig{
				Tokens:        "tokens.txt",
				FireRedAsrCtc: sherpa.OfflineFireRedAsrCtcModelConfig{Model: "m.onnx"},
			},
		},
		{
			name:    OfflineFamilyDolphin,
			offline: OfflineConfig{ModelFamily: OfflineFamilyDolphin, TokensPath: "tokens.txt", ModelPath: "m.onnx"},
			want: sherpa.OfflineModelConfig{
				Tokens:  "tokens.txt",
				Dolphin: sherpa.OfflineDolphinModelConfig{Model: "m.onnx"},
			},
		},
		{
			name:    OfflineFamilyTDNN,
			offline: OfflineConfig{ModelFamily: OfflineFamilyTDNN, TokensPath: "tokens.txt", ModelPath: "m.onnx"},
			want: sherpa.OfflineModelConfig{
				Tokens: "tokens.txt",
				Tdnn:   sherpa.OfflineTdnnModelConfig{Model: "m.onnx"},
			},
		},
		{
			name: OfflineFamilyFireRedASR,
			offline: OfflineConfig{
				ModelFamily: OfflineFamilyFireRedASR,
				TokensPath:  "tokens.txt",
				EncoderPath: "enc.onnx",
				DecoderPath: "dec.onnx",
			},
			want: sherpa.OfflineModelConfig{
				Tokens:     "tokens.txt",
				FireRedAsr: sherpa.OfflineFireRedAsrModelConfig{Encoder: "enc.onnx", Decoder: "dec.onnx"},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := offlineBaseConfig()
			c.Offline = tc.offline
			got, err := buildOfflineRecognizerConfig(&c)
			if err != nil {
				t.Fatalf("buildOfflineRecognizerConfig: %v", err)
			}
			want := expectedOfflineConfig(&c, tc.want)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("config mismatch\n got: %+v\nwant: %+v", got, want)
			}
		})
	}
}

// An explicit Offline.ModelType is passed through for every family; only the
// original six also default it to the family name.
func TestBuildOfflineRecognizerConfigModelTypeHint(t *testing.T) {
	for _, family := range []string{
		OfflineFamilyTransducer, OfflineFamilyParaformer, OfflineFamilyZipformerCTC,
		OfflineFamilyNemoCTC, OfflineFamilySenseVoice, OfflineFamilyWhisper,
	} {
		t.Run("default "+family, func(t *testing.T) {
			c := offlineBaseConfig()
			c.Offline = OfflineConfig{ModelFamily: family}
			got, err := buildOfflineRecognizerConfig(&c)
			if err != nil {
				t.Fatal(err)
			}
			if got.ModelConfig.ModelType != family {
				t.Errorf("ModelType = %q, want %q", got.ModelConfig.ModelType, family)
			}
		})
	}
	for _, family := range extendedOfflineFamilies {
		t.Run("default "+family, func(t *testing.T) {
			c := offlineBaseConfig()
			c.Offline = OfflineConfig{ModelFamily: family}
			got, err := buildOfflineRecognizerConfig(&c)
			if err != nil {
				t.Fatal(err)
			}
			if got.ModelConfig.ModelType != "" {
				t.Errorf("ModelType = %q, want empty so sherpa detects it", got.ModelConfig.ModelType)
			}
		})
		t.Run("explicit "+family, func(t *testing.T) {
			c := offlineBaseConfig()
			c.Offline = OfflineConfig{ModelFamily: family, ModelType: "hint"}
			got, err := buildOfflineRecognizerConfig(&c)
			if err != nil {
				t.Fatal(err)
			}
			if got.ModelConfig.ModelType != "hint" {
				t.Errorf("ModelType = %q, want hint", got.ModelConfig.ModelType)
			}
		})
	}
}

var extendedOfflineFamilies = []string{
	OfflineFamilyFunASRNano, OfflineFamilyMoonshine, OfflineFamilyQwen3ASR,
	OfflineFamilyCanary, OfflineFamilyCohereTranscribe, OfflineFamilyWenetCTC,
	OfflineFamilyOmnilingualCTC, OfflineFamilyMedASRCTC, OfflineFamilyFireRedASRCTC,
	OfflineFamilyDolphin, OfflineFamilyTDNN, OfflineFamilyFireRedASR,
}

func TestOfflineFamilyConstantValues(t *testing.T) {
	want := map[string]string{
		OfflineFamilyTransducer:       "transducer",
		OfflineFamilyParaformer:       "paraformer",
		OfflineFamilyZipformerCTC:     "zipformer_ctc",
		OfflineFamilyNemoCTC:          "nemo_ctc",
		OfflineFamilySenseVoice:       "sense_voice",
		OfflineFamilyWhisper:          "whisper",
		OfflineFamilyFunASRNano:       "funasr_nano",
		OfflineFamilyMoonshine:        "moonshine",
		OfflineFamilyQwen3ASR:         "qwen3_asr",
		OfflineFamilyCanary:           "canary",
		OfflineFamilyCohereTranscribe: "cohere_transcribe",
		OfflineFamilyWenetCTC:         "wenet_ctc",
		OfflineFamilyOmnilingualCTC:   "omnilingual_ctc",
		OfflineFamilyMedASRCTC:        "med_asr_ctc",
		OfflineFamilyFireRedASRCTC:    "firered_asr_ctc",
		OfflineFamilyDolphin:          "dolphin",
		OfflineFamilyTDNN:             "tdnn",
		OfflineFamilyFireRedASR:       "firered_asr",
	}
	for got, literal := range want {
		if got != literal {
			t.Errorf("family constant = %q, want %q", got, literal)
		}
	}
}

// familyValidationCase builds a complete, valid OfflineConfig for one family
// and lists the fields whose removal must be reported by name.
type familyValidationCase struct {
	name     string
	build    func(f *familyFiles) OfflineConfig
	required map[string]func(o *OfflineConfig)
}

func clearTokens(o *OfflineConfig)  { o.TokensPath = "" }
func clearModel(o *OfflineConfig)   { o.ModelPath = "" }
func clearEncoder(o *OfflineConfig) { o.EncoderPath = "" }
func clearDecoder(o *OfflineConfig) { o.DecoderPath = "" }

func singleFileCTCCase(family string) familyValidationCase {
	return familyValidationCase{
		name: family,
		build: func(f *familyFiles) OfflineConfig {
			return OfflineConfig{ModelFamily: family, TokensPath: f.file("tokens.txt"), ModelPath: f.file("model.onnx")}
		},
		required: map[string]func(*OfflineConfig){"tokens": clearTokens, "model": clearModel},
	}
}

func encoderDecoderCase(family string) familyValidationCase {
	return familyValidationCase{
		name: family,
		build: func(f *familyFiles) OfflineConfig {
			return OfflineConfig{
				ModelFamily: family,
				TokensPath:  f.file("tokens.txt"),
				EncoderPath: f.file("encoder.onnx"),
				DecoderPath: f.file("decoder.onnx"),
			}
		},
		required: map[string]func(*OfflineConfig){"tokens": clearTokens, "encoder": clearEncoder, "decoder": clearDecoder},
	}
}

func familyValidationCases() []familyValidationCase {
	return []familyValidationCase{
		{
			name: OfflineFamilyFunASRNano,
			build: func(f *familyFiles) OfflineConfig {
				return OfflineConfig{
					ModelFamily:        OfflineFamilyFunASRNano,
					EncoderAdaptorPath: f.file("adaptor.onnx"),
					LLMPath:            f.file("llm.onnx"),
					EmbeddingPath:      f.file("embedding.onnx"),
					TokenizerPath:      f.subdir("tokenizer"),
				}
			},
			required: map[string]func(*OfflineConfig){
				"encoder adaptor": func(o *OfflineConfig) { o.EncoderAdaptorPath = "" },
				"llm":             func(o *OfflineConfig) { o.LLMPath = "" },
				"embedding":       func(o *OfflineConfig) { o.EmbeddingPath = "" },
				"tokenizer":       func(o *OfflineConfig) { o.TokenizerPath = "" },
			},
		},
		{
			name: OfflineFamilyQwen3ASR,
			build: func(f *familyFiles) OfflineConfig {
				return OfflineConfig{
					ModelFamily:      OfflineFamilyQwen3ASR,
					ConvFrontendPath: f.file("conv.onnx"),
					EncoderPath:      f.file("encoder.onnx"),
					DecoderPath:      f.file("decoder.onnx"),
					TokenizerPath:    f.subdir("tokenizer"),
				}
			},
			required: map[string]func(*OfflineConfig){
				"conv frontend": func(o *OfflineConfig) { o.ConvFrontendPath = "" },
				"encoder":       clearEncoder,
				"decoder":       clearDecoder,
				"tokenizer":     func(o *OfflineConfig) { o.TokenizerPath = "" },
			},
		},
		encoderDecoderCase(OfflineFamilyCanary),
		encoderDecoderCase(OfflineFamilyCohereTranscribe),
		encoderDecoderCase(OfflineFamilyFireRedASR),
		encoderDecoderCase(OfflineFamilyWhisper),
		singleFileCTCCase(OfflineFamilyWenetCTC),
		singleFileCTCCase(OfflineFamilyOmnilingualCTC),
		singleFileCTCCase(OfflineFamilyMedASRCTC),
		singleFileCTCCase(OfflineFamilyFireRedASRCTC),
		singleFileCTCCase(OfflineFamilyDolphin),
		singleFileCTCCase(OfflineFamilyTDNN),
		singleFileCTCCase(OfflineFamilySenseVoice),
	}
}

func offlineValidationConfig(o OfflineConfig) Config {
	c := offlineBaseConfig()
	c.Offline = o
	return c
}

func TestValidateOfflineFamiliesAcceptCompleteConfig(t *testing.T) {
	for _, tc := range familyValidationCases() {
		t.Run(tc.name, func(t *testing.T) {
			c := offlineValidationConfig(tc.build(newFamilyFiles(t)))
			if err := c.Validate(); err != nil {
				t.Fatalf("complete %s config rejected: %v", tc.name, err)
			}
		})
	}
}

func TestValidateOfflineFamiliesReportEachMissingPath(t *testing.T) {
	for _, tc := range familyValidationCases() {
		for field, clear := range tc.required {
			t.Run(tc.name+"/"+field, func(t *testing.T) {
				o := tc.build(newFamilyFiles(t))
				clear(&o)
				c := offlineValidationConfig(o)
				err := c.Validate()
				if err == nil {
					t.Fatalf("%s without %s validated", tc.name, field)
				}
				if want := field + " path is required"; !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not contain %q", err, want)
				}
			})
		}
	}
}

func TestValidateOfflineFamiliesRejectNonexistentFiles(t *testing.T) {
	for _, tc := range familyValidationCases() {
		t.Run(tc.name, func(t *testing.T) {
			f := newFamilyFiles(t)
			o := tc.build(f)
			// Point the first fillable path field at a file that does not exist.
			missing := filepath.Join(f.dir, "absent.onnx")
			switch {
			case o.ModelPath != "":
				o.ModelPath = missing
			case o.EncoderPath != "":
				o.EncoderPath = missing
			default:
				o.EncoderAdaptorPath = missing
			}
			c := offlineValidationConfig(o)
			err := c.Validate()
			if err == nil {
				t.Fatal("expected error for a path that does not exist")
			}
			if !strings.Contains(err.Error(), "absent.onnx") {
				t.Errorf("error %q does not name the missing file", err)
			}
		})
	}
}

// qwen3_asr and funasr_nano read their tokenizer as a directory, so it is
// validated with requireDir: a regular file or a missing path is refused, and
// tokens.txt is not required.
func TestValidateOfflineTokenizerIsDirectory(t *testing.T) {
	for _, tc := range familyValidationCases() {
		if tc.name != OfflineFamilyQwen3ASR && tc.name != OfflineFamilyFunASRNano {
			continue
		}
		t.Run(tc.name+"/file rejected", func(t *testing.T) {
			f := newFamilyFiles(t)
			o := tc.build(f)
			o.TokenizerPath = f.file("tokenizer.json")
			c := offlineValidationConfig(o)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), "expected directory") {
				t.Fatalf("got %v, want an expected-directory error", err)
			}
		})
		t.Run(tc.name+"/missing rejected", func(t *testing.T) {
			f := newFamilyFiles(t)
			o := tc.build(f)
			o.TokenizerPath = filepath.Join(f.dir, "no-such-dir")
			c := offlineValidationConfig(o)
			if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "tokenizer path") {
				t.Fatalf("got %v, want a tokenizer path error", err)
			}
		})
		t.Run(tc.name+"/no tokens needed", func(t *testing.T) {
			o := tc.build(newFamilyFiles(t))
			if o.TokensPath != "" {
				t.Fatal("fixture unexpectedly sets tokens")
			}
			c := offlineValidationConfig(o)
			if err := c.Validate(); err != nil {
				t.Fatalf("%s must validate without tokens: %v", tc.name, err)
			}
		})
	}
}

// Sherpa refuses these families without tokens before it loads any model, so
// validation must too (sense_voice and whisper used to pass without them).
func TestValidateOfflineRequiresTokensForSenseVoiceAndWhisper(t *testing.T) {
	f := newFamilyFiles(t)
	cases := map[string]OfflineConfig{
		OfflineFamilySenseVoice: {ModelFamily: OfflineFamilySenseVoice, ModelPath: f.file("model.onnx")},
		OfflineFamilyWhisper: {
			ModelFamily: OfflineFamilyWhisper,
			EncoderPath: f.file("encoder.onnx"),
			DecoderPath: f.file("decoder.onnx"),
		},
	}
	for family, o := range cases {
		t.Run(family, func(t *testing.T) {
			c := offlineValidationConfig(o)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), "tokens path is required") {
				t.Fatalf("got %v, want a tokens-path-required error", err)
			}
		})
	}
}

func moonshineConfig(f *familyFiles, v1, v2 bool) OfflineConfig {
	o := OfflineConfig{ModelFamily: OfflineFamilyMoonshine, TokensPath: f.file("tokens.txt")}
	if v1 || v2 {
		o.EncoderPath = f.file("encoder.onnx")
	}
	if v1 {
		o.PreprocessorPath = f.file("preprocessor.onnx")
		o.UncachedDecoderPath = f.file("uncached.onnx")
		o.CachedDecoderPath = f.file("cached.onnx")
	}
	if v2 {
		o.MergedDecoderPath = f.file("merged.onnx")
	}
	return o
}

func TestValidateOfflineMoonshineAlternatives(t *testing.T) {
	const either = "requires either the v1 paths"
	tests := []struct {
		name    string
		mutate  func(f *familyFiles, o *OfflineConfig)
		wantErr string // empty means valid
	}{
		{"v1 complete", func(f *familyFiles, o *OfflineConfig) { *o = moonshineConfig(f, true, false) }, ""},
		{"v2 complete", func(f *familyFiles, o *OfflineConfig) { *o = moonshineConfig(f, false, true) }, ""},
		{"both sets complete", func(f *familyFiles, o *OfflineConfig) { *o = moonshineConfig(f, true, true) }, ""},
		{"nothing", func(f *familyFiles, o *OfflineConfig) { *o = moonshineConfig(f, false, false) }, either},
		{"v1 without preprocessor", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, true, false)
			o.PreprocessorPath = ""
		}, either},
		{"v1 without uncached decoder", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, true, false)
			o.UncachedDecoderPath = ""
		}, either},
		{"v1 without cached decoder", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, true, false)
			o.CachedDecoderPath = ""
		}, either},
		{"v1 without encoder", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, true, false)
			o.EncoderPath = ""
		}, either},
		{"v2 without merged decoder", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, false, true)
			o.MergedDecoderPath = ""
		}, either},
		{"v2 without encoder", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, false, true)
			o.EncoderPath = ""
		}, either},
		{"v2 without tokens", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, false, true)
			o.TokensPath = ""
		}, "tokens path is required"},
		{"v1 without tokens", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, true, false)
			o.TokensPath = ""
		}, "tokens path is required"},
		{"v2 merged decoder missing on disk", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, false, true)
			o.MergedDecoderPath = filepath.Join(f.dir, "gone.onnx")
		}, "gone.onnx"},
		{"v1 cached decoder missing on disk", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, true, false)
			o.CachedDecoderPath = filepath.Join(f.dir, "gone.onnx")
		}, "gone.onnx"},
		{"complete v2 with a stale v1 path", func(f *familyFiles, o *OfflineConfig) {
			*o = moonshineConfig(f, false, true)
			o.PreprocessorPath = filepath.Join(f.dir, "gone.onnx")
		}, "gone.onnx"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var o OfflineConfig
			tc.mutate(newFamilyFiles(t), &o)
			c := offlineValidationConfig(o)
			err := c.Validate()
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tc.wantErr != "" && err == nil:
				t.Fatalf("expected an error containing %q", tc.wantErr)
			case tc.wantErr != "" && !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("error %q does not contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestValidateOfflineRejectsUnknownFamily(t *testing.T) {
	c := offlineValidationConfig(OfflineConfig{ModelFamily: "does_not_exist"})
	err := c.Validate()
	if err == nil || !strings.Contains(err.Error(), `unsupported offline model family "does_not_exist"`) {
		t.Fatalf("got %v, want unsupported-family error", err)
	}
}

func TestOfflineConfigJSONOmitsUnsetExtendedFields(t *testing.T) {
	// The extended fields are omitempty so existing serialized configs and the
	// six original families keep their exact JSON shape.
	raw, err := json.Marshal(OfflineConfig{ModelFamily: OfflineFamilyWhisper, EncoderPath: "e", DecoderPath: "d", TokensPath: "t"})
	if err != nil {
		t.Fatal(err)
	}
	data := string(raw)
	for _, key := range []string{
		"preprocessor_path", "uncached_decoder_path", "cached_decoder_path", "merged_decoder_path",
		"conv_frontend_path", "encoder_adaptor_path", "llm_path", "embedding_path", "tokenizer_path",
		"src_lang", "tgt_lang", "use_punctuation", "system_prompt", "user_prompt",
		"max_total_len", "max_new_tokens", "temperature", "top_p", "seed", "hotwords",
	} {
		if strings.Contains(data, `"`+key+`"`) {
			t.Errorf("unset %s serialized: %s", key, data)
		}
	}
}
