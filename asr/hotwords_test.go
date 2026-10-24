package asr

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Hermetic tests for recognizer-level hotwords: the config-to-Sherpa mapping
// and the validation that refuses setups Sherpa ignores or exits on.

func touch(t *testing.T, dir, name string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func onlineHotwordsConfig(t *testing.T) Config {
	t.Helper()
	dir := t.TempDir()
	c := DefaultConfig()
	c.Enabled = true
	c.Backend = BackendSherpaOnline
	c.DecodingMethod = "modified_beam_search"
	c.Online = OnlineConfig{
		TokensPath:  touch(t, dir, "tokens.txt"),
		EncoderPath: touch(t, dir, "enc.onnx"),
		DecoderPath: touch(t, dir, "dec.onnx"),
		JoinerPath:  touch(t, dir, "join.onnx"),
	}
	return c
}

func offlineHotwordsConfig(t *testing.T) Config {
	t.Helper()
	dir := t.TempDir()
	c := DefaultConfig()
	c.Enabled = true
	c.Backend = BackendSherpaOffline
	c.DecodingMethod = "modified_beam_search"
	c.Offline = OfflineConfig{
		ModelFamily: OfflineFamilyTransducer,
		TokensPath:  touch(t, dir, "tokens.txt"),
		EncoderPath: touch(t, dir, "enc.onnx"),
		DecoderPath: touch(t, dir, "dec.onnx"),
		JoinerPath:  touch(t, dir, "join.onnx"),
	}
	return c
}

func TestBuildRecognizerConfigMapsHotwords(t *testing.T) {
	t.Run("online file", func(t *testing.T) {
		c := onlineHotwordsConfig(t)
		c.Online.HotwordsFile = "hw.txt"
		c.Online.HotwordsScore = 2.5
		c.Online.ModelingUnit = "bpe"
		c.Online.BpeVocab = "bpe.vocab"
		got, err := buildOnlineRecognizerConfig(&c)
		if err != nil {
			t.Fatal(err)
		}
		if got.HotwordsFile != "hw.txt" || got.HotwordsScore != 2.5 {
			t.Errorf("hotwords file/score = %q/%v", got.HotwordsFile, got.HotwordsScore)
		}
		if got.ModelConfig.ModelingUnit != "bpe" || got.ModelConfig.BpeVocab != "bpe.vocab" {
			t.Errorf("modeling unit/vocab = %q/%q", got.ModelConfig.ModelingUnit, got.ModelConfig.BpeVocab)
		}
		if got.HotwordsBuf != "" || got.HotwordsBufSize != 0 {
			t.Errorf("unexpected in-memory hotwords %q/%d", got.HotwordsBuf, got.HotwordsBufSize)
		}
	})
	t.Run("online text", func(t *testing.T) {
		c := onlineHotwordsConfig(t)
		c.Online.HotwordsText = "alpha :3\nbeta\n"
		got, err := buildOnlineRecognizerConfig(&c)
		if err != nil {
			t.Fatal(err)
		}
		if got.HotwordsBuf != c.Online.HotwordsText || got.HotwordsBufSize != len(c.Online.HotwordsText) {
			t.Errorf("buf = %q size %d", got.HotwordsBuf, got.HotwordsBufSize)
		}
		if got.HotwordsFile != "" {
			t.Errorf("HotwordsFile = %q, want empty", got.HotwordsFile)
		}
	})
	t.Run("online models entry", func(t *testing.T) {
		c := onlineHotwordsConfig(t)
		entry := c.Online
		entry.HotwordsFile = "hw.txt"
		got, err := buildOnlineRecognizerConfigFor(&c, entry)
		if err != nil {
			t.Fatal(err)
		}
		if got.HotwordsFile != "hw.txt" {
			t.Errorf("HotwordsFile = %q", got.HotwordsFile)
		}
	})
	t.Run("offline transducer", func(t *testing.T) {
		c := offlineHotwordsConfig(t)
		c.Offline.HotwordsFile = "hw.txt"
		c.Offline.HotwordsScore = 3
		c.Offline.ModelingUnit = "cjkchar+bpe"
		c.Offline.BpeVocab = "bpe.vocab"
		got, err := buildOfflineRecognizerConfig(&c)
		if err != nil {
			t.Fatal(err)
		}
		if got.HotwordsFile != "hw.txt" || got.HotwordsScore != 3 {
			t.Errorf("hotwords file/score = %q/%v", got.HotwordsFile, got.HotwordsScore)
		}
		if got.ModelConfig.ModelingUnit != "cjkchar+bpe" || got.ModelConfig.BpeVocab != "bpe.vocab" {
			t.Errorf("modeling unit/vocab = %q/%q", got.ModelConfig.ModelingUnit, got.ModelConfig.BpeVocab)
		}
	})
	t.Run("unset leaves native defaults", func(t *testing.T) {
		c := offlineHotwordsConfig(t)
		got, err := buildOfflineRecognizerConfig(&c)
		if err != nil {
			t.Fatal(err)
		}
		if got.HotwordsFile != "" || got.HotwordsScore != 0 || got.ModelConfig.ModelingUnit != "" || got.ModelConfig.BpeVocab != "" {
			t.Errorf("hotwords leaked into the native config: %+v", got)
		}
	})
	t.Run("prompt Hotwords stays on the family field", func(t *testing.T) {
		c := offlineBaseConfig()
		c.Offline = OfflineConfig{
			ModelFamily: OfflineFamilyQwen3ASR, Hotwords: "alpha",
			ConvFrontendPath: "cf", EncoderPath: "e", DecoderPath: "d", TokenizerPath: "tk",
		}
		got, err := buildOfflineRecognizerConfig(&c)
		if err != nil {
			t.Fatal(err)
		}
		if got.HotwordsFile != "" || got.ModelConfig.Qwen3ASR.Hotwords != "alpha" {
			t.Errorf("prompt hotwords mis-mapped: file %q qwen %q", got.HotwordsFile, got.ModelConfig.Qwen3ASR.Hotwords)
		}
	})
}

func TestValidateHotwordsAccepts(t *testing.T) {
	dir := t.TempDir()
	hw := touch(t, dir, "hw.txt")
	vocab := touch(t, dir, "bpe.vocab")

	t.Run("online file with bpe", func(t *testing.T) {
		c := onlineHotwordsConfig(t)
		c.Online.HotwordsFile, c.Online.ModelingUnit, c.Online.BpeVocab = hw, "bpe", vocab
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("online text default unit", func(t *testing.T) {
		c := onlineHotwordsConfig(t)
		c.Online.HotwordsText = "alpha"
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("online models entries", func(t *testing.T) {
		c := onlineHotwordsConfig(t)
		e := c.Online
		e.Name, e.HotwordsFile = "m1", hw
		c.OnlineModels = []OnlineConfig{e}
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("offline transducer", func(t *testing.T) {
		c := offlineHotwordsConfig(t)
		c.Offline.HotwordsFile, c.Offline.HotwordsScore, c.Offline.ModelingUnit, c.Offline.BpeVocab = hw, 2, "bpe", vocab
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
	})
}

func TestValidateHotwordsRefuses(t *testing.T) {
	dir := t.TempDir()
	hw := touch(t, dir, "hw.txt")
	missing := filepath.Join(dir, "nope")

	cases := []struct {
		name string
		cfg  func(t *testing.T) Config
		want string
	}{
		{"offline greedy search", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.DecodingMethod = "greedy_search"
			c.Offline.HotwordsFile = hw
			return c
		}, `require decoding method "modified_beam_search"`},
		{"online greedy search with text", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			c.DecodingMethod = "greedy_search"
			c.Online.HotwordsText = "alpha"
			return c
		}, `require decoding method "modified_beam_search"`},
		{"offline missing file", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile = missing
			return c
		}, "hotwords file"},
		{"offline file is a directory", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile = dir
			return c
		}, "is a directory"},
		{"offline non-transducer family", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.ModelFamily = OfflineFamilyNemoCTC
			c.Offline.ModelPath = c.Offline.EncoderPath
			c.Offline.HotwordsFile = hw
			return c
		}, "require a transducer model"},
		{"online single-file ctc", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			c.Online.EncoderPath, c.Online.DecoderPath, c.Online.JoinerPath = "", "", ""
			c.Online.ModelPath = c.Online.TokensPath
			c.Online.HotwordsFile = hw
			return c
		}, "require a transducer model"},
		{"online file and text", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			c.Online.HotwordsFile, c.Online.HotwordsText = hw, "alpha"
			return c
		}, "both a hotwords file and hotwords text"},
		{"online blank text", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			c.Online.HotwordsText = " \n "
			return c
		}, "text is blank"},
		{"negative score", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile, c.Offline.HotwordsScore = hw, -1
			return c
		}, "score must be"},
		{"NaN score", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile, c.Offline.HotwordsScore = hw, float32(math.NaN())
			return c
		}, "score must be"},
		{"score without hotwords", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsScore = 2
			return c
		}, "score is set without"},
		{"modeling unit without hotwords", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			c.Online.ModelingUnit = "bpe"
			return c
		}, "apply only to hotwords"},
		{"unknown modeling unit exits the process", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile, c.Offline.ModelingUnit = hw, "word"
			return c
		}, `modeling unit "word" is not one of`},
		{"bpe without vocab dereferences null", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile, c.Offline.ModelingUnit = hw, "bpe"
			return c
		}, "requires a BPE vocab file"},
		{"cjkchar+bpe without vocab", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			c.Online.HotwordsText, c.Online.ModelingUnit = "alpha", "cjkchar+bpe"
			return c
		}, "requires a BPE vocab file"},
		{"missing vocab file", func(t *testing.T) Config {
			c := offlineHotwordsConfig(t)
			c.Offline.HotwordsFile, c.Offline.ModelingUnit, c.Offline.BpeVocab = hw, "bpe", missing
			return c
		}, "BPE vocab"},
		{"online models entry names itself", func(t *testing.T) Config {
			c := onlineHotwordsConfig(t)
			e := c.Online
			e.Name, e.HotwordsFile = "m1", missing
			c.OnlineModels = []OnlineConfig{e}
			return c
		}, `online model "m1" hotwords file`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := tc.cfg(t)
			err := c.Validate()
			if err == nil {
				t.Fatal("Validate accepted the config")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}
