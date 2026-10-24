package asr

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"go.glpx.pro/gdk/voice/audio"
)

// TestSherpaOfflineHotwordsIntegration loads an offline transducer with a
// hotwords file under modified_beam_search and decodes with it. It uses the
// same VOICEKIT_OFFLINE_ASR_* variables as TestSherpaOfflineModelIntegration
// (family transducer) plus, optionally:
//
//	VOICEKIT_OFFLINE_ASR_BPE_VOCAB   - sentencepiece vocabulary; when unset the
//	                                   test derives one from tokens.txt
//	VOICEKIT_OFFLINE_ASR_HOTWORDS_WAV - 16 kHz mono WAV to decode
//	VOICEKIT_OFFLINE_ASR_HOTWORD     - phrase to boost on that recording
//
// Without the WAV it only proves the recognizer loads and decodes silence.
// Verified on the Parakeet TDT 0.6b v3 int8 export with FLEURS de clip
// 17113436913741548046.wav: unboosted "Hundehudel", boosted "Hunderudel" for
// the hotword "Hunderudel".
func TestSherpaOfflineHotwordsIntegration(t *testing.T) {
	base := DefaultConfig()
	base.Enabled = true
	base.Backend = BackendSherpaOffline
	base.DefaultModel = "integration_hotwords"
	base.DecodingMethod = "modified_beam_search"
	base.Offline = OfflineConfig{
		ModelFamily: OfflineFamilyTransducer,
		TokensPath:  os.Getenv("VOICEKIT_OFFLINE_ASR_TOKENS"),
		EncoderPath: os.Getenv("VOICEKIT_OFFLINE_ASR_ENCODER"),
		DecoderPath: os.Getenv("VOICEKIT_OFFLINE_ASR_DECODER"),
		JoinerPath:  os.Getenv("VOICEKIT_OFFLINE_ASR_JOINER"),
	}
	if os.Getenv("VOICEKIT_OFFLINE_ASR_FAMILY") != OfflineFamilyTransducer || !offlineIntegrationConfigured(base.Offline) {
		t.Skip("set VOICEKIT_OFFLINE_ASR_FAMILY=transducer and the transducer path env vars to run the hotwords integration")
	}

	dir := t.TempDir()
	vocab := os.Getenv("VOICEKIT_OFFLINE_ASR_BPE_VOCAB")
	if vocab == "" {
		vocab = filepath.Join(dir, "bpe.vocab")
		if err := writeBPEVocabFromTokens(base.Offline.TokensPath, vocab); err != nil {
			t.Fatal(err)
		}
	}
	word := os.Getenv("VOICEKIT_OFFLINE_ASR_HOTWORD")
	if word == "" {
		word = "hello"
	}
	hotwordsFile := filepath.Join(dir, "hotwords.txt")
	if err := os.WriteFile(hotwordsFile, []byte(word+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	decode := func(t *testing.T, cfg Config, samples []float32) string {
		t.Helper()
		model, err := NewSherpaOfflineModel(&cfg)
		if err != nil {
			t.Fatalf("create model: %v", err)
		}
		defer model.Close()
		result, err := model.Transcribe(context.Background(), samples, 16000)
		if err != nil {
			t.Fatalf("transcribe: %v", err)
		}
		return result.Text
	}

	biased := base
	biased.Offline.HotwordsFile = hotwordsFile
	biased.Offline.HotwordsScore = 3
	biased.Offline.ModelingUnit = "bpe"
	biased.Offline.BpeVocab = vocab

	t.Run("loads and decodes silence", func(t *testing.T) {
		decode(t, biased, make([]float32, 16000))
	})

	wav := os.Getenv("VOICEKIT_OFFLINE_ASR_HOTWORDS_WAV")
	if wav == "" {
		return
	}
	data, err := os.ReadFile(wav)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := audio.DecodeMono(data, audio.FormatWAV, 16000, audio.DecodeLimits{})
	if err != nil {
		t.Fatal(err)
	}
	t.Run("boost changes the transcript", func(t *testing.T) {
		plain := decode(t, base, samples)
		boosted := decode(t, biased, samples)
		t.Logf("plain:   %s\nboosted: %s", plain, boosted)
		if strings.Contains(strings.ToLower(plain), strings.ToLower(word)) {
			t.Skipf("the unboosted transcript already contains %q; pick a hotword the model gets wrong", word)
		}
		if !strings.Contains(strings.ToLower(boosted), strings.ToLower(word)) {
			t.Fatalf("hotword %q missing from the boosted transcript", word)
		}
	})
}

// writeBPEVocabFromTokens writes a sentencepiece-style "piece<TAB>score" file
// from a sherpa tokens.txt ("piece id" lines), scoring by id so earlier pieces
// win merges the way a BPE vocabulary's rank does.
func writeBPEVocabFromTokens(tokens, out string) error {
	data, err := os.ReadFile(tokens)
	if err != nil {
		return err
	}
	var b strings.Builder
	for line := range strings.SplitSeq(string(data), "\n") {
		i := strings.LastIndexByte(line, ' ')
		if i <= 0 {
			continue
		}
		var id int
		if _, err := fmt.Sscanf(line[i+1:], "%d", &id); err != nil {
			continue
		}
		fmt.Fprintf(&b, "%s\t%d\n", line[:i], -id)
	}
	return os.WriteFile(out, []byte(b.String()), 0o600)
}
