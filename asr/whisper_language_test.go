package asr

import (
	"strings"
	"testing"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
)

func whisperTestConfig(t *testing.T, language, task string) Config {
	t.Helper()
	files := newFamilyFiles(t)
	c := offlineBaseConfig()
	c.Language = "en"
	c.Offline = OfflineConfig{
		ModelFamily: OfflineFamilyWhisper,
		TokensPath:  files.file("tokens.txt"),
		EncoderPath: files.file("encoder.onnx"),
		DecoderPath: files.file("decoder.onnx"),
		Language:    language,
		Task:        task,
	}
	return c
}

func TestWhisperLanguageSetHas100Codes(t *testing.T) {
	if got := len(whisperLanguages); got != 100 {
		t.Fatalf("whisper language set has %d codes, want 100", got)
	}
	for _, code := range []string{"yue", "haw", "jw", "en", "zh"} {
		if _, ok := whisperLanguages[code]; !ok {
			t.Errorf("missing whisper language %q", code)
		}
	}
}

func TestWhisperEmptyAndAutoLanguageReachSherpaEmpty(t *testing.T) {
	for _, language := range []string{"", "auto", "AUTO", " auto "} {
		t.Run("language="+language, func(t *testing.T) {
			c := whisperTestConfig(t, language, "")
			if err := c.Validate(); err != nil {
				t.Fatalf("validate: %v", err)
			}
			mapped, err := buildOfflineRecognizerConfig(&c)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			if got := mapped.ModelConfig.Whisper.Language; got != "" {
				t.Fatalf("sherpa whisper language = %q, want empty (auto-detect)", got)
			}
			if got := offlineFallbackLanguage(c.Offline); got != "" {
				t.Fatalf("fallback result language = %q, want empty", got)
			}
			if mapped.ModelConfig.Whisper.Task != "transcribe" {
				t.Fatalf("task = %q, want default transcribe", mapped.ModelConfig.Whisper.Task)
			}
		})
	}
}

func TestWhisperExplicitLanguagePassesThrough(t *testing.T) {
	for _, language := range []string{"de", "yue", "haw", "jw", "en"} {
		c := whisperTestConfig(t, language, "translate")
		if err := c.Validate(); err != nil {
			t.Fatalf("%s: validate: %v", language, err)
		}
		mapped, err := buildOfflineRecognizerConfig(&c)
		if err != nil {
			t.Fatalf("%s: build: %v", language, err)
		}
		if got := mapped.ModelConfig.Whisper.Language; got != language {
			t.Errorf("sherpa whisper language = %q, want %q", got, language)
		}
	}
}

func TestWhisperRejectsUnknownLanguageAndTask(t *testing.T) {
	cases := []struct {
		name, language, task, want string
	}{
		{"unknown language", "xx", "", "language"},
		{"long name", "german", "", "language"},
		{"uppercase code", "DE", "", "language"},
		{"bad task", "de", "summarize", "task"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := whisperTestConfig(t, tc.language, tc.task)
			err := c.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate() = %v, want error mentioning %q", err, tc.want)
			}
			if _, err := NewOfflineTranscriber(&c); err == nil {
				t.Fatal("NewOfflineTranscriber accepted an invalid whisper config")
			}
		})
	}
}

func TestOtherOfflineFamiliesStillInheritConfigLanguage(t *testing.T) {
	c := offlineBaseConfig()
	c.Language = "de"
	c.Offline = OfflineConfig{ModelFamily: OfflineFamilySenseVoice}
	c.ApplyDefaults()
	if c.Offline.Language != "de" {
		t.Fatalf("sense_voice language = %q, want inherited de", c.Offline.Language)
	}
}

func TestTranscriptionLanguagePrefersDetectedCode(t *testing.T) {
	got := transcriptionFromOfflineResult(&sherpa.OfflineRecognizerResult{Text: "hallo", Lang: "de"}, "", 16000, 16000)
	if got.Language != "de" {
		t.Fatalf("language = %q, want detected de", got.Language)
	}
}
