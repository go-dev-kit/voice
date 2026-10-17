package asr

import (
	"context"
	"os"
	"testing"
)

func TestSherpaOfflineModelIntegration(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOffline
	config.DefaultModel = "integration_sherpa_offline"
	config.Offline.ModelFamily = os.Getenv("VOICEKIT_OFFLINE_ASR_FAMILY")
	if config.Offline.ModelFamily == "" {
		config.Offline.ModelFamily = OfflineFamilySenseVoice
	}
	config.Offline.TokensPath = os.Getenv("VOICEKIT_OFFLINE_ASR_TOKENS")
	config.Offline.ModelPath = os.Getenv("VOICEKIT_OFFLINE_ASR_MODEL")
	config.Offline.EncoderPath = os.Getenv("VOICEKIT_OFFLINE_ASR_ENCODER")
	config.Offline.DecoderPath = os.Getenv("VOICEKIT_OFFLINE_ASR_DECODER")
	config.Offline.JoinerPath = os.Getenv("VOICEKIT_OFFLINE_ASR_JOINER")
	config.Offline.Language = "en"

	if !offlineIntegrationConfigured(config.Offline) {
		t.Skip("set VOICEKIT_OFFLINE_ASR_* model path env vars to run Sherpa offline ASR integration")
	}

	model, err := NewSherpaOfflineModel(&config)
	if err != nil {
		t.Fatalf("failed to create Sherpa offline model: %v", err)
	}
	defer model.Close()

	audio := make([]float32, 16000)
	result, err := model.Transcribe(context.Background(), audio, 16000)
	if err != nil {
		t.Fatalf("offline transcription failed: %v", err)
	}
	if result == nil {
		t.Fatal("expected transcription result")
	}
}

func offlineIntegrationConfigured(config OfflineConfig) bool {
	switch config.ModelFamily {
	case OfflineFamilyTransducer:
		return config.TokensPath != "" && config.EncoderPath != "" && config.DecoderPath != "" && config.JoinerPath != ""
	case OfflineFamilyParaformer, OfflineFamilyZipformerCTC, OfflineFamilyNemoCTC:
		return config.TokensPath != "" && config.ModelPath != ""
	case OfflineFamilySenseVoice:
		return config.TokensPath != "" && config.ModelPath != ""
	case OfflineFamilyWhisper:
		return config.TokensPath != "" && config.EncoderPath != "" && config.DecoderPath != ""
	default:
		return false
	}
}
