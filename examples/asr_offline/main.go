package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"go.glpx.pro/gdk/voice"
	"go.glpx.pro/gdk/voice/audio"
)

func main() {
	wavPath := os.Getenv("VOICEKIT_OFFLINE_ASR_WAV")
	if wavPath == "" {
		fmt.Println("Set VOICEKIT_OFFLINE_ASR_WAV to a mono WAV file to run this example.")
		return
	}

	asrConfig, ok := offlineASRConfigFromEnv()
	if !ok {
		fmt.Println("Sherpa offline ASR model paths are not configured.")
		fmt.Println("Set VOICEKIT_OFFLINE_ASR_FAMILY plus the required VOICEKIT_OFFLINE_ASR_* model path variables.")
		return
	}

	sampleRate := envInt("VOICEKIT_OFFLINE_ASR_SAMPLE_RATE", 16000)
	config := &voicekit.Config{
		Audio: voicekit.AudioConfig{
			SampleRate:      sampleRate,
			Channels:        1,
			NormalizeFactor: 32768.0,
		},
		ASR: asrConfig,
	}

	vk, err := voicekit.NewVoiceKit(config)
	if err != nil {
		log.Fatalf("failed to create VoiceKit: %v", err)
	}
	defer vk.Close()

	wavBytes, err := os.ReadFile(filepath.Clean(wavPath))
	if err != nil {
		log.Fatalf("failed to read WAV file: %v", err)
	}
	wavConfig := audio.DefaultWAVConfig()
	wavConfig.SampleRate = sampleRate
	samples, err := vk.Audio().ConvertToFloat32(wavBytes, wavConfig)
	if err != nil {
		log.Fatalf("failed to decode WAV file: %v", err)
	}

	result, err := vk.Transcriber().Transcribe(context.Background(), samples, sampleRate)
	if err != nil {
		log.Fatalf("offline transcription failed: %v", err)
	}
	fmt.Println(result.Text)
}

func offlineASRConfigFromEnv() (voicekit.ASRConfig, bool) {
	config := voicekit.ASRConfig{
		Enabled:      true,
		Backend:      "sherpa_offline",
		DefaultModel: "sherpa_offline",
		Language:     "en",
		Offline: voicekit.OfflineConfig{
			ModelFamily: os.Getenv("VOICEKIT_OFFLINE_ASR_FAMILY"),
			TokensPath:  os.Getenv("VOICEKIT_OFFLINE_ASR_TOKENS"),
			ModelPath:   os.Getenv("VOICEKIT_OFFLINE_ASR_MODEL"),
			EncoderPath: os.Getenv("VOICEKIT_OFFLINE_ASR_ENCODER"),
			DecoderPath: os.Getenv("VOICEKIT_OFFLINE_ASR_DECODER"),
			JoinerPath:  os.Getenv("VOICEKIT_OFFLINE_ASR_JOINER"),
			Language:    "en",
			Task:        "transcribe",
		},
	}
	if config.Offline.ModelFamily == "" {
		config.Offline.ModelFamily = "sense_voice"
	}

	switch config.Offline.ModelFamily {
	case "transducer":
		return config, config.Offline.TokensPath != "" && config.Offline.EncoderPath != "" &&
			config.Offline.DecoderPath != "" && config.Offline.JoinerPath != ""
	case "paraformer", "zipformer_ctc", "nemo_ctc":
		return config, config.Offline.TokensPath != "" && config.Offline.ModelPath != ""
	case "sense_voice":
		return config, config.Offline.ModelPath != ""
	case "whisper":
		return config, config.Offline.EncoderPath != "" && config.Offline.DecoderPath != ""
	default:
		return config, false
	}
}

func envInt(name string, fallback int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}
