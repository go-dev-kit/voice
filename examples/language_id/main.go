package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"go.glpx.pro/gdk/voice/asr"
	"go.glpx.pro/gdk/voice/audio"
)

func main() {
	wavPath := os.Getenv("VOICEKIT_LID_WAV")
	if wavPath == "" {
		fmt.Println("Set VOICEKIT_LID_WAV to a mono WAV file to run this example.")
		return
	}

	config, ok := languageIDConfigFromEnv()
	if !ok {
		fmt.Println("Sherpa spoken language identification model paths are not configured.")
		fmt.Println("Set VOICEKIT_LID_ENCODER and VOICEKIT_LID_DECODER to a Whisper encoder/decoder ONNX model pair.")
		return
	}

	identifier, err := asr.NewSherpaLanguageIdentifier(&config)
	if err != nil {
		log.Fatalf("failed to create language identifier: %v", err)
	}
	defer identifier.Close()

	sampleRate := envInt("VOICEKIT_LID_SAMPLE_RATE", 16000)
	wavBytes, err := os.ReadFile(filepath.Clean(wavPath))
	if err != nil {
		log.Fatalf("failed to read WAV file: %v", err)
	}

	converter, err := audio.NewConverter(nil)
	if err != nil {
		log.Fatalf("failed to create audio converter: %v", err)
	}
	wavConfig := audio.DefaultWAVConfig()
	wavConfig.SampleRate = sampleRate
	samples, err := converter.ConvertToFloat32(wavBytes, wavConfig)
	if err != nil {
		log.Fatalf("failed to decode WAV file: %v", err)
	}

	result, err := identifier.Identify(context.Background(), samples, sampleRate)
	if err != nil {
		log.Fatalf("language identification failed: %v", err)
	}
	fmt.Printf("Detected language: %s\n", result.Language)
}

func languageIDConfigFromEnv() (asr.LanguageIDConfig, bool) {
	config := asr.DefaultLanguageIDConfig()
	config.EncoderPath = os.Getenv("VOICEKIT_LID_ENCODER")
	config.DecoderPath = os.Getenv("VOICEKIT_LID_DECODER")
	config.Provider = envString("VOICEKIT_LID_PROVIDER", config.Provider)
	config.NumThreads = envInt("VOICEKIT_LID_THREADS", config.NumThreads)
	return config, config.EncoderPath != "" && config.DecoderPath != ""
}

func envString(name, fallback string) string {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	return value
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
