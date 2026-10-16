// Command denoise demonstrates VoiceKit's Sherpa ONNX speech denoiser
// (GTCRN/DPDFNet), both offline (whole-buffer) and online
// (streaming/chunked) modes. It requires real Sherpa ONNX speech-enhancement
// model files, so it is gated on environment variables and prints setup
// guidance instead of failing when they are unset.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"go.glpx.pro/gdk/voice/audio"
	"go.glpx.pro/gdk/voice/denoise"
)

func main() {
	fmt.Println("VoiceKit Speech Denoiser Example")
	fmt.Println("=================================")

	wavPath := os.Getenv("VOICEKIT_DENOISE_WAV")
	if wavPath == "" {
		fmt.Println("Set VOICEKIT_DENOISE_WAV to a mono WAV file to run this example.")
		return
	}

	config, ok := denoiserConfigFromEnv()
	if !ok {
		fmt.Println("Sherpa speech denoiser model paths are not configured.")
		fmt.Println("Set exactly one of:")
		fmt.Println("  - VOICEKIT_DENOISE_GTCRN_MODEL (GTCRN)")
		fmt.Println("  - VOICEKIT_DENOISE_DPDFNET_MODEL (DPDFNet)")
		return
	}

	sampleRate := envInt("VOICEKIT_DENOISE_SAMPLE_RATE", 16000)
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

	runOffline(config, samples, sampleRate)
	runStreaming(config, samples)

	fmt.Println()
	fmt.Println("Speech denoiser demonstration completed.")
}

func runOffline(config denoise.DenoiserConfig, samples []float32, sampleRate int) {
	fmt.Println()
	fmt.Println("Offline (whole-buffer) denoising:")

	denoiser, err := denoise.NewSherpaSpeechDenoiser(&config)
	if err != nil {
		log.Fatalf("failed to create offline speech denoiser: %v", err)
	}
	defer denoiser.Close()

	denoised, err := denoiser.Denoise(context.Background(), samples, sampleRate)
	if err != nil {
		log.Fatalf("offline denoising failed: %v", err)
	}
	fmt.Printf("  input samples:    %d\n", len(samples))
	fmt.Printf("  denoised samples: %d\n", len(denoised))
}

// runStreaming demonstrates the online/streaming denoiser by feeding the
// same input in fixed-size chunks. The streaming model's sample rate is
// fixed at construction (see SherpaStreamingDenoiser docs); this demo does
// not resample, so it is only representative when the WAV's decode sample
// rate matches the streaming model's rate.
func runStreaming(config denoise.DenoiserConfig, samples []float32) {
	fmt.Println()
	fmt.Println("Online (streaming/chunked) denoising:")

	denoiser, err := denoise.NewSherpaStreamingDenoiser(&config)
	if err != nil {
		log.Fatalf("failed to create streaming speech denoiser: %v", err)
	}
	defer denoiser.Close()

	const chunkSize = 1600 // 100ms at 16kHz
	total := 0
	ctx := context.Background()
	for start := 0; start < len(samples); start += chunkSize {
		end := start + chunkSize
		if end > len(samples) {
			end = len(samples)
		}
		out, err := denoiser.Accept(ctx, samples[start:end])
		if err != nil {
			log.Fatalf("streaming denoising failed: %v", err)
		}
		total += len(out)
	}

	flushed, err := denoiser.Flush()
	if err != nil {
		log.Fatalf("streaming denoiser flush failed: %v", err)
	}
	total += len(flushed)

	// Reset clears stream state so the same instance can be reused for a new
	// stream; demonstrated here rather than acted upon further.
	denoiser.Reset()

	fmt.Printf("  input samples:    %d\n", len(samples))
	fmt.Printf("  denoised samples: %d (including flush)\n", total)
}

func denoiserConfigFromEnv() (denoise.DenoiserConfig, bool) {
	config := denoise.DefaultDenoiserConfig()
	config.GtcrnModel = os.Getenv("VOICEKIT_DENOISE_GTCRN_MODEL")
	config.DpdfNetModel = os.Getenv("VOICEKIT_DENOISE_DPDFNET_MODEL")
	config.Provider = envString("VOICEKIT_DENOISE_PROVIDER", config.Provider)
	config.NumThreads = envInt("VOICEKIT_DENOISE_THREADS", config.NumThreads)

	hasGtcrn := config.GtcrnModel != ""
	hasDpdfNet := config.DpdfNetModel != ""
	return config, hasGtcrn != hasDpdfNet
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
