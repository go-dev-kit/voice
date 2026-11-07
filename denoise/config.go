// Package denoise wraps Sherpa ONNX's speech-enhancement (denoising) models
// (GTCRN/DPDFNet) for both offline (whole-buffer) and online
// (streaming/chunked) use, satisfying types.SpeechDenoiser and
// types.StreamingDenoiser respectively.
package denoise

import (
	"fmt"
	"math"
	"os"
)

// DenoiserConfig owns Sherpa speech-enhancement model configuration shared
// by both the offline (SherpaSpeechDenoiser) and streaming
// (SherpaStreamingDenoiser) denoisers. Exactly one model family must be
// set: GtcrnModel (GTCRN) or DpdfNetModel (DPDFNet) — never both, never
// neither.
type DenoiserConfig struct {
	// GtcrnModel is the path to a GTCRN speech-enhancement ONNX model.
	// Mutually exclusive with DpdfNetModel.
	GtcrnModel string `json:"gtcrn_model,omitempty"`
	// DpdfNetModel is the path to a DPDFNet speech-enhancement ONNX model.
	// Mutually exclusive with GtcrnModel.
	DpdfNetModel string `json:"dpdfnet_model,omitempty"`

	Provider   string `json:"provider"`
	NumThreads int    `json:"num_threads"`
	Debug      bool   `json:"debug"`
}

// DefaultDenoiserConfig returns runtime defaults. Model paths are left
// empty since exactly one must be supplied by the caller.
func DefaultDenoiserConfig() DenoiserConfig {
	return DenoiserConfig{
		Provider:   "cpu",
		NumThreads: 1,
	}
}

// ApplyDefaults merges zero-value runtime settings with conservative
// defaults. Model paths are left untouched.
func (c *DenoiserConfig) ApplyDefaults() {
	defaults := DefaultDenoiserConfig()
	if c.Provider == "" {
		c.Provider = defaults.Provider
	}
	if c.NumThreads <= 0 {
		c.NumThreads = defaults.NumThreads
	}
}

// Validate validates denoiser configuration, including that exactly one
// model family (GTCRN xor DPDFNet) is configured and its model file exists.
func (c *DenoiserConfig) Validate() error {
	c.ApplyDefaults()

	var errs []error
	hasGtcrn := c.GtcrnModel != ""
	hasDpdfNet := c.DpdfNetModel != ""

	switch {
	case hasGtcrn && hasDpdfNet:
		errs = append(errs, fmt.Errorf("exactly one of GtcrnModel or DpdfNetModel must be set, got both"))
	case hasGtcrn:
		if err := requireFile(c.GtcrnModel); err != nil {
			errs = append(errs, fmt.Errorf("gtcrn model path: %w", err))
		}
	case hasDpdfNet:
		if err := requireFile(c.DpdfNetModel); err != nil {
			errs = append(errs, fmt.Errorf("dpdfnet model path: %w", err))
		}
	default:
		errs = append(errs, fmt.Errorf("exactly one of GtcrnModel or DpdfNetModel must be set, got neither"))
	}

	if c.NumThreads <= 0 {
		errs = append(errs, fmt.Errorf("num threads must be positive, got %d", c.NumThreads))
	} else if c.NumThreads > math.MaxInt32 {
		errs = append(errs, fmt.Errorf("num threads must not exceed %d, got %d", math.MaxInt32, c.NumThreads))
	}

	if len(errs) > 0 {
		return fmt.Errorf("denoiser config validation failed: %v", errs)
	}
	return nil
}

// requireFile validates that path exists and is a regular file, not a
// directory.
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
