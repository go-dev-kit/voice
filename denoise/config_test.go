package denoise

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestDenoiserConfig_ApplyDefaults(t *testing.T) {
	var c DenoiserConfig
	c.ApplyDefaults()
	if c.Provider != "cpu" {
		t.Fatalf("Provider = %q, want cpu", c.Provider)
	}
	if c.NumThreads != 1 {
		t.Fatalf("NumThreads = %d, want 1", c.NumThreads)
	}
}

func TestDenoiserConfig_ApplyDefaults_PreservesExplicitValues(t *testing.T) {
	c := DenoiserConfig{Provider: "cuda", NumThreads: 4}
	c.ApplyDefaults()
	if c.Provider != "cuda" {
		t.Fatalf("Provider = %q, want cuda", c.Provider)
	}
	if c.NumThreads != 4 {
		t.Fatalf("NumThreads = %d, want 4", c.NumThreads)
	}
}

func TestDenoiserConfig_Validate_ModelFamilyXOR(t *testing.T) {
	modelPath := writeTempModel(t)

	tests := []struct {
		name    string
		config  DenoiserConfig
		wantErr bool
	}{
		{name: "neither set", config: DenoiserConfig{}, wantErr: true},
		{name: "gtcrn only", config: DenoiserConfig{GtcrnModel: modelPath}, wantErr: false},
		{name: "dpdfnet only", config: DenoiserConfig{DpdfNetModel: modelPath}, wantErr: false},
		{name: "both set", config: DenoiserConfig{GtcrnModel: modelPath, DpdfNetModel: modelPath}, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if tt.wantErr && err == nil {
				t.Fatal("expected validation error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("unexpected validation error: %v", err)
			}
		})
	}
}

func TestDenoiserConfig_Validate_MissingModelFile(t *testing.T) {
	c := DenoiserConfig{GtcrnModel: filepath.Join(t.TempDir(), "missing.onnx")}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error for missing model file")
	}
}

func TestDenoiserConfig_Validate_ModelPathIsDirectory(t *testing.T) {
	c := DenoiserConfig{GtcrnModel: t.TempDir()}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error when model path is a directory")
	}
}

func TestDenoiserConfig_Validate_NonPositiveNumThreadsDefaulted(t *testing.T) {
	modelPath := writeTempModel(t)
	c := DenoiserConfig{GtcrnModel: modelPath, NumThreads: -1}
	// ApplyDefaults runs inside Validate and resets non-positive NumThreads
	// to the default (1) before the positivity check, so this should not
	// error.
	if err := c.Validate(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.NumThreads != 1 {
		t.Fatalf("NumThreads = %d, want 1", c.NumThreads)
	}
}

func TestDenoiserConfig_Validate_RejectsNativeThreadOverflow(t *testing.T) {
	if math.MaxInt == math.MaxInt32 {
		t.Skip("int cannot exceed the native int32 field on this architecture")
	}
	c := DenoiserConfig{GtcrnModel: writeTempModel(t), NumThreads: math.MaxInt}
	if err := c.Validate(); err == nil {
		t.Fatal("expected error for NumThreads above the native int32 range")
	}
}

func writeTempModel(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "model.onnx")
	if err := os.WriteFile(path, []byte("fake"), 0o600); err != nil {
		t.Fatalf("failed to write temp model file: %v", err)
	}
	return path
}
