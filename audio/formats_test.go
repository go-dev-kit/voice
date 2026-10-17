package audio

import "testing"

func TestGetFormatFromExtension(t *testing.T) {
	tests := []struct {
		name    string
		want    AudioFormat
		wantErr bool
	}{
		{"speech.wav", FormatWAV, false},
		{"speech.WAV", FormatWAV, false},
		{"a.b.flac", FormatFLAC, false},
		{"x.mp3", FormatMP3, false},
		{"x.ogg", FormatOGG, false},
		{"x.m4a", FormatM4A, false},
		{"x.aac", FormatAAC, false},
		{"x.pcm", FormatPCM, false},
		{"x.opus", "", true},
		{"noextension", "", true},
		{"", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetFormatFromExtension(tt.name)
			if (err != nil) != tt.wantErr || got != tt.want {
				t.Errorf("got %q, %v; want %q, error=%v", got, err, tt.want, tt.wantErr)
			}
		})
	}
}

func TestAudioFormatProperties(t *testing.T) {
	tests := []struct {
		format     AudioFormat
		mime, ext  string
		lossy, zip bool
	}{
		{FormatWAV, "audio/wav", ".wav", false, false},
		{FormatFLAC, "audio/flac", ".flac", false, true},
		{FormatMP3, "audio/mpeg", ".mp3", true, true},
		{FormatOGG, "audio/ogg", ".ogg", true, true},
		{FormatM4A, "audio/mp4", ".m4a", true, true},
		{FormatAAC, "audio/aac", ".aac", true, true},
		{FormatPCM, "audio/pcm", ".pcm", false, false},
		{AudioFormat("opus"), "application/octet-stream", ".bin", false, false},
	}
	for _, tt := range tests {
		t.Run(string(tt.format), func(t *testing.T) {
			if got := tt.format.GetMimeType(); got != tt.mime {
				t.Errorf("mime = %q, want %q", got, tt.mime)
			}
			if got := tt.format.GetFileExtension(); got != tt.ext {
				t.Errorf("extension = %q, want %q", got, tt.ext)
			}
			if got := tt.format.IsLossy(); got != tt.lossy {
				t.Errorf("IsLossy = %v, want %v", got, tt.lossy)
			}
			if got := tt.format.IsCompressed(); got != tt.zip {
				t.Errorf("IsCompressed = %v, want %v", got, tt.zip)
			}
		})
	}
}

func TestDefaultConfigsAreValid(t *testing.T) {
	// M4A is a known format but not a supported one, so its default config
	// must fail validation rather than be accepted and fail later.
	for name, cfg := range map[string]*AudioConfig{
		"wav": DefaultWAVConfig(), "flac": DefaultFLACConfig(), "mp3": DefaultMP3Config(), "ogg": DefaultOGGConfig(),
	} {
		if err := cfg.Validate(); err != nil {
			t.Errorf("%s default config invalid: %v", name, err)
		}
	}
	if err := DefaultM4AConfig().Validate(); err == nil {
		t.Error("M4A default config should be rejected as unsupported")
	}
}

func TestAudioConfigValidateRejects(t *testing.T) {
	base := func() AudioConfig { return *DefaultWAVConfig() }
	tests := []struct {
		name   string
		mutate func(*AudioConfig)
	}{
		{"unsupported format", func(c *AudioConfig) { c.Format = FormatAAC }},
		{"zero rate", func(c *AudioConfig) { c.SampleRate = 0 }},
		{"rate below 8 kHz", func(c *AudioConfig) { c.SampleRate = 4000 }},
		{"rate above 192 kHz", func(c *AudioConfig) { c.SampleRate = 384000 }},
		{"zero channels", func(c *AudioConfig) { c.Channels = 0 }},
		{"nine channels", func(c *AudioConfig) { c.Channels = 9 }},
		{"bad bit depth", func(c *AudioConfig) { c.BitsPerSample = 12 }},
		{"lossy quality too high", func(c *AudioConfig) { *c = *DefaultMP3Config(); c.Quality = 101 }},
		{"lossy quality negative", func(c *AudioConfig) { *c = *DefaultMP3Config(); c.Quality = -1 }},
		{"negative bitrate", func(c *AudioConfig) { *c = *DefaultMP3Config(); c.Bitrate = -1 }},
		{"bitrate too low", func(c *AudioConfig) { *c = *DefaultMP3Config(); c.Bitrate = 16 }},
		{"bitrate too high", func(c *AudioConfig) { *c = *DefaultMP3Config(); c.Bitrate = 512 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := base()
			tt.mutate(&cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatal("expected a validation error")
			}
		})
	}
}

func TestAudioConfigSizes(t *testing.T) {
	cfg := &AudioConfig{Format: FormatWAV, SampleRate: 16000, Channels: 2, BitsPerSample: 16}
	if got := cfg.GetBytesPerSample(); got != 2 {
		t.Errorf("GetBytesPerSample = %d", got)
	}
	if got := cfg.GetBytesPerSecond(); got != 64000 {
		t.Errorf("GetBytesPerSecond = %d", got)
	}
	if got := cfg.EstimateFileSize(2); got != 128000 {
		t.Errorf("EstimateFileSize(WAV) = %d", got)
	}
	mp3 := DefaultMP3Config() // 128 kbps
	if got := mp3.EstimateFileSize(10); got != 160000 {
		t.Errorf("EstimateFileSize(MP3) = %d, want 160000", got)
	}
}
