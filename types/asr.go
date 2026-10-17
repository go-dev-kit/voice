package types

import (
	"context"
	"time"
)

// ASRService defines the interface for ASR (Automatic Speech Recognition) services
type ASRService interface {
	// ProcessAudioChunk processes a chunk of audio for streaming ASR
	ProcessAudioChunk(ctx context.Context, sessionID string, audio []float32) (*Transcription, error)

	// FinishStream finalizes the streaming session identified by sessionID,
	// signaling that no more input will arrive, flushing the recognizer's
	// internal feature buffers, and returning a non-partial transcription.
	// Previously submitted audio is not replayed. Callers invoke it at their own
	// utterance boundary instead of waiting for a VAD endpoint. Finalizing a
	// session that does not exist or has no accepted speech yields an empty,
	// non-partial transcription rather than an error.
	FinishStream(ctx context.Context, sessionID string) (*Transcription, error)

	// RegisterModel registers a new ASR model
	RegisterModel(model ASRModel) error

	// GetModel returns a model by name
	GetModel(name string) (ASRModel, bool)

	// SelectModel selects the best model for given requirements
	SelectModel(language string, requirements *ModelRequirements) (ASRModel, error)

	// Close releases all resources
	Close() error
}

// ASRModel represents an ASR model interface
type ASRModel interface {
	Name() string
	Language() string
	Quantization() string
	ProcessAudio(ctx context.Context, audio []float32, state *StreamingState) (*Transcription, error)
	SupportsLanguage(lang string) bool
	Latency() time.Duration
	Close() error
}

// Transcriber defines batch/offline transcription over complete audio inputs.
type Transcriber interface {
	Transcribe(ctx context.Context, audio []float32, sampleRate int) (*Transcription, error)
	Close() error
}

// Transcription represents a transcription result
type Transcription struct {
	Text       string        `json:"text"`
	IsPartial  bool          `json:"is_partial"`
	Confidence float64       `json:"confidence"`
	Language   string        `json:"language"`
	Emotion    string        `json:"emotion,omitempty"`
	Event      string        `json:"event,omitempty"`
	Timestamp  time.Time     `json:"timestamp"`
	Tokens     []Token       `json:"tokens,omitempty"`
	Words      []Word        `json:"words,omitempty"`
	StartTime  time.Duration `json:"start_time,omitempty"`
	EndTime    time.Duration `json:"end_time,omitempty"`
	// Segments is set by transcribers that decode audio in several pieces,
	// with times relative to the start of the input. When there is more than
	// one, Tokens may be partial: a piece whose overlap with a neighbor was
	// stitched by text rather than token timing contributes no tokens, so
	// Tokens never repeats a word. Text and Segments are always complete.
	Segments []TranscriptSegment `json:"segments,omitempty"`
}

// TranscriptSegment is the text of one decoded piece of a longer input.
type TranscriptSegment struct {
	Text      string        `json:"text"`
	StartTime time.Duration `json:"start_time"`
	EndTime   time.Duration `json:"end_time"`
}

// Token is a model-native ASR token with optional timing. Tokens may be
// characters, subwords, or complete words depending on the recognizer and its
// vocabulary; callers must not treat them as word segmentation. HasTiming
// distinguishes a real zero timestamp from a recognizer that exposed no token
// timing. A zero Duration means no token duration was exposed.
type Token struct {
	Text      string        `json:"text"`
	HasTiming bool          `json:"has_timing"`
	StartTime time.Duration `json:"start_time"`
	Duration  time.Duration `json:"duration,omitempty"`
}

// Word represents a word-level transcription with timing
type Word struct {
	Text       string        `json:"text"`
	StartTime  time.Duration `json:"start_time"`
	EndTime    time.Duration `json:"end_time"`
	Confidence float64       `json:"confidence"`
}

// StreamingState represents the state of a streaming session
type StreamingState struct {
	SessionID      string
	Language       string
	Buffer         AudioBufferInterface
	VADState       any
	ASRState       any
	LastActivity   time.Time
	PartialResults []Transcription
}

// AudioBufferInterface defines the interface for audio buffering
type AudioBufferInterface interface {
	Append(audio []float32) error
	GetRecentChunk() []float32
	ReturnBuffer(buf []float32)
	IsReady() bool
	Reset()
	Size() int
	Capacity() int
}

// StreamingConfig holds streaming configuration
type StreamingConfig struct {
	// MaxConcurrentStreams limits admitted active session streams. Final results
	// return their slot; non-positive values use the manager default.
	MaxConcurrentStreams int
	ChunkSize            int
	OverlapSize          int
	BufferSize           int
	SampleRate           int
	StreamTimeout        time.Duration
	IdleTimeout          time.Duration
	FlushInterval        time.Duration
	PartialResults       bool
	StabilityThreshold   float64
	MinConfidence        float64
}

// ModelRequirements defines requirements for model selection
type ModelRequirements struct {
	MaxLatency     time.Duration
	MinAccuracy    float64
	PreferSpeed    bool
	PreferAccuracy bool
}

// VADConfig holds VAD configuration
type VADConfig struct {
	Provider   string  `json:"provider"`
	Threshold  float64 `json:"threshold"`
	WindowSize int     `json:"window_size"`
	SampleRate int     `json:"sample_rate"`
}

// VADResult represents VAD processing result
type VADResult struct {
	IsSpeech   bool
	IsEndpoint bool
	Confidence float64
	State      any
}
