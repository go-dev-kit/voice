# VoiceKit

A reusable Go 1.26.4 library for local voice processing experiments and audio plumbing. VoiceKit now defaults to audio-only initialization; speaker recognition, ASR, VAD, diarization, and TTS are opt-in runtime components that require explicit model configuration.

## Requirements

- Go 1.26.4 or newer 1.26.x toolchain.
- CGO enabled for Sherpa-ONNX-backed ASR, VAD, diarization, speaker embeddings, and TTS.
- Model files supplied by the caller or deployment environment. This repository does not commit model assets.

## Maturity Matrix

| Capability | Status | Runtime requirements |
| --- | --- | --- |
| Audio conversion/resampling | Usable foundation | WAV/PCM encode/decode; FLAC/MP3/Ogg-Vorbis decode via pure-Go libraries. AAC/M4A and compressed encoding return explicit unsupported errors. |
| Speaker recognition | Experimental but real | Sherpa speaker embedding model path and speaker data directory. Speaker embedding streams are single-use and released after extraction. |
| ASR | Long-audio-safe Sherpa offline transcriber plus bounded Sherpa online streaming service | Offline or online Sherpa model files. Offline input of any length is cut into decodes of at most 25 s (Silero/TEN VAD segmentation when configured), never truncated. Streaming admission enforces `MaxConcurrentStreams`; model-native token timing is exposed separately from real word segmentation. |
| VAD | Runtime seam with explicit providers | `none`, `energy`, or Sherpa Silero/TEN with a configured model path. |
| Diarization | Sherpa offline diarization backend | Segmentation and embedding model paths. The old silence/clustering path is an explicit `basic` backend for tests/local experiments. |
| TTS | Sherpa offline and streaming speech synthesis | VITS, Matcha, Kokoro, KittenTTS, ZipVoice, Pocket, or Supertonic model files. TTS enabled without required backend paths is a validation error. |
| Meeting intelligence | Product-domain foundation | Typed meeting artifacts, deterministic mapping from diarized ASR output, Markdown/JSON/SRT/WebVTT exports, export-time redaction enforcement, provider-neutral analysis, a stdlib-only reference analyzer, and deterministic evaluation metrics. |

For the developer handoff guide covering meeting data flow, analyzer extension
points, redaction/export safety, evaluation metrics, and validation commands,
see [MEETING_INTELLIGENCE.md](MEETING_INTELLIGENCE.md).

### Recommended models (July 2026)

Model files are supplied by the caller; [`testdata/model_matrix.yaml`](testdata/model_matrix.yaml)
records the recommended choices and their sources:

- **Streaming ASR**: NVIDIA Nemotron Speech Streaming 0.6B (RNNT transducer — use
  the online `transducer` family); Zipformer2-CTC / Parakeet as alternatives.
- **Offline ASR**: SenseVoice (multilingual, single-file); Whisper for reference.
- **VAD**: Silero VAD v6 (MIT) by default; TEN VAD is opt-in pending a license review.
- **Diarization**: pyannote segmentation-3.0 (or 4.0 community-1) + 3D-Speaker CAM++.
- **TTS**: Kokoro-82M as the efficiency default.

Runs on sherpa-onnx-go v1.13.8 (ONNX Runtime 1.28.2). Local and CI verification use the Go 1.26.4 module floor.

## Installation

Add the released module to your project:

```bash
go get go.glpx.pro/gdk/voice@latest
```

## Architecture Overview

VoiceKit implements a modular, layered architecture for voice processing operations:

### Core Architecture Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      VoiceKit (Unified API)                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 Component Managers                       │    │
│  │  ┌─────────────┬─────────────┬─────────────┬───────┐   │    │
│  │  │  ASR        │ Audio       │ Speaker     │ Dia-  │   │    │
│  │  │ Streaming   │ Processing  │ Recognition │ riza- │   │    │
│  │  │             │             │             │ tion  │   │    │
│  │  └─────────────┴─────────────┴─────────────┴───────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                       │
                         ┌─────────────┼─────────────┐
                         │             │             │
                 ┌───────▼────┐ ┌──────▼────┐ ┌─────▼─────┐
                 │   ASR      │ │   Audio    │ │   Speaker  │
                 │ Streaming  │ │ Processing│ │ Recognition│
                 │ Engine     │ │ Primitives│ │ Engine     │
                 └────────────┘ └────────────┘ └────────────┘
                         │             │             │
                 ┌───────▼────┐ ┌──────▼────┐ ┌─────▼─────┐
                 │   VAD      │ │   Resam-  │ │   Sherpa-  │
                 │ Processing │ │   pling   │ │   ONNX     │
                 └────────────┘ └────────────┘ └────────────┘
```

### Component Responsibilities

- **VoiceKit (Top Level)**: Orchestrates component interactions, provides unified API
- **ASR**: Offline transcription through `Transcriber` and real-time recognition through `ASRService`
- **Audio Processing**: Handles sample rate conversion, format transcoding, normalization
- **Speaker Recognition**: Manages speaker embeddings, similarity computation, database operations
- **Diarization**: Adapts Sherpa offline speaker diarization into VoiceKit result types
- **TTS**: Synthesizes complete text inputs to normalized PCM samples through Sherpa offline TTS
- **Meeting Intelligence**: Maps diarized ASR results into meeting artifacts, exports notes/captions, redacts sensitive text before export when required, validates provider-neutral analyzer output, and measures deterministic meeting-quality metrics
- **VAD Processing**: Voice activity detection for intelligent audio segmentation

### Technical Approaches

#### Audio Processing Architecture
- **Resampling**: Implements multiple interpolation algorithms (Linear, Cubic, Lanczos, Sinc)
- **Format Conversion**: Uses streaming approach with configurable buffer sizes
- **Channel Handling**: Supports mono↔stereo conversion with proper downmixing/upmixing
- **Memory Management**: Implements object pooling for float32 slices to reduce GC pressure

#### Speaker Recognition Architecture
- **Embedding Extraction**: Leverages Sherpa-ONNX neural network inference
- **Native Stream Lifecycle**: Creates a fresh embedding stream per extraction and deletes it after `InputFinished`
- **Similarity Computation**: Uses cosine similarity on normalized embedding vectors
- **Database Design**: JSON-based persistence with in-memory caching for fast lookups
- **Thread Safety**: RWMutex-based concurrent access to speaker database

#### ASR Architecture
- **Offline Transcription**: `asr.SherpaOfflineModel` maps complete audio inputs to `types.Transcription`
- **Streaming Lifecycle**: Sherpa online recognition preserves bounded VAD pre-roll, sends each utterance sample to one session-owned native stream exactly once, calls `InputFinished` without replaying accepted audio, and deletes stream state on finalization, processing failure, session cleanup, or service close
- **Model Registry**: Dynamic model selection based on language and performance requirements
- **Audio Buffering**: A bounded rolling buffer remains for compatibility with custom non-finalizable models; native online recognition consumes caller chunks directly
- **VAD Integration**: Explicit `none`, `energy`, Sherpa Silero, and Sherpa TEN providers, with one stateful detector per streaming session and bounded pre-roll retained until speech activation
- **Bounded Admission**: `MaxConcurrentStreams` limits active session streams; finalization returns the slot and capacity failures remain discoverable as `*asr.StreamCapacityError` through `errors.As`
- **Timing Contract**: Sherpa tokens populate `Transcription.Tokens`; `Transcription.Words` is reserved for output that has undergone real word segmentation
- **Test/Demo Fakes**: Fake ASR models are available only through explicit registration

#### Diarization Architecture
- **Backend Boundary**: `diarization.Backend` owns native processing and lifecycle
- **Sherpa Offline Backend**: Uses Sherpa segmentation, embedding, clustering, and min-duration settings
- **Basic Backend**: Keeps the old silence-plus-embedding clustering path available only by explicit `BackendBasic`
- **Integration**: Time-based alignment between audio segments and genuine word timestamps when an upstream recognizer or segmenter provides them

#### TTS Architecture
- **Offline Synthesis**: `tts.SherpaOfflineSynthesizer` maps text requests to Sherpa-generated float32 PCM samples
- **Family-Specific Config**: Supported Sherpa families have explicit config blocks instead of one flat path bag
- **Resource Lifecycle**: Native TTS resources are owned by the synthesizer and closed through `VoiceKit.Close`

#### Streaming and Auxiliary Capabilities

Optional Sherpa-backed capabilities that wrap binding surfaces beyond the core
services. Each has a provider-neutral interface in `types`, a standalone
constructor re-exported at the root, and an env-gated example under `examples/`.

- **Streaming ASR finalization**: `ASRService.FinishStream(ctx, sessionID)` forces
  a final hypothesis (native `InputFinished` + flush-decode) at a caller-chosen
  utterance boundary instead of waiting for a VAD endpoint.
- **Streaming admission and timing**: new active sessions are bounded by
  `MaxConcurrentStreams`; Sherpa token text/timing is exposed through
  `Transcription.Tokens` without labeling subword tokens as words.
- **Streaming TTS**: `tts.SherpaStreamingSynthesizer` (`voicekit.NewStreamingSynthesizer`)
  implements `types.StreamingSynthesizer`, delivering audio chunks via a sink
  callback (return `false` to interrupt) with progress. Example: `examples/tts_streaming`.
- **Punctuation restoration**: `asr.SherpaPunctuation` (`voicekit.NewPunctuation`)
  implements `types.Punctuation`, a post-ASR normalizer over Sherpa OfflinePunctuation.
  Example: `examples/punctuation`.
- **Keyword spotting**: `asr.SherpaKeywordSpotter` (`voicekit.NewKeywordSpotter`)
  implements `types.KeywordSpotter` with per-session streaming detection.
  Example: `examples/keyword_spotting`.
- **Spoken language identification**: `asr.SherpaLanguageIdentifier`
  (`voicekit.NewLanguageIdentifier`) implements `types.LanguageIdentifier` over a
  complete audio segment. Example: `examples/language_id`.
- **Provider aliasing / validation**: VAD provider names accept aliases
  (`silero`→`silero_vad`, `ten`→`ten_vad`); multilingual Kokoro requires an explicit
  `lang` when a lexicon is set. Unknown providers still fail loudly.

## Technical Implementation Details

### Memory Management Strategy

#### Object Pooling
```go
// Audio processing uses object pooling to reduce GC overhead
type float32Pool struct {
    pool sync.Pool
}

func (p *float32Pool) Get(size int) []float32 {
    // Returns pooled slice or allocates new one
}

func (p *float32Pool) Put(slice []float32) {
    // Returns slice to pool for reuse
}
```

#### Buffer Management
- **Streaming Processing**: Audio data processed in configurable chunks (default 8KB)
- **Zero-Copy Operations**: Where possible, slices reference original data
- **Pre-allocated Buffers**: Reuse buffers for resampling and conversion operations

### Performance Optimizations

#### Audio Processing
- **SIMD-Compatible**: Algorithms designed for potential SIMD vectorization
- **Cache-Friendly**: Sequential memory access patterns in resampling loops
- **Branch Prediction**: Algorithms structured to minimize branch mispredictions

#### Speaker Recognition
- **Batch Processing**: Sherpa-ONNX processes multiple audio segments efficiently
- **Embedding Caching**: Speaker embeddings cached in memory for fast similarity searches
- **Similarity Optimization**: Early termination in similarity searches using thresholds

#### Diarization
- **Incremental Processing**: Segments processed as they become available
- **Parallel Clustering**: Potential for parallel clustering of independent segments
- **Lazy Evaluation**: Results computed only when requested

### Threading and Concurrency

#### Synchronization Strategy
- **RWMutex**: Reader-writer locks for database operations (multiple readers, exclusive writer)
- **Atomic Operations**: Used for statistics counters in high-throughput scenarios
- **Channel-Based**: Internal communication between processing stages

#### Goroutine Safety
- **Stateless Processing**: Audio processing functions are stateless and goroutine-safe
- **Immutable Results**: Result structures are immutable after creation
- **Interface-Based**: Dependency injection allows for safe concurrent testing

### Error Handling Architecture

#### Error Types
```go
// Structured error types with context
type AudioError struct {
    Op   string    // Operation that failed
    Format string  // Audio format involved
    Err  error     // Underlying error
}

type SpeakerError struct {
    Op        string // Operation
    SpeakerID string // Speaker involved
    Err       error  // Underlying error
}
```

#### Error Propagation
- **Wrapped Errors**: All errors implement `Unwrap()` for error chain inspection
- **Context Preservation**: Errors maintain operation context for debugging
- **Graceful Degradation**: Components fail independently without cascading failures

### Integration Patterns

#### ASR Integration
- **Word-Level Alignment**: Uses timestamp information for precise speaker-text alignment
- **Confidence Propagation**: Preserves confidence fields when upstream systems provide them; Sherpa ASR confidence is `0` when not exposed by the binding
- **Metadata Preservation**: Maintains original ASR metadata alongside speaker information

#### WebSocket Streaming
- **Real-time Processing**: Designed for streaming audio with low latency
- **Stateful Sessions**: Maintains per-session state for continuous processing
- **Message Batching**: Groups related results for efficient network transmission

### Extension Points

#### Custom Speaker Backends
```go
type SpeakerDatabase interface {
    GetSpeakerEmbedding(speakerID string) ([]float32, error)
    CalculateSimilarity(emb1, emb2 []float32) float32
    RegisterSpeakerEmbedding(id string, emb []float32) error
}
```

#### Custom Audio Processors
- **Resampler Interface**: Pluggable resampling algorithms
- **Format Handlers**: Extensible audio format support
- **Normalizer Interface**: Custom normalization strategies

#### Logger Integration
```go
type Logger interface {
    Infof(format string, args ...interface{})
    Warnf(format string, args ...interface{})
    Errorf(format string, args ...interface{})
}
```

## Dependencies

### Core Dependencies

#### Sherpa-ONNX (github.com/k2-fsa/sherpa-onnx-go)
- **Role**: Neural network inference for speaker embedding extraction
- **Architecture**: ONNX Runtime integration with platform-specific binaries
- **Performance**: GPU acceleration support, optimized for real-time processing
- **Memory**: Models loaded once, shared across processing sessions
- **Threading**: Thread-safe inference with configurable thread pools

#### WAV Library (github.com/go-audio/wav)
- **Role**: WAV file parsing and PCM data extraction
- **Implementation**: Streaming decoder with minimal memory footprint
- **Compatibility**: Supports various WAV formats (mono/stereo, different bit depths)

### Internal Dependencies

#### Logger Interface
- **Purpose**: Decoupled logging allows integration with various logging frameworks
- **Default**: No-op implementation for minimal dependency footprint
- **Integration**: Compatible with logrus, zap, standard library log, etc.

#### Configuration System
- **Approach**: Struct-based configuration with validation
- **Defaults**: Sensible defaults for all parameters
- **Validation**: Compile-time and runtime configuration validation

## Performance Characteristics

### Latency Breakdown
- **Audio Conversion**: O(n) where n is sample count, dominated by resampling
- **Speaker Recognition**: O(m*k) where m is speakers, k is embedding dimension
- **Diarization**: O(s²) where s is number of segments (clustering complexity)
- **ASR Streaming**: Depends on the configured Sherpa model, provider, hardware, and chunk size

### Memory Usage
- **Audio Buffers**: Proportional to audio duration and sample rate
- **Speaker Database**: Scales with number of enrolled speakers
- **Embedding Cache**: Fixed size per speaker (typically 192 floats)
- **ASR Models**: Native model memory depends on the configured Sherpa model family, provider, and quantization; model assets are not bundled

### Measured Local Baseline
- **Current baseline**: Go 1.26.4, darwin/arm64, Apple M2, local CPU benchmarks.
- **Latest measured wins**: WAV/PCM decode and basic diarization allocation paths were optimized in July 2026; see `PERFORMANCE_IMPROVEMENTS.md` for the benchstat table and exact workflow.
- **Benchmark artifacts**: Write ad hoc before/after files outside tracked source, for example `/tmp/voicekit-benchmarks`.

### Scalability Considerations
- **Horizontal**: Stateless audio processing scales linearly
- **Vertical**: Speaker database operations bottleneck at high concurrency
- **Memory**: Bounded by speaker database size and audio buffer pools

## Design Decisions and Trade-offs

### Modularity vs Performance
- **Decision**: Interface-based design prioritizes modularity over micro-optimizations
- **Trade-off**: Virtual function calls vs direct function calls
- **Mitigation**: Inlining and interface optimizations in Go compiler

### Memory vs Speed
- **Decision**: Object pooling reduces GC pressure but increases memory usage
- **Trade-off**: Higher baseline memory usage for lower latency
- **Rationale**: Voice processing often requires consistent low latency

### Simplicity vs Features
- **Decision**: Core functionality only, advanced features via extension
- **Trade-off**: Smaller API surface vs comprehensive feature set
- **Benefit**: Easier testing, maintenance, and integration

### Platform Compatibility
- **Decision**: Pure Go implementation where possible, CGO only for performance-critical code
- **Trade-off**: Development complexity vs runtime performance
- **Implementation**: Sherpa-ONNX provides cross-platform CGO bindings

## Limitations and Known Issues

### Audio Format Support
- **Compressed Formats**: FLAC, MP3, and Ogg/Vorbis **decoding** are supported via pure-Go libraries (no CGO); AAC/M4A decoding and all compressed **encoding** are not implemented and return explicit unsupported-format errors
- **Metadata**: Format-specific metadata (tags, etc.) not preserved
- **Sample Rates**: Limited testing on extreme sample rates (>192kHz, <8kHz)

### Speaker Recognition
- **Model Dependencies**: Requires specific Sherpa-ONNX model files
- **Embedding Quality**: Performance depends on training data quality
- **Real-time Constraints**: Embedding extraction is single-use stream based until model-backed benchmarks prove a safe pooling strategy

### Diarization
- **Model Dependencies**: Production diarization requires Sherpa segmentation and embedding model files
- **Speaker Count**: Clustering quality depends on configured threshold or exact cluster count
- **Native Runtime**: Sherpa calls are mutex-protected until concurrency benchmarks prove shared native use is safe

### ASR
- **Model Dependencies**: Requires Sherpa offline or online model files, depending on `ASR.Backend`
- **CGO Requirements**: Uses Sherpa ONNX native bindings; CGO must be enabled
- **Runtime Provider**: CPU is the default; CUDA/CoreML availability depends on the Sherpa build and platform
- **Language Support**: Determined by the configured Sherpa model
- **Streaming Finalization**: `ProcessAudioChunk` returns partial results until native endpoint detection or VAD finalization produces a final result
- **Long Offline Audio**: Sherpa offline Whisper keeps only the first 30 s of one decode and NeMo Parakeet exports fail from about 30 s, while Silero's `max_speech_duration` does not cap segment length. `asr.NewOfflineTranscriber` (used by `VoiceKit.Transcriber()`) therefore never hands a decoder more than `asr.MaxDecodeDuration` (25 s): it segments with VAD, merges short spans, and cuts longer spans at the quietest point with a 2 s overlap. A bare `SherpaOfflineModel` refuses over-long input with `*asr.AudioTooLongError` (30 s for Whisper by default, `Offline.MaxDecodeSeconds` otherwise) instead of truncating
- **Offline Concurrency**: one `SherpaOfflineModel` decodes one request at a time unless `Offline.MaxConcurrentDecodes` allows more on the shared native recognizer (verified to give identical text for Parakeet TDT and Whisper); queued callers leave the queue when their context ends, and `Close` waits for in-flight decodes

### TTS
- **Model Dependencies**: Requires Sherpa offline TTS model files for the configured family
- **Output Format**: `SynthesizedSpeech` returns normalized float32 samples; callers own encoding to WAV/PCM containers
- **Voice Selection**: Speaker IDs are validated against the native model-reported speaker count when available

### Meeting Intelligence
- **Reference Analyzer**: `meeting.HeuristicAnalyzer` is a deterministic baseline for local examples and tests, not an LLM-quality summarizer.
- **Redaction Scope**: The built-in redactor covers emails, phone-like numbers, URLs, and configured literal terms. It is not a complete privacy, consent, retention, or compliance system.
- **Evaluation Scope**: The `evaluation` package provides deterministic WER/CER, speaker attribution, action-item, real-time-factor, and diarization-error-rate metrics. It does not yet include model-backed benchmark corpora or hosted-provider comparisons.

### Memory Management
- **Large Files**: Memory usage scales linearly with audio duration
- **Long Sessions**: Set `Speaker.MaxSpeakers` to bound the database; new registrations beyond the cap evict the oldest speakers (by `CreatedAt`). Left at 0, the database grows without automatic cleanup.
- **Goroutine Leaks**: Improper cleanup may leave goroutines running

## Future Enhancements

### Planned Improvements
- **GPU Acceleration**: Enhanced CUDA/OpenCL integration for neural network inference
- **Diarization Evaluation**: DER-focused benchmarks across Sherpa diarization models and thresholds
- **TTS Evaluation**: MOS-style and latency benchmarks across Sherpa TTS model families
- **Two-Pass Decoding**: CTC + attention for improved streaming accuracy
- **WebRTC Integration**: Native WebRTC support for ultra-low latency streaming
- **Model Quantization**: 8-bit quantization for reduced memory footprint

### Extension Points
- **Custom Models**: Support for different embedding model architectures
- **Plugin System**: Dynamic loading of audio processing plugins
- **Cloud Integration**: Backend support for cloud-based processing

## Quick Start

### Basic Usage

```go
package main

import (
    "fmt"
    "log"

    "go.glpx.pro/gdk/voice"
    "go.glpx.pro/gdk/voice/audio"
    "go.glpx.pro/gdk/voice/speaker"
)

func main() {
    // Create a simple logger
    logger := &simpleLogger{}

    // Configure VoiceKit
    config := &voicekit.Config{
        Audio: voicekit.AudioConfig{
            SampleRate:      16000,
            Channels:        1,
            NormalizeFactor: 32768.0,
        },
        Speaker: voicekit.SpeakerConfig{
            ModelPath:  "/path/to/speaker/model",
            DataDir:    "/path/to/speaker/data",
            Threshold:  0.5,
            Logger:     logger,
        },
        Diarization: voicekit.DiarizationConfig{
            Enabled: true,
            Logger:  logger,
        },
    }

    // Create VoiceKit instance
    vk, err := voicekit.NewVoiceKit(config)
    if err != nil {
        log.Fatal(err)
    }
    defer vk.Close()

    // Load audio data (example: WAV file)
    audioData := loadAudioFile("example.wav")
    inputConfig := audio.DefaultWAVConfig()

    // Process audio through the full pipeline
    speakerResult, diarizationResult, err := vk.ProcessAudio(audioData, inputConfig, "session-123")
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    if speakerResult != nil && speakerResult.Identified {
        fmt.Printf("Identified speaker: %s (%s)\n",
            speakerResult.SpeakerName, speakerResult.SpeakerID)
    }

    if diarizationResult != nil {
        fmt.Printf("Found %d speakers in %d segments\n",
            diarizationResult.SpeakerCount, len(diarizationResult.Segments))
    }
}
```

### ASR Streaming

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "log"
    "time"

    "go.glpx.pro/gdk/voice"
    voiceasr "go.glpx.pro/gdk/voice/asr"
)

func main() {
    // Configure VoiceKit with Sherpa online ASR enabled.
    // Model files are not committed to this repository.
    config := &voicekit.Config{
        ASR: voicekit.ASRConfig{
            Enabled:              true,
            Backend:              "sherpa_online",
            DefaultModel:         "sherpa_online",
            Language:             "en",
            Quantization:         "float32",
            MaxConcurrentStreams: 10,
            StreamTimeout:        300,
            ChunkSize:            16000,
            Online: voicekit.OnlineConfig{
                TokensPath:  "/models/sherpa/tokens.txt",
                EncoderPath: "/models/sherpa/encoder.onnx",
                DecoderPath: "/models/sherpa/decoder.onnx",
                JoinerPath:  "/models/sherpa/joiner.onnx",
            },
            VADProvider:          "none",
        },
    }

    // Create VoiceKit instance
    vk, err := voicekit.NewVoiceKit(config)
    if err != nil {
        log.Fatal(err)
    }
    defer vk.Close()

    // Get ASR service
    asrService := vk.ASR()

    // Simulate streaming audio chunks
    sessionID := "stream-session-1"
    ctx := context.Background()

    // Process audio chunks in a loop (simulate real-time streaming)
    for i := 0; i < 10; i++ {
        // Generate or load audio chunk (1 second at 16kHz)
        audioChunk := generateAudioChunk() // []float32 with 16000 samples

        // Process chunk through ASR
        transcription, err := asrService.ProcessAudioChunk(ctx, sessionID, audioChunk)
        if err != nil {
            var capacityErr *voiceasr.StreamCapacityError
            if errors.As(err, &capacityErr) {
                log.Printf("ASR capacity reached (%d/%d)", capacityErr.Active, capacityErr.Limit)
                break
            }
            log.Printf("ASR processing error: %v", err)
            continue
        }

        // Handle transcription results
        if transcription != nil {
            if transcription.IsPartial {
                fmt.Printf("Partial: %s\n", transcription.Text)
            } else {
                fmt.Printf("Final: %s (confidence: %.2f)\n",
                    transcription.Text, transcription.Confidence)
            }
        }

        // Simulate real-time delay
        time.Sleep(100 * time.Millisecond)
    }

    final, err := asrService.FinishStream(ctx, sessionID)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Final: %s\n", final.Text)
    fmt.Println("ASR streaming session completed")
}

func generateAudioChunk() []float32 {
    // Generate a 1-second audio chunk (16kHz = 16000 samples)
    chunk := make([]float32, 16000)
    // Fill with actual audio data or synthesized speech
    return chunk
}
```

### ASR Offline

```go
package main

import (
    "context"
    "log"
    "os"
    "time"

    "go.glpx.pro/gdk/voice"
    "go.glpx.pro/gdk/voice/audio"
)

func main() {
    config := &voicekit.Config{
        ASR: voicekit.ASRConfig{
            Enabled:      true,
            Backend:      "sherpa_offline",
            DefaultModel: "sherpa_offline",
            Language:     "en",
            Offline: voicekit.OfflineConfig{
                ModelFamily: "transducer",
                ModelType:   "nemo_transducer", // Parakeet TDT
                TokensPath:  "/models/parakeet/tokens.txt",
                EncoderPath: "/models/parakeet/encoder.int8.onnx",
                DecoderPath: "/models/parakeet/decoder.int8.onnx",
                JoinerPath:  "/models/parakeet/joiner.int8.onnx",
                Language:    "en",
            },
            // Optional: decode only the speech. Without a VAD the whole
            // input is still cut into decodes of at most 25 s.
            VADProvider:  "silero_vad",
            VADModelPath: "/models/silero_vad.onnx",
        },
    }

    vk, err := voicekit.NewVoiceKit(config)
    if err != nil {
        log.Fatal(err)
    }
    defer vk.Close()

    // Any length works; result.Segments holds one entry per decode.
    data, _ := os.ReadFile("meeting.mp3")
    samples, err := audio.DecodeMono(data, audio.FormatMP3, 16000, audio.DecodeLimits{MaxDuration: 4 * time.Hour})
    if err != nil {
        log.Fatal(err)
    }
    result, err := vk.Transcriber().Transcribe(context.Background(), samples, 16000)
    if err != nil {
        log.Fatal(err)
    }
    log.Println(result.Text)
}
```

### TTS Offline

```go
package main

import (
    "context"
    "log"

    "go.glpx.pro/gdk/voice"
)

func main() {
    config := &voicekit.Config{
        TTS: voicekit.TTSConfig{
            Enabled:     true,
            Backend:     "sherpa_offline",
            ModelFamily: "kokoro",
            Kokoro: voicekit.TTSKokoroConfig{
                Model:   "/models/sherpa/kokoro/model.onnx",
                Voices:  "/models/sherpa/kokoro/voices.bin",
                Tokens:  "/models/sherpa/kokoro/tokens.txt",
                DataDir: "/models/sherpa/kokoro/espeak-ng-data",
            },
        },
    }

    vk, err := voicekit.NewVoiceKit(config)
    if err != nil {
        log.Fatal(err)
    }
    defer vk.Close()

    result, err := vk.Synthesizer().Synthesize(context.Background(), voicekit.SynthesisRequest{
        Text: "VoiceKit can synthesize speech locally.",
    })
    if err != nil {
        log.Fatal(err)
    }
    log.Println(result.SampleRate, len(result.Samples), result.Duration)
}
```

### Speaker Recognition Only

```go
package main

import (
    "go.glpx.pro/gdk/voice/speaker"
)

func main() {
    // Create speaker manager
    config := &speaker.Config{
        ModelPath: "/path/to/speaker/model",
        DataDir:   "/path/to/speaker/data",
        Threshold: 0.5,
    }

    manager, err := speaker.NewManager(config)
    if err != nil {
        panic(err)
    }
    defer manager.Close()

    // Register a speaker
    audioData := loadAudioData() // []float32
    err = manager.RegisterSpeaker("john_doe", "John Doe", audioData, 16000)
    if err != nil {
        panic(err)
    }

    // Identify a speaker
    unknownAudio := loadUnknownAudio() // []float32
    result, err := manager.IdentifySpeaker(unknownAudio, 16000)
    if err != nil {
        panic(err)
    }

    if result.Identified {
        println("Identified:", result.SpeakerName)
    } else {
        println("Speaker not recognized")
    }
}
```

### Audio Processing Only

```go
package main

import (
    "go.glpx.pro/gdk/voice/audio"
)

func main() {
    // Create audio converter
    config := audio.DefaultConverterConfig()
    converter := audio.NewConverter(config)

    // Convert audio format
    inputData := loadAudioBytes()
    inputConfig := audio.DefaultWAVConfig()
    outputConfig := audio.DefaultMP3Config()

    convertedData, err := converter.ConvertAudio(inputData, inputConfig, outputConfig)
    if err != nil {
        panic(err)
    }

    // Create resampler
    resampleConfig := &audio.ResampleConfig{
        Method: audio.MethodCubic,
        Quality: audio.QualityMedium,
    }
    resampler := audio.NewResampler(resampleConfig)

    // Resample audio
    audioSamples := loadFloat32Samples()
    resampled, err := resampler.Resample(audioSamples, 44100, 16000)
    if err != nil {
        panic(err)
    }

    println("Resampled from 44100Hz to 16000Hz")
}
```

### Diarization Only

```go
package main

import (
    "go.glpx.pro/gdk/voice/diarization"
)

func main() {
    config := diarization.DefaultDiarizationConfig()
    config.Enabled = true
    config.Backend = diarization.BackendSherpaOffline
    config.SegmentationModelPath = "/models/sherpa/segmentation.onnx"
    config.EmbeddingModelPath = "/models/sherpa/embedding.onnx"

    manager, err := diarization.NewManager(config, nil)
    if err != nil {
        panic(err)
    }
    defer manager.Close()

    // Process audio for diarization
    audioData := loadAudioSamples() // []float32
    result, err := manager.ProcessAudio(audioData, 16000, "session-123")
    if err != nil {
        panic(err)
    }

    println("Found", result.SpeakerCount, "speakers")
    for _, segment := range result.Segments {
        println("Speaker", segment.SpeakerID, "speaks from",
            segment.StartTime, "to", segment.EndTime)
    }
}
```

## API Reference

### VoiceKit (Unified Interface)

#### `NewVoiceKit(config *Config) (*VoiceKit, error)`
Creates a new VoiceKit instance with audio initialized and optional runtime components initialized only when enabled/configured.

#### `(*VoiceKit) Close() error`
Releases all resources held by VoiceKit.

#### `(*VoiceKit) Speaker() *speaker.Manager`
Returns the speaker recognition manager, or nil when speaker recognition is not configured.

#### `(*VoiceKit) Audio() *audio.Converter`
Returns the audio converter.

#### `(*VoiceKit) Diarization() *diarization.Manager`
Returns the diarization manager, or nil when diarization is disabled.

#### `(*VoiceKit) ASR() ASRService`
Returns the ASR service for real-time speech recognition, or nil when ASR is disabled or configured for offline transcription.

#### `(*VoiceKit) Transcriber() Transcriber`
Returns the batch/offline ASR transcriber, or nil when ASR is disabled or configured for streaming.

#### `(*VoiceKit) Synthesizer() SpeechSynthesizer`
Returns the text-to-speech synthesizer, or nil when TTS is disabled.

#### `(*VoiceKit) ProcessAudio(audioData []byte, inputConfig *audio.AudioConfig, sessionID string) (*speaker.IdentifyResult, *diarization.DiarizationResult, error)`
Always converts audio; runs speaker identification and diarization only when those managers are configured.

### Speaker Recognition

#### `speaker.NewManager(config *Config) (*Manager, error)`
Creates a new speaker recognition manager.

#### `(*Manager) RegisterSpeaker(speakerID, speakerName string, audioData []float32, sampleRate int) error`
Registers a new speaker with audio sample.

#### `(*Manager) IdentifySpeaker(audioData []float32, sampleRate int) (*IdentifyResult, error)`
Identifies the speaker in the provided audio.

#### `(*Manager) VerifySpeaker(speakerID string, audioData []float32, sampleRate int) (*VerifyResult, error)`
Verifies if the audio belongs to the specified speaker.

#### `(*Manager) GetAllSpeakers() []*SpeakerInfo`
Returns information about all registered speakers.

#### `(*Manager) DeleteSpeaker(speakerID string) error`
Deletes a speaker from the database.

### Audio Processing

#### `audio.DecodeMono(data []byte, format AudioFormat, targetRate int, limits DecodeLimits) ([]float32, error)`
Decodes a complete WAV (8/16/24/32-bit PCM or 32/64-bit float), FLAC, MP3 or Ogg/Vorbis file, averages its channels and resamples it with a band-limited windowed-sinc filter. `DecodeLimits.MaxDuration` stops decoding as soon as the audio passes the limit and returns `*audio.DurationLimitError`, so a small compressed upload cannot expand to gigabytes. `audio.Decode` returns the native rate and channels instead.

#### `audio.NewConverter(config *ConverterConfig) *Converter`
Creates a new audio converter.

#### `(*Converter) ConvertAudio(inputData []byte, inputConfig, outputConfig *AudioConfig) ([]byte, error)`
Converts audio between formats.

#### `(*Converter) ConvertToFloat32(data []byte, config *AudioConfig) ([]float32, error)`
Converts audio data to float32 samples.

#### `audio.NewResampler(config *ResampleConfig) *Resampler`
Creates a new audio resampler.

#### `(*Resampler) Resample(audioData []float32, sourceRate, targetRate int) ([]float32, error)`
Resamples audio to a different sample rate.

### Diarization

#### `diarization.NewManager(config *DiarizationConfig, speakerDB SpeakerDatabase) (*Manager, error)`
Creates a new diarization manager for the configured backend.

#### `diarization.NewManagerWithBackend(config *DiarizationConfig, speakerDB SpeakerDatabase, extractor EmbeddingExtractor) (*Manager, error)`
Creates a new diarization manager with an explicit extractor for the `basic` backend.

#### `(*Manager) ProcessAudio(audioData []float32, sampleRate int, sessionID string) (*DiarizationResult, error)`
Processes audio for speaker diarization.

#### `diarization.NewSegmenter(config *DiarizationConfig) *Segmenter`
Creates a new audio segmenter.

#### `(*Segmenter) SegmentBySilence(audioData []float32, sampleRate int) ([]AudioSegment, error)`
Segments audio based on silence detection.

### Meeting Intelligence

#### `meeting.FromIntegratedResult(result *diarization.IntegratedResult, options meeting.BuildOptions) (*meeting.Meeting, error)`
Maps a typed ASR plus diarization result into a meeting artifact with transcript turns, participant labels, source metadata, and timing validation.

#### `(*meeting.Meeting) Validate() error`
Validates required meeting IDs, participant uniqueness, wall-clock timing, transcript turn timing, and word timing.

#### `(*meeting.Meeting) Duration() time.Duration`
Returns the best-known meeting duration from wall-clock start/end times or transcript turn timing.

#### `(*meeting.Meeting) SpeakerTalkTime() map[string]time.Duration`
Returns total transcript duration by diarization speaker ID.

#### `(*meeting.Meeting) ExportJSON() ([]byte, error)`
Returns stable product-facing JSON with transcript and summary timing represented in seconds.

#### `(*meeting.Meeting) ExportJSONWithOptions(options meeting.ExportOptions) ([]byte, error)`
Returns JSON while applying export safety options such as required prior redaction.

#### `(*meeting.Meeting) ExportMarkdown() (string, error)`
Renders meeting metadata, summary sections, action items, and transcript turns as Markdown notes.

#### `(*meeting.Meeting) ExportMarkdownWithOptions(options meeting.ExportOptions) (string, error)`
Renders Markdown while applying export safety options such as required prior redaction.

#### `(*meeting.Meeting) ExportSRT() (string, error)`
Renders speaker-labeled transcript turns as SubRip subtitles. Caption exports require positive turn timing.

#### `(*meeting.Meeting) ExportSRTWithOptions(options meeting.ExportOptions) (string, error)`
Renders SubRip subtitles while applying export safety options such as required prior redaction.

#### `(*meeting.Meeting) ExportWebVTT() (string, error)`
Renders speaker-labeled transcript turns as WebVTT captions. Caption exports require positive turn timing.

#### `(*meeting.Meeting) ExportWebVTTWithOptions(options meeting.ExportOptions) (string, error)`
Renders WebVTT captions while applying export safety options such as required prior redaction.

#### `meeting.ExportOptions`
Configures export-time safety checks:

```go
type ExportOptions struct {
    RequireRedaction bool
}
```

When `RequireRedaction` is true, export methods return `meeting.ErrRedactionRequired` unless the meeting was produced by `Meeting.Redact`.

#### `meeting.Redactor`
Defines the redaction boundary used before export:

```go
type Redactor interface {
    Redact(field string, text string) (string, []RedactionMatch, error)
}
```

#### `meeting.DefaultRedactionPolicy()` and `meeting.NewPatternRedactor(policy meeting.RedactionPolicy)`
Create the built-in stdlib redactor for emails, phone-like numbers, URLs, and configured literal terms. `RedactionMatch` records field, kind, byte offsets, and replacement text without storing the raw matched value.

#### `(*meeting.Meeting) Redact(redactor meeting.Redactor) (meeting.Meeting, meeting.RedactionReport, error)`
Returns a redacted meeting copy, marks it as redacted, and reports the redaction matches across meeting metadata, participants, transcript turns, word text, and summary artifacts.

#### `meeting.Analyzer`
Defines a provider-neutral analysis boundary:

```go
type Analyzer interface {
    AnalyzeMeeting(ctx context.Context, meeting Meeting, request AnalysisRequest) (*AnalysisResult, error)
}
```

#### `(*meeting.Meeting) Analyze(ctx context.Context, analyzer meeting.Analyzer, request meeting.AnalysisRequest) (meeting.Meeting, meeting.AnalysisResult, error)`
Runs a caller-supplied analyzer, normalizes the output, validates transcript evidence IDs, and returns a copy of the meeting with `Summary` applied.

#### `meeting.HeuristicAnalyzer`
Provides a stdlib-only deterministic reference analyzer for local examples and baseline tests. It extracts an overview, decisions, action items, open questions, and risks from transcript text using simple rules and transcript turn IDs as evidence IDs. It is not an LLM-quality analyzer.

#### `(*meeting.Meeting) WithAnalysis(result meeting.AnalysisResult) (meeting.Meeting, meeting.AnalysisResult, error)`
Applies already-generated analyzer output to a meeting copy after normalization and validation.

#### `(meeting.MeetingSummary) Normalize() meeting.MeetingSummary`
Trims generated text, fills stable artifact IDs, defaults missing action item status to `proposed`, and removes empty/duplicate evidence IDs.

#### `(meeting.MeetingSummary) ValidateForMeeting(meeting.Meeting) error`
Validates generated topics, decisions, action items, follow-ups, open questions, risks, timing, statuses, duplicate IDs, and transcript evidence references.

### Meeting Evaluation

#### `evaluation.EvaluateMeeting(reference evaluation.MeetingReference, prediction meeting.Meeting) (evaluation.Report, error)`
Scores a predicted meeting against deterministic fixture data. The report includes transcript WER/CER, turn-level speaker attribution accuracy, exact normalized action-item precision/recall/F1, and an optional real-time factor.

#### `evaluation.EvaluateTranscriptText(reference []evaluation.ReferenceTurn, prediction []meeting.TranscriptTurn) evaluation.TextReport`
Computes word and character edit-distance metrics using stable normalization.

#### `evaluation.EvaluateSpeakerAttribution(reference []evaluation.ReferenceTurn, prediction []meeting.TranscriptTurn) evaluation.SpeakerAttributionReport`
Compares speaker IDs by transcript turn ID with index fallback for fixture data that omits IDs.

#### `evaluation.EvaluateDiarization(reference, hypothesis []evaluation.DiarizationSegment, options evaluation.DiarizationOptions) (evaluation.DiarizationReport, error)`
Computes diarization error rate with optimal speaker mapping, missed speech, false alarm, and confusion components. Optional boundary collars and overlap skipping follow common md-eval/pyannote-style evaluation controls.

#### `evaluation.EvaluateActionItems(reference []meeting.ActionItem, prediction []meeting.ActionItem) evaluation.ClassificationReport`
Scores extracted action items with exact normalized text matching.

#### `evaluation.RealTimeFactor(processingDuration time.Duration, audioDuration time.Duration) (float64, error)`
Returns processing duration divided by audio duration and rejects invalid durations.

### ASR Streaming

#### `asr.NewService(config *voicekit.ASRConfig) (*Service, error)`
Creates a new ASR service for real-time speech recognition.

#### `(*Service) ProcessAudioChunk(ctx context.Context, sessionID string, audio []float32) (*types.Transcription, error)`
Processes a chunk of audio for streaming ASR, returning partial or final transcription results.

VoiceKit keeps the native Sherpa online stream in session state across chunks. With VAD enabled, it retains bounded pre-roll until speech activation; after activation, each caller-provided sample is accepted exactly once, including chunks smaller than `ChunkSize` or larger than the rolling compatibility buffer. Native state is discarded before a processing error returns its admission slot. New session streams are admitted up to `MaxConcurrentStreams`; excess admission returns a wrapped `*asr.StreamCapacityError` that callers can inspect with `errors.As`.

#### `(*Service) FinishStream(ctx context.Context, sessionID string) (*types.Transcription, error)`
Finalizes the current utterance with `InputFinished` without replaying old audio, deletes native ASR/VAD state, and returns the active-stream slot. Session metadata is retained so the same ID can start a later utterance and preserve its selected language.

#### `types.Transcription` timing contract
Sherpa online and offline results expose model-native token text and timing through `Tokens []types.Token`. A token may be a character, subword, or word depending on the model. `Token.HasTiming` distinguishes a real zero start from missing timing, and offline token duration is retained when Sherpa supplies it. `Words []types.Word` is intentionally empty unless a recognizer or downstream stage performs real word segmentation.

#### `(*Service) RegisterModel(model asr.Model) error`
Registers a new ASR model with the service.

#### `(*Service) GetModel(name string) (asr.Model, bool)`
Retrieves a registered ASR model by name.

#### `(*Service) SelectModel(language string, requirements *types.ModelRequirements) (asr.Model, error)`
Selects the best ASR model based on language and performance requirements.

#### `(*Service) Close() error`
Closes the ASR service and releases all resources.

### ASR Offline

#### `asr.NewOfflineTranscriber(config *voicekit.ASRConfig) (*SegmentedTranscriber, error)`
Creates the long-audio-safe offline transcriber: a `SherpaOfflineModel` behind a `SegmentedTranscriber`, with a Silero/TEN `SpeechSegmenter` when `VADProvider` names one. Chunking comes from `Offline.MaxChunkSeconds`, `ChunkOverlapSeconds`, `SpeechPaddingSeconds` and `SpeechMergeGapSeconds` (zero means the `DefaultChunkOptions` value).

#### `asr.NewSegmentedTranscriber(inner types.Transcriber, segmenter SpeechSegmenter, opts ChunkOptions) (*SegmentedTranscriber, error)`
Wraps any offline transcriber. `Transcribe` finds speech with the segmenter (nil means the whole input), plans chunks with `PlanChunks`, decodes them in order, and stitches them: around an overlapping cut, tokens are kept per word by timestamp when the model's tokens spell its text, otherwise repeated words are dropped from the text. `Transcription.Segments` holds one entry per decode and `Transcription.Tokens` carries absolute token times. Silence returns an empty transcription. Context cancellation is checked between decodes.

#### `asr.PlanChunks(spans []SpeechSpan, samples []float32, sampleRate int, opts ChunkOptions) ([]Chunk, error)`
Pure planning step: clamps, pads and merges speech spans, then cuts any span longer than `opts.MaxChunk` (at most `asr.MaxDecodeDuration`) at the quietest 200 ms within `opts.CutSearch` of the cap, with `opts.Overlap` of shared audio. Every speech sample belongs to exactly one chunk's keep range.

#### `asr.NewSpeechSegmenter(config *VADConfig) (SpeechSegmenter, error)`
Returns a `SherpaSegmenter` for `silero_vad`/`ten_vad` (each call runs its own native detector, so calls may run concurrently; other sample rates are resampled for detection) or nil for `none`/`energy`.

#### `asr.NewSherpaOfflineModel(config *voicekit.ASRConfig) (*SherpaOfflineModel, error)`
Creates a Sherpa offline model for one of the `Offline.ModelFamily` values: `transducer` (set `ModelType: "nemo_transducer"` for Parakeet TDT), `paraformer`, `zipformer_ctc`, `nemo_ctc`, `wenet_ctc`, `omnilingual_ctc`, `med_asr_ctc`, `firered_asr_ctc`, `firered_asr`, `dolphin`, `tdnn`, `sense_voice`, `whisper`, `moonshine` (v1 or v2 file sets), `canary`, `cohere_transcribe`, `qwen3_asr` and `funasr_nano` (these two take a tokenizer directory instead of a tokens file). Validation requires every path the family needs, including `tokens` for every family except the two tokenizer-directory ones. `Transcribe` refuses input longer than `Offline.MaxDecodeSeconds` (30 s for Whisper by default) with `*asr.AudioTooLongError`, checks the context again after waiting for the decode lock, and returns native Sherpa token metadata in `Transcription.Tokens`; it is not presented as word segmentation.

### TTS Offline

#### `tts.NewSherpaOfflineSynthesizer(config *voicekit.TTSConfig) (*SherpaOfflineSynthesizer, error)`
Creates a Sherpa offline speech synthesizer for complete text inputs.

#### `(*SherpaOfflineSynthesizer) Synthesize(ctx context.Context, request types.SynthesisRequest) (*types.SynthesizedSpeech, error)`
Synthesizes text into normalized float32 PCM samples with non-empty text validation before native Sherpa calls.

## Configuration

### Main Config
```go
type Config struct {
    Audio       AudioConfig
    Speaker     SpeakerConfig
    Diarization DiarizationConfig
    ASR         ASRConfig
    TTS         TTSConfig
}
```

### Audio Config
```go
type AudioConfig struct {
    SampleRate      int     // Sample rate in Hz (e.g., 16000)
    Channels        int     // Number of channels (1=mono, 2=stereo)
    NormalizeFactor float32 // Normalization factor for audio levels
}
```

### Speaker Config
```go
type SpeakerConfig struct {
    ModelPath  string  // Path to Sherpa-ONNX speaker model
    NumThreads int     // Number of threads for inference
    Provider   string  // Compute provider ("cpu", "cuda", etc.)
    Threshold  float32 // Similarity threshold (0.0-1.0)
    DataDir    string  // Directory for speaker database
    Logger     Logger  // Optional logger
}
```

### Diarization Config
```go
type DiarizationConfig struct {
    Enabled               bool
    Backend               string  // "sherpa_offline" or explicit "basic"
    SegmentationModelPath string  // Sherpa diarization segmentation model
    EmbeddingModelPath    string  // Sherpa speaker embedding model
    Provider              string  // "cpu", "cuda", or "coreml"
    NumThreads            int
    ClusteringThreshold   float32
    NumClusters           int     // Optional exact speaker count
    MinDurationOn         float32
    MinDurationOff        float32
    Logger                Logger
}
```

### ASR Config
```go
type ASRConfig struct {
    Enabled              bool    // Whether ASR is enabled
    Backend              string  // "sherpa_offline" or "sherpa_online"
    DefaultModel         string  // Default model/transcriber name
    Language             string  // Default language ("en")
    Quantization         string  // Informational quantization label
    MaxConcurrentStreams int     // Hard limit for active streaming session admissions
    StreamTimeout        int     // Stream timeout in seconds
    ChunkSize            int     // Audio chunk size in samples (16000 = 1 second at 16kHz)
    SampleRate           int     // ASR sample rate
    FeatureDim           int     // Sherpa feature dimension
    Provider             string  // Sherpa runtime provider ("cpu", "cuda", "coreml")
    NumThreads           int     // Sherpa thread count
    Online               OnlineConfig
    Offline              OfflineConfig
    VADProvider          string  // "none", "energy", "silero_vad", or "ten_vad"
    VADModelPath         string  // Required for Sherpa VAD providers
    Logger               Logger  // Optional logger
}

type OnlineConfig struct {
    TokensPath  string
    EncoderPath string
    DecoderPath string
    JoinerPath  string
    ModelPath   string
    ModelType   string
}

type OfflineConfig struct {
    ModelFamily string // "transducer", "paraformer", "zipformer_ctc", "nemo_ctc", "sense_voice", or "whisper"
    TokensPath  string
    EncoderPath string
    DecoderPath string
    JoinerPath  string
    ModelPath   string
    Language    string
    Task        string // Whisper task, usually "transcribe"
}
```

### TTS Config
```go
type TTSConfig struct {
    Enabled         bool
    Backend         string // "sherpa_offline"
    ModelFamily     string // "vits", "matcha", "kokoro", "kitten", "zipvoice", "pocket", or "supertonic"
    Provider        string // Sherpa runtime provider ("cpu", "cuda", "coreml")
    NumThreads      int
    MaxNumSentences int
    SilenceScale    float32
    SpeakerID       int
    Speed           float32
    Vits            TTSVitsConfig
    Matcha          TTSMatchaConfig
    Kokoro          TTSKokoroConfig
    Kitten          TTSKittenConfig
    Zipvoice        TTSZipvoiceConfig
    Pocket          TTSPocketConfig
    Supertonic      TTSSupertonicConfig
    Logger          Logger
}
```

## Error Handling

VoiceKit provides specific error types for different components:

- `AudioError`: Audio processing errors
- `SpeakerError`: Speaker recognition errors
- `DiarizationError`: Diarization errors
- `ASRError`: ASR streaming errors
- TTS returns wrapped configuration or synthesis errors from the `tts` package

All errors implement `Unwrap()` for compatibility with `errors.Is()` and `errors.As()`.

## Logger Interface

Implement this interface for custom logging:

```go
type Logger interface {
    Infof(format string, args ...interface{})
    Warnf(format string, args ...interface{})
    Errorf(format string, args ...interface{})
}
```

## Examples

See the `examples/` directory for complete working examples:
- `examples/basic_usage/` - Basic VoiceKit usage
- `examples/asr_streaming/` - Real-time ASR streaming
- `examples/asr_offline/` - Batch/offline ASR transcription
- `examples/tts_offline/` - Batch/offline speech synthesis
- `examples/speaker_only/` - Speaker recognition only
- `examples/audio_processing/` - Audio format conversion
- `examples/diarization/` - Speaker diarization

## Model Requirements

### ASR Models

VoiceKit's ASR runtime uses explicit Sherpa backend configs:

- `ASR.Backend = "sherpa_offline"` uses `ASR.Offline`.
  Supported first-pass families are `transducer`, `paraformer`, `zipformer_ctc`, `nemo_ctc`, `sense_voice`, and `whisper`.
- `ASR.Backend = "sherpa_online"` uses `ASR.Online`.
  Online transducer requires `Online.TokensPath`, `Online.EncoderPath`, `Online.DecoderPath`, and `Online.JoinerPath`.
  Online CTC requires `Online.TokensPath`, `Online.ModelPath`, and `Online.ModelType`.

ASR is disabled by default. If `ASR.Enabled` is true and required model files are absent, configuration validation fails.

### TTS Models

VoiceKit's TTS runtime uses `TTS.Backend = "sherpa_offline"` and `TTS.ModelFamily` to choose a Sherpa offline TTS family. Supported families are `vits`, `matcha`, `kokoro`, `kitten`, `zipvoice`, `pocket`, and `supertonic`.

TTS is disabled by default. If `TTS.Enabled` is true and required model files or data directories are absent, configuration validation fails.

### Diarization Models

Production diarization uses `Diarization.Backend = "sherpa_offline"` and requires:

- `SegmentationModelPath`: Sherpa offline diarization segmentation model.
- `EmbeddingModelPath`: Sherpa speaker embedding model.
- Optional clustering settings: `ClusteringThreshold`, `NumClusters`, `MinDurationOn`, and `MinDurationOff`.

The `basic` backend is explicit and intended for tests or local experiments that inject an `EmbeddingExtractor`.

### VAD Models

VoiceKit supports:

- `none`: no VAD service.
- `energy`: simple energy detector for tests/basic local filtering.
- `silero_vad`: Sherpa Silero VAD with `VADModelPath`.
- `ten_vad`: Sherpa TEN VAD with `VADModelPath`.

Sherpa VAD providers require model files and CGO-enabled Sherpa bindings.

### Speaker Recognition Models
Download speaker embedding models from [Sherpa-ONNX Model Zoo](https://github.com/k2-fsa/sherpa-onnx):

```bash
# Example: 3D Speaker model
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
```

Speaker embedding extraction creates a new Sherpa stream per extraction. Do not reuse streams after `InputFinished`; VoiceKit deletes them immediately after computing the embedding.

### Audio Format Support
- **Input**: WAV, PCM, FLAC, MP3, and Ogg-Vorbis
- **Output**: WAV and PCM
- **Unsupported**: AAC/M4A input and compressed output return explicit errors

## Testing Strategy

### Unit Testing Architecture
- **Isolation**: Each component tested independently with mocked dependencies
- **Interface Mocking**: Dependency injection enables comprehensive mocking
- **Table-Driven Tests**: Go's testing framework used extensively for data-driven tests
- **Native Lifecycle**: ASR and speaker tests use fake native adapters to verify stream reuse, finalization, reset, and close behavior without model files

### Integration Testing
- **Component Interaction**: Tests verify correct interaction between audio, speaker, and diarization components
- **End-to-End Pipelines**: Full voice processing pipelines tested with synthetic data
- **Performance Benchmarks**: `testing.B` benchmarks for performance regression detection

### Local Verification and CI
Run the same gate locally that CI runs:

```bash
make verify
```

`make verify` checks forbidden tracked artifacts, verifies modules, checks
`gofmt`, verifies `go mod tidy`, runs `golangci-lint`, `go vet ./...`,
`go test ./...`, and `go test -race ./...` when the local platform supports
the race detector. The Gitea Actions workflow in `.gitea/workflows/ci.yml` uses
`actions/setup-go` with `go-version-file: go.mod` and delegates to
`make verify`, so local and remote gates stay aligned.

The forbidden-file check rejects committed local environment files, model
artifacts, runtime profiles, and generated benchmark directories. Model files
remain caller/deployment owned and are intentionally not committed.

### Benchmark Workflow
```bash
make bench-all
make bench-compare BEFORE=/tmp/voicekit-benchmarks/before.txt AFTER=/tmp/voicekit-benchmarks/after.txt
```

Use `benchstat` for before/after comparisons. Keep model-backed Sherpa RTF
benchmarks env-gated unless local model paths and audio fixtures are supplied.
Raw benchmark and profile artifacts go under `/tmp/voicekit-benchmarks` and
`/tmp/voicekit-profiles` by default, not tracked source directories.

### Test Data Management
- **Synthetic Audio**: Generated test audio with known characteristics
- **Speaker Embeddings**: Pre-computed embeddings for consistent testing
- **Edge Cases**: Tests cover boundary conditions (empty audio, extreme sample rates, etc.)

### Coverage Goals
- **Core Logic**: >90% coverage for business logic functions
- **Error Paths**: All error conditions tested with appropriate assertions
- **Concurrency**: Race condition testing using `go test -race`

## Security Considerations

### Input Validation
- **Audio Format Sanitization**: Strict validation of audio file headers
- **Buffer Bounds Checking**: All array operations bounds-checked
- **Size Limits**: Configurable limits on audio file sizes and durations

### Memory Safety
- **Project-Owned Safety**: Project-owned code avoids unsafe pointer operations; Sherpa integration is isolated behind CGO-backed adapters
- **Slice Bounds**: All slice operations use safe indexing
- **Resource Cleanup**: Proper cleanup of external library resources

### Data Privacy
- **Speaker Embeddings**: Sensitive biometric data stored securely
- **Temporary Files**: Audio processing uses memory buffers, not temporary files
- **Logging Sanitization**: Speaker IDs and sensitive data not logged

### Denial of Service Prevention
- **Resource Limits**: Configurable timeouts and memory limits
- **CPU Bounds**: Thread pool limits prevent resource exhaustion
- **Rate Limiting**: Integration-ready for rate limiting at higher layers

## Deployment and Operations

### Configuration Management
- **Environment Variables**: Support for 12-factor app configuration
- **Validation**: Configuration validated at startup with detailed error messages
- **Hot Reload**: Select components support configuration updates without restart

### Monitoring and Observability
- **Metrics Integration**: Compatible with Prometheus, statsd, etc.
- **Structured Logging**: Consistent log formats with contextual information
- **Health Checks**: Component-level health reporting for orchestration systems

### Performance Tuning
- **Buffer Sizes**: Tunable buffer sizes for memory/throughput trade-offs
- **Thread Pools**: Configurable thread counts for different workloads
- **Caching Strategies**: LRU caches for frequently accessed speaker data

### Production Checklist
- [ ] Configure appropriate log levels for production
- [ ] Set resource limits (memory, CPU, file descriptors)
- [ ] Enable metrics collection and monitoring
- [ ] Configure proper timeouts for all operations
- [ ] Set up proper speaker database backup/restore procedures
- [ ] Test with production-like audio data and speaker counts
- [ ] Verify CGO dependencies are properly installed
- [ ] Configure appropriate security policies for biometric data

## Code Organization and Development Practices

### Package Structure Rationale

```
voicekit/
├── voicekit.go      # Main API and orchestration
├── config.go        # Configuration types and validation
├── logger.go        # Logging interface and default implementation
├── errors.go        # Error types and handling utilities
├── asr/             # Sherpa ASR online/offline system
│   ├── service.go   # Streaming ASR orchestration and model management
│   ├── streaming.go # Session management and streaming state
│   ├── buffer.go    # Cache-aware audio buffering
│   ├── vad.go       # Explicit VAD provider seam
│   ├── sherpa_online.go # Sherpa online recognizer implementation
│   ├── sherpa_offline.go # Sherpa offline transcriber implementation
│   ├── fake_model.go # Explicit fake model for tests/demos
│   └── *_test.go    # Comprehensive test coverage
├── tts/             # Sherpa offline text-to-speech system
│   ├── config.go    # TTS config, defaults, and model-path validation
│   ├── sherpa_offline.go # Sherpa offline synthesizer implementation
│   └── *_test.go    # Unit tests with fake native synthesizer
├── meeting/         # Product-level meeting intelligence artifacts
│   ├── meeting.go   # Meeting, participant, transcript, summary, and task contracts
│   ├── from_diarization.go # Mapping from diarized ASR output into meetings
│   ├── export.go    # Markdown, JSON, SRT, and WebVTT meeting exports
│   ├── analyzer.go  # Provider-neutral analyzer contract, normalization, and validation
│   ├── heuristic_analyzer.go # Stdlib-only deterministic reference analyzer
│   ├── redaction.go # Redaction policy, redactor boundary, and meeting redaction
│   └── *_test.go    # Artifact validation and mapping tests
├── evaluation/      # Deterministic meeting quality metrics
│   ├── evaluation.go # WER/CER, speaker attribution, action item, and RTF metrics
│   └── *_test.go    # Fixture metric tests
├── examples/
│   └── meeting_intelligence/ # Analyze, redact, export, and evaluate flow
├── MEETING_INTELLIGENCE.md # Meeting intelligence developer guide
├── audio/           # Audio processing primitives
│   ├── resampler.go # Sample rate conversion algorithms
│   ├── converter.go # Format conversion and I/O
│   ├── formats.go   # Audio format definitions and utilities
│   └── types.go     # Audio-specific type definitions
├── speaker/         # Speaker recognition system
│   ├── manager.go   # Core speaker operations (register/identify/verify)
│   ├── embedding_adapter.go # Neural network adapter integration
│   ├── database.go  # Persistence layer and caching
│   ├── parser.go    # Audio parsing utilities
│   └── types.go     # Speaker domain types and interfaces
└── diarization/     # Speaker diarization system
    ├── manager.go   # Diarization backend orchestration
    ├── sherpa_offline.go # Sherpa offline diarization backend
    ├── basic_backend.go # Explicit basic clustering backend
    ├── segmenter.go # Audio segmentation algorithms
    ├── integrator.go# ASR integration and result alignment
    └── *_test.go    # Unit and env-gated integration tests
```

### Development Principles

#### Single Responsibility Principle
- **Audio Package**: Only audio format conversion and processing
- **Speaker Package**: Only speaker recognition operations
- **Diarization Package**: Only speaker diarization and segmentation
- **Meeting Package**: Only product-level meeting artifacts, deterministic mapping from lower-level voice results, exports, redaction, and analyzer contracts
- **Evaluation Package**: Only deterministic fixture metrics for product-quality measurement
- **Main Package**: Only orchestration and API provision

#### Dependency Injection
- **Interface-Based**: All external dependencies injected as interfaces
- **Testability**: Easy mocking for comprehensive unit testing
- **Flexibility**: Runtime configuration of behavior

#### Error Handling Strategy
- **Structured Errors**: Custom error types with context
- **Error Wrapping**: Preserves error chains for debugging
- **Graceful Degradation**: Components fail independently

#### Memory Management
- **Object Pooling**: Reduces garbage collection pressure
- **Streaming Processing**: Minimizes memory footprint for large files
- **Resource Cleanup**: Proper cleanup of external resources

### Code Quality Standards

#### Naming Conventions
- **Exported Functions**: PascalCase, descriptive names (e.g., `NewVoiceKit`, `ProcessAudio`)
- **Internal Functions**: camelCase, implementation details (e.g., `extractEmbedding`)
- **Variables**: camelCase, descriptive and concise
- **Types**: PascalCase for exported, meaningful names

#### Documentation Standards
- **Package Comments**: Every package has a package comment explaining its purpose
- **Function Comments**: All exported functions documented with purpose and parameters
- **Code Comments**: Complex algorithms explained with comments
- **Example Code**: README contains working code examples

#### Testing Standards
- **Unit Tests**: Every exported function has corresponding unit tests
- **Integration Tests**: Component interaction tested
- **Benchmark Tests**: Performance-critical code benchmarked
- **Edge Cases**: Error conditions and boundary cases tested
- **CI Parity**: Keep `make verify` as the single local and CI entrypoint for
  formatting, lint, vet, tests, race checks, and repository hygiene

### Performance Considerations

#### Algorithm Complexity
- **Audio Resampling**: O(n) linear time complexity
- **Speaker Search**: O(m*k) where m=speakers, k=embedding dimension
- **Diarization Clustering**: O(s²) where s=segments (acceptable for typical use cases)

#### Memory Efficiency
- **Buffer Reuse**: Object pools prevent unnecessary allocations
- **Streaming I/O**: Large files processed in chunks
- **Copy Minimization**: Zero-copy operations where safe

#### Concurrency Design
- **Goroutine Safety**: All public APIs goroutine-safe
- **Lock Granularity**: Fine-grained locking to minimize contention
- **Channel Communication**: Internal coordination uses channels

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Review Guidelines

- **Architecture**: Ensure changes align with layered architecture
- **Dependencies**: No new external dependencies without justification
- **Performance**: Consider algorithmic complexity and memory usage
- **Testing**: All changes include appropriate test coverage
- **Documentation**: Update README and code comments as needed
- **Security**: Consider security implications of changes

## Algorithms and Mathematical Foundations

### Audio Processing Algorithms

#### Resampling Mathematics
The resampling implementation uses interpolation theory:

**Linear Interpolation:**
```
y(x) = y₁ + (y₂ - y₁) * (x - x₁)/(x₂ - x₁)
```
- **Pros**: Fast, simple, low latency
- **Cons**: Aliasing artifacts, poor high-frequency response
- **Use Case**: Real-time applications prioritizing speed

**Cubic Interpolation:**
```
y(x) = a*x³ + b*x² + c*x + d
```
Where coefficients are solved from 4 neighboring points using:
```
[a,b,c,d] = [-0.5, 1.5, -1.5, 0.5] * [y₀, y₁, y₂, y₃]ᵀ
```
- **Pros**: Better frequency response than linear
- **Cons**: Higher computational cost
- **Use Case**: Balanced quality/performance applications

**Lanczos Interpolation:**
```
L(x) = sinc(x) * sinc(x/a) for |x| < a, 0 otherwise
```
- **Pros**: Excellent anti-aliasing properties
- **Cons**: Highest computational cost
- **Use Case**: High-quality offline processing

#### Audio Normalization
**RMS-Based Normalization:**
```
RMS = √(Σ(xᵢ²)/n)
scale_factor = target_rms / current_rms
normalized = input * scale_factor
```
- **Purpose**: Consistent audio levels for downstream processing
- **Range**: Typically normalizes to -20dB to -30dB RMS
- **Clamping**: Prevents over-amplification (>4x gain limit)

### Speaker Recognition Mathematics

#### Cosine Similarity
Speaker comparison uses cosine similarity on embedding vectors:
```
cosine_similarity(a,b) = (a·b) / (||a||₂ * ||b||₂)
```
- **Range**: [-1, 1] where 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
- **Properties**: Independent of vector magnitude, rotationally invariant
- **Advantages**: Computationally efficient, works well for high-dimensional embeddings

#### Embedding Space
- **Dimensionality**: Typically 192-512 dimensions from neural networks
- **Normalization**: Embeddings usually L2-normalized before similarity computation
- **Clustering**: Speaker embeddings form clusters in vector space

### Diarization Algorithms

#### Agglomerative Clustering
Speaker clustering uses bottom-up hierarchical clustering:

1. **Initialization**: Each segment starts as its own cluster
2. **Similarity Matrix**: Compute pairwise similarities between all clusters
3. **Merge Step**: Find most similar pair, merge them
4. **Iteration**: Repeat until similarity threshold reached
5. **Labeling**: Assign speaker IDs to final clusters

**Complexity**: O(n³) worst case, but practical for typical speaker counts

#### Audio Segmentation

**Silence-Based Segmentation:**
- **Energy Threshold**: Detect silence using RMS energy below threshold
- **Hysteresis**: Require consecutive silence frames to avoid noise triggers
- **Gap Filling**: Merge segments separated by short silence periods

**VAD-Based Segmentation:**
- **Voice Activity Detection**: Use external VAD to identify speech regions
- **Boundary Refinement**: Adjust boundaries based on energy analysis
- **Overlap Handling**: Merge overlapping speech segments

### Numerical Stability Considerations

#### Floating Point Precision
- **Accumulation**: Use Kahan summation for precise RMS calculations
- **Normalization**: Careful handling of near-zero values in similarity computations
- **Clamping**: Prevent numerical instabilities in interpolation

#### Memory Layout
- **Cache Alignment**: Struct fields ordered for optimal memory access
- **SIMD Opportunities**: Algorithms structured for potential vectorization
- **Memory Coherence**: Sequential access patterns minimize cache misses

### Performance Optimizations

#### Algorithmic Optimizations
- **Early Termination**: Similarity searches stop when threshold exceeded
- **Approximate Methods**: Use approximate nearest neighbor for large speaker databases
- **Incremental Updates**: Update clustering incrementally rather than recomputing

#### Memory Optimizations
- **Buffer Pools**: Reuse allocated buffers to reduce GC pressure
- **Streaming Processing**: Process audio in chunks to bound memory usage
- **Lazy Evaluation**: Compute results only when requested

#### Parallelization Opportunities
- **Independent Segments**: Diarization segments can be processed in parallel
- **Batch Processing**: Multiple audio files processed concurrently
- **GPU Acceleration**: Neural network inference can use GPU acceleration

## License

This library is released under the MIT License. See LICENSE file for details.

## Migration from Internal Packages

When migrating from the original ASR server's internal packages:

1. Replace `internal/speaker` imports with `go.glpx.pro/gdk/voice/speaker`
2. Replace `internal/audio` imports with `go.glpx.pro/gdk/voice/audio`
3. Replace `internal/diarization` imports with `go.glpx.pro/gdk/voice/diarization`
4. Update configuration structures to use `voicekit.Config`
5. Remove dependencies on `internal/logger` and `internal/config`

Example migration:

```go
// Before
import "asr_server/internal/speaker"

// After
import "go.glpx.pro/gdk/voice/speaker"
```
