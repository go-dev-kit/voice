# VoiceKit Product Roadmap 2026

## Scope

This roadmap records the July 4, 2026 product and market review for VoiceKit.
It complements the stack-focused `VOICE_STACK_RESEARCH_2026.md` and the
performance-focused `PERFORMANCE_IMPROVEMENTS.md`.

VoiceKit is currently a local-first Go voice-processing library. Its strongest
foundation is local ASR, VAD, speaker recognition, diarization, TTS, audio
plumbing, and a deterministic meeting-intelligence SDK layer through explicit
Sherpa-backed runtime configuration. It is not yet a complete meeting-notes
product or hosted speech platform.

## Product Positioning

Recommended positioning:

> Local-first voice engine SDK for private meeting intelligence and voice agents.

This avoids competing head-on with full SaaS meeting assistants before VoiceKit
has capture, collaboration, search, privacy, and workflow integrations. It also
uses the current advantage: local inference, explicit model ownership, and a Go
library boundary that can be embedded into agents or private services.

## Market Snapshot

Meeting-note products such as Otter, Fireflies, Fathom, Granola, tl;dv,
Microsoft Teams recap, and Google Meet with Gemini compete on workflow outcomes:
capture meetings, create summaries, extract action items, search prior meetings,
share notes, and sync records into CRMs or work tools.

Developer voice platforms such as Deepgram, AssemblyAI, ElevenLabs, OpenAI audio
APIs, and Sherpa-ONNX compete on realtime APIs, transcription quality,
diarization, redaction, custom vocabulary, language coverage, TTS quality,
latency, and evaluation tooling.

VoiceKit should bridge those two markets from the local-first side:

- SDK consumers need clean contracts, model/runtime control, and predictable
  evaluation.
- Product consumers need meeting artifacts, privacy controls, search, and
  workflow output.

## Current VoiceKit Baseline

Current strengths:

- Local-first Sherpa-backed ASR with separate online and offline surfaces.
- Sherpa offline diarization behind an explicit backend boundary.
- Speaker recognition and speaker embedding persistence.
- Sherpa offline TTS through a separate synthesizer facet.
- WAV/PCM audio conversion and resampling foundation.
- Meeting artifacts with Markdown, JSON, SRT, and WebVTT exports.
- Provider-neutral analyzer contract plus a stdlib-only deterministic reference
  analyzer for examples and baseline tests.
- Built-in deterministic redaction for emails, phone-like numbers, URLs, and
  configured literal terms, with export-time enforcement when requested.
- Deterministic meeting evaluation metrics for transcript WER/CER, turn-level
  speaker attribution, action-item precision/recall, and real-time factor.
- Measured Go 1.26.4 local CPU performance baseline with optimized WAV/PCM
  decode, basic diarization segmentation, and benchmark coverage for
  meeting/evaluation surfaces.
- Repository hygiene and CI guardrails: Gitea Actions workflow,
  `make verify`, golangci-lint v2 config, forbidden-file checks, and ignored
  local benchmark/model/runtime artifacts.
- Focused tests, race validation, and env-gated model integration tests.

Current product gaps:

- No LLM-quality meeting analyzer implementation for local or hosted providers;
  the built-in `meeting.HeuristicAnalyzer` is only a deterministic reference
  baseline.
- No meeting capture adapters for calendar, desktop audio, browser extension,
  Zoom, Google Meet, Teams, or mobile capture.
- No AAC/M4A or video ingestion (WAV, PCM, FLAC, MP3 and Ogg/Vorbis decode,
  and VAD-segmented long audio through `asr.NewOfflineTranscriber`, exist).
- No Google Docs, email recap, Slack, Jira, Asana, Notion, Salesforce, or
  HubSpot export adapters beyond the current Markdown, JSON, SRT, WebVTT, and
  DOCX exports.
- No transcript-cleanup policy surface. Recognizer-level hotwords exist for
  transducer models; keyword spotting is a separate streaming capability.
- No consent capture flow or retention enforcement (deletion jobs). Consent
  and retention metadata, `DeletionDue`, export-time `RequireConsent`, and
  deterministic PII-style redaction exist but are not a compliance guarantee.
- No searchable meeting memory or cross-meeting retrieval.
- No voice-agent event protocol with barge-in, end-of-turn, streaming TTS, or
  tool calls.
- No DER-focused benchmark corpus, latency dashboard, or model/provider
  comparison harness beyond deterministic fixture metrics, corpus loaders, and
  env-gated real-model benchmarks.

## Engine Track — Streaming Completeness & Capability Exposure (2026-07)

This historical track complemented the product initiatives below. Its completed
scope closed the core Sherpa streaming lifecycle, exposed the high-value binding
capabilities selected for VoiceKit, and paid down validation/DX debt surfaced by
the first real downstream integration (`asr_server`). The September correctness
sprint completed admission control and truthful token timing for the v0.4.0
release scope. Full sprint evidence remains in `todo.md`.

### Integration-driven findings and status

- **Streaming lifecycle is implemented.** `types.ASRService` exposes
  `FinishStream`; bounded VAD pre-roll prevents delayed activation from dropping
  speech, caller audio is accepted exactly once, duration is cumulative per
  utterance, VAD is session-owned, and native state is closed before final or
  failed operations return bounded `MaxConcurrentStreams` admission slots.
- **Timing semantics are explicit.** Sherpa model tokens populate
  `Transcription.Tokens`; `Transcription.Words` is reserved for real word
  segmentation instead of relabeling subword tokens.
- **Provider validation is implemented.** VAD aliases such as
  `silero`→`silero_vad` normalize explicitly, unknown providers fail loudly, and
  multilingual Kokoro validates its language/lexicon requirements.
- **Nemotron online decoding is greedy-only** (no hotwords). Hotword/glossary
  support must be explicit about model-family limits.

### Capability exposure status

VoiceKit now wraps `KeywordSpotter`, offline punctuation, online/offline speech
denoising, `SpokenLanguageIdentification`, and streaming TTS through Sherpa's
generated-audio callback. `SourceSeparation` and `AudioTagging` remain optional
future wrappers. QNN/RKNN/Ascend NPU providers are still not exposed by the Go
binding (which exposes cpu/cuda/coreml), so they require upstream binding work
and are not a simple VoiceKit wrapper.

### Prioritized objectives

**P0 — streaming lifecycle + validation (completed)**
1. Public `FinishStream`, exact-once sample ownership, per-session VAD, bounded
   admission, and model-native token timing.
2. `make fetch-test-models` plus env-gated native smoke across
   ASR/VAD/TTS/diarization/speaker, driven by `testdata/model_matrix.yaml`.
3. Provider-name validation/aliasing plus multilingual-Kokoro validation.

**P1 — capability exposure (completed)**
4. Streaming TTS (`StreamingSynthesizer` over the GeneratedAudio callback;
   chunked, interruptible).
5. Punctuation restoration (post-ASR normalizer).
6. Keyword spotting (streaming wake-word/hotword).
7. Spoken language identification (auto language routing).

**P2 — engine hardening (completed or evidence-gated)**
8. Speech denoiser preprocessing stage (optional, before VAD/ASR).
9. Multi-online-model hosting (relax the single-backend-per-instance constraint
   so one service can host more than one online model).
10. Recognizer init/first-chunk benchmark shipped. October 2026 real-model
    measurements (offline init about 1.9 s Parakeet TDT int8, 1 to 1.5 s
    Whisper small int8; speaker stream creation under 0.1 percent of an
    extraction) do not justify a recognizer or stream pool; a warm-up decode
    at construction is pending a quiet-host rerun of the first-decode penalty.
11. `evaluation` DER metric; speaker DB retention/eviction (existing stretch).

Deferred/strategic (see Later): source separation, audio tagging, overlap-aware
diarization, NPU providers (needs binding work), SIMD resampling, embedding
quantization, capture adapters, LLM analyzer adapters, searchable meeting memory.

## Top 10 Initiatives

### 1. Meeting Intelligence Domain

Hypothesis: We believe adding first-class meeting artifacts for SDK consumers
will make VoiceKit usable as a meeting-notes foundation because current outputs
stop at ASR and diarization primitives.

Deliverables:

- `meeting.Meeting`, `Participant`, `TranscriptTurn`, `MeetingSummary`,
  `Decision`, `ActionItem`, `FollowUp`, and `Topic`.
- Mapping from `diarization.IntegratedResult` into transcript turns.
- Validation helpers for IDs, timing, participant mappings, and empty content.

Success metrics:

- SDK callers can represent a diarized transcript without inventing their own
  schema.
- Meeting package has focused tests and no dependency on provider-specific LLMs.

Effort: Small.

### 2. Meeting Artifact Exports

Hypothesis: We believe structured exports will make VoiceKit useful in real
workflows because meeting-note users consume Markdown, captions, task payloads,
and recap emails more often than raw Go structs.

Deliverables:

- Markdown and JSON exporters.
- SRT and WebVTT transcript exporters.
- Stable export interfaces that can later target Google Docs, Slack, Jira,
  Notion, Asana, Salesforce, and HubSpot.

Success metrics:

- One meeting object can produce transcript, recap, and caption artifacts.
- Export tests cover timing, escaping, empty fields, and speaker labels.

Effort: Small to Medium.

### 3. Summaries, Decisions, and Action Items

Hypothesis: We believe provider-neutral summarization contracts will let callers
use local or hosted LLMs while keeping VoiceKit independent of one AI provider.

Deliverables:

- `meeting.Analyzer` interface for generating structured summaries.
- JSON-schema-like typed output contracts for summaries, decisions, action
  items, risks, open questions, and follow-ups.
- Deterministic cleanup and validation of analyzer output.

Success metrics:

- Analyzer implementations can be swapped without changing meeting schemas.
- Invalid or partial analyzer output fails with actionable validation errors.

Effort: Medium.

### 4. Product-Quality Evaluation Harness

Hypothesis: We believe model and workflow choices should be driven by measured
quality because meeting-note quality depends on transcript accuracy, speaker
attribution, and downstream extraction precision.

Deliverables:

- Fixtures for realistic meetings, noisy calls, multilingual calls, and
  overlapping speakers.
- Metrics for WER/CER, diarization error rate, speaker attribution accuracy,
  action-item precision/recall, latency, real-time factor, and memory.
- Benchmark reports that separate model quality from pipeline overhead.

Success metrics:

- Every major model/provider change has before/after evaluation data.
- Roadmap decisions cite measurements instead of intuition.

Effort: Medium.

### 5. Capture and Ingestion Adapters

Hypothesis: We believe VoiceKit must handle real meeting inputs because the
market expects capture from desktop audio, meeting platforms, uploads, and
mobile/phone calls.

Deliverables:

- File ingestion for long audio and common compressed formats.
- Optional ffmpeg-backed extraction adapter kept behind an explicit boundary.
- Interfaces for desktop capture, microphone capture, and platform importers.

Success metrics:

- A caller can ingest a typical meeting recording without pre-converting it to
  mono WAV/PCM outside VoiceKit.
- Unsupported formats fail with clear errors and remediation.

Effort: Medium.

### 6. Custom Vocabulary and Transcript Cleanup

Hypothesis: We believe domain terms and names are a core accuracy lever because
general ASR often misses product names, people names, acronyms, and customer
vocabulary.

Deliverables:

- Glossary and hotword config for supported Sherpa transducer models.
- Post-transcription replacement policy with audit metadata.
- Punctuation, inverse text normalization, casing, and disfluency cleanup hooks.

Success metrics:

- Domain fixture keyword recall improves without hurting general transcript
  readability.
- Hotword support is explicit about model-family limitations.

Effort: Medium.

### 7. Privacy, Consent, Retention, and PII Controls

Hypothesis: We believe local-first privacy can be VoiceKit's strongest
differentiator if it is modeled explicitly instead of left to callers.

Deliverables:

- Consent state attached to meeting capture and artifacts.
- Retention policy metadata for audio, transcript, embeddings, and summaries.
- PII redaction interface for transcript text and exported artifacts.
- Biometric data handling guidance for speaker embeddings.

Success metrics:

- Meeting records expose what may be stored, exported, or deleted.
- Tests verify redaction before export paths.

Effort: Medium to Large.

### 8. Searchable Meeting Memory

Hypothesis: We believe cross-meeting search will move VoiceKit from transcript
generation to reusable organizational memory.

Deliverables:

- Transcript and meeting-artifact store interfaces.
- Chunking model for turns, topics, decisions, and action items.
- Embedding/retrieval adapter boundaries.
- Query result types with citations back to speaker/time ranges.

Success metrics:

- Callers can answer questions across prior meetings with source-linked
  transcript evidence.
- Storage remains pluggable and local-first by default.

Effort: Large.

### 9. Realtime Voice-Agent Pipeline

Hypothesis: We believe a realtime event protocol will let VoiceKit support
assistants, live captions, and voice agents without coupling ASR, LLM, and TTS
into one package.

Deliverables:

- Event protocol for partial transcript, final transcript, endpoint, action,
  TTS chunk, interruption, and error events.
- End-of-turn and barge-in semantics.
- Streaming TTS abstraction separate from offline TTS.
- WebSocket/WebRTC/SIP adapter seams.

Success metrics:

- A sample agent can listen, decide, call a tool, and speak with bounded latency.
- ASR-only and TTS-only users still avoid agent complexity.

Effort: Large.

### 10. Packaging, Model Management, and Developer Experience

Hypothesis: We believe adoption depends on model lifecycle clarity because
VoiceKit intentionally does not commit model files.

Deliverables:

- Model manifest format with paths, checksums, licenses, sample rates, and
  supported capabilities.
- Download/verify helper CLI or package.
- Docker/reference service examples.
- Health checks and richer operational metrics for ASR, diarization, TTS, and
  meeting analysis.

Success metrics:

- New developers can run a real local demo from documented commands.
- Production users can audit model versions and licenses.

Effort: Medium to Large.

## Now / Next / Later

### Now

- Adopt the v0.4.0 `FinishStream`, bounded-admission, and `Tokens` contracts in
  downstream services; add word segmentation only where a consumer actually
  requires word-level alignment.
- Rerun the env-gated real-model benchmarks on a quiet host (first-decode
  penalty, Whisper auto vs forced language, transcriber RTF with VAD, and the
  go-hnswlib v1.5.0 Add path) before adding a warm-up decode.
- Expand evaluation fixtures toward realistic meetings, noisy calls,
  multilingual calls, and overlapping speakers.
- Expand the implemented DER metric with representative diarization benchmark
  corpora and add latency/memory reports.
- Keep performance decisions tied to Go 1.26.4 benchstat output in
  `/tmp/voicekit-benchmarks` and avoid tracked one-off benchmark artifacts.
- Keep `make verify` as the local and CI source of truth for formatting, lint,
  vet, tests, race checks, and forbidden-file hygiene.
- Design the next analyzer adapter boundary for a local LLM or hosted provider
  while keeping provider SDKs out of core meeting schemas.
- Ratchet pebble stores to format major version 13 in a v1.1.5 release, then
  move to pebble v2 (v2 refuses the format 1 stores VoiceKit writes today).

### Next

- Measure hotword biasing on a domain corpus (term recall vs WER cost of
  modified_beam_search).
- Add file ingestion adapters for common meeting recordings.
- Add DOCX, Google Docs, Slack, Jira, Notion, Asana, Salesforce, and HubSpot
  export adapters behind explicit boundaries.

### Later

- Add searchable meeting memory.
- Add workflow integrations and connector adapters.
- Add realtime voice-agent event protocol.
- Move verified model artifacts to the shared gdk/ml kit (draft in
  `internal/modelstore`), then bridge `testdata/model_matrix.yaml` to it.

## Initial Implementation Slice

The first implementation slice completed initiative 1:

- Add a `meeting` package.
- Define stable meeting intelligence types.
- Map `diarization.IntegratedResult` into transcript turns.
- Validate required IDs and timing.
- Cover the mapping and validation behavior with focused tests.

Follow-on slices completed meeting exports, analyzer contracts, deterministic
evaluation metrics, deterministic redaction, and a stdlib-only reference
analyzer. These intentionally do not add an LLM dependency, hosted service
dependency, or new model dependency. The developer handoff guide is
`MEETING_INTELLIGENCE.md`.

## Implementation Status

- Completed: `meeting` domain package with product-level artifacts,
  participant-aware transcript turns, validation helpers, and deterministic
  mapping from `diarization.IntegratedResult`.
- Completed: Markdown, product-facing JSON, SRT, and WebVTT exports from
  `meeting.Meeting`.
- Completed: provider-neutral `meeting.Analyzer` contract, analysis request and
  result types, default scope handling, analyzer output normalization, and
  validation for summaries, decisions, action items, follow-ups, risks, open
  questions, duplicate IDs, timing, statuses, and transcript evidence IDs.
- Completed: `evaluation` package with deterministic fixture metrics for
  transcript WER/CER, turn-level speaker attribution accuracy, action-item
  precision/recall/F1, and real-time factor.
- Completed: meeting redaction boundary, built-in stdlib pattern redactor, copy
  based `Meeting.Redact`, redaction reports, and export-time
  `RequireRedaction` enforcement for JSON, Markdown, SRT, and WebVTT.
- Completed: `meeting.HeuristicAnalyzer` as a stdlib-only deterministic
  reference analyzer, plus an end-to-end meeting-intelligence example.
- Completed: `MEETING_INTELLIGENCE.md` as the next-developer guide for data
  flow, ownership, analyzer extension, redaction/export safety, evaluation
  metrics, and validation commands.
- Completed: July 2026 performance sprint for local SDK overhead: direct
  PCM/WAV decode, range-first basic diarization segmentation, direct
  union-find clustering for current sizes, real root benchmarks, and
  meeting/evaluation benchmark coverage.
- Completed: repository maintenance baseline with `.gitignore`,
  golangci-lint v2, `make verify`, Gitea Actions CI, forbidden-file
  checks, external benchmark/profile output paths, and removal of tracked
  generated benchmark artifacts plus the stray `final-test` binary archive.
- Completed: v0.4.0 ASR streaming correctness scope with bounded VAD pre-roll,
  exact-once samples, explicit finalization, cumulative duration, per-session
  VAD ownership, retired-session guards, deterministic bounded admission,
  terminal native-state cleanup, idempotent shutdown, and model-native token
  timing separated from word segmentation.
- Completed (October 2026): Whisper language auto-detect and config-time
  validation of Whisper language/task; recognizer-level hotwords for
  transducer models; `evaluation` German normalizer (numbers, signs, ranges and
  unit symbols kept intact, versioned), `RTFx`, FLEURS/JSONL loaders, word
  error breakdown, and seeded paired bootstrap; meeting consent/retention metadata, `RequireConsent`, and DOCX
  export; an internal model manifest/verify/fetch draft (for gdk/ml) and
  pinned release URLs;
  extreme sample-rate coverage, WAV fuzzing, and two audio fixes; go-hnswlib
  v1.5.0; env-gated real-model benchmarks.
- Still open: LLM-quality analyzer implementations, export adapters for
  external tools beyond DOCX, retention enforcement, searchable meeting memory,
  DER benchmark corpora, latency dashboards, real capture/ingestion adapters,
  AAC/M4A decode, and the pebble v2 migration.

## Source Notes

Market references checked during the review:

- Otter: https://otter.ai/
- Fireflies: https://fireflies.ai/
- Fathom: https://www.fathom.ai/
- Granola: https://www.granola.ai/
- tl;dv: https://tldv.io/
- Microsoft Teams Recap: https://support.microsoft.com/en-us/teams/meetings/recap-in-microsoft-teams
- Google Workspace AI note taking: https://workspace.google.com/solutions/ai/ai-note-taking/
- Deepgram speech-to-text: https://deepgram.com/product/speech-to-text
- AssemblyAI speaker diarization: https://www.assemblyai.com/features/speaker-diarization
- ElevenLabs API: https://elevenlabs.io/api
- OpenAI audio docs: https://developers.openai.com/api/docs/guides/audio
- Sherpa-ONNX docs: https://k2-fsa.github.io/sherpa/onnx/index.html
