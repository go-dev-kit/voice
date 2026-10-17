package asr

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sherpa "github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/types"
)

func TestConfigValidateDisabledSkipsModelPaths(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = false
	config.Offline.ModelPath = ""

	if err := config.Validate(); err != nil {
		t.Fatalf("disabled ASR should skip model path validation: %v", err)
	}
}

func TestConfigValidateEnabledRequiresSherpaPaths(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true

	if err := config.Validate(); err == nil {
		t.Fatal("enabled ASR without Sherpa model paths should fail validation")
	}
}

func TestConfigValidateOfflineTransducerRequiresAllPaths(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOffline
	config.Offline = OfflineConfig{
		ModelFamily: OfflineFamilyTransducer,
		EncoderPath: tempModelFile(t, "encoder.onnx"),
		DecoderPath: tempModelFile(t, "decoder.onnx"),
	}

	if err := config.Validate(); err == nil {
		t.Fatal("offline transducer without tokens and joiner should fail")
	}
}

func TestBuildOfflineRecognizerConfigMapsWhisper(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOffline
	config.SampleRate = 16000
	config.FeatureDim = 80
	config.Provider = "cpu"
	config.NumThreads = 2
	config.Offline = OfflineConfig{
		ModelFamily:             OfflineFamilyWhisper,
		EncoderPath:             tempModelFile(t, "whisper-encoder.onnx"),
		DecoderPath:             tempModelFile(t, "whisper-decoder.onnx"),
		TokensPath:              tempModelFile(t, "whisper-tokens.txt"),
		Language:                "en",
		Task:                    "translate",
		TailPaddings:            4,
		EnableTokenTimestamps:   true,
		EnableSegmentTimestamps: true,
	}

	if err := config.Validate(); err != nil {
		t.Fatalf("whisper config should validate: %v", err)
	}
	mapped, err := buildOfflineRecognizerConfig(&config)
	if err != nil {
		t.Fatalf("failed to build offline recognizer config: %v", err)
	}
	if mapped.ModelConfig.Whisper.Encoder != config.Offline.EncoderPath {
		t.Fatalf("encoder path not mapped")
	}
	if mapped.ModelConfig.Whisper.Decoder != config.Offline.DecoderPath {
		t.Fatalf("decoder path not mapped")
	}
	if mapped.ModelConfig.Whisper.Task != "translate" {
		t.Fatalf("expected translate task, got %q", mapped.ModelConfig.Whisper.Task)
	}
	if mapped.ModelConfig.Whisper.EnableTokenTimestamps != 1 {
		t.Fatalf("expected token timestamps enabled")
	}
	if mapped.ModelConfig.NumThreads != 2 {
		t.Fatalf("expected two threads, got %d", mapped.ModelConfig.NumThreads)
	}
}

func TestSherpaOfflineModelTranscribeMapsResult(t *testing.T) {
	stream := &fakeOfflineStream{
		result: &sherpa.OfflineRecognizerResult{
			Text:       "hello world",
			Tokens:     []string{"hello", "world"},
			Timestamps: []float32{0, 0.5},
			Durations:  []float32{0.4, 0.3},
			Lang:       "en",
			Emotion:    "neutral",
			Event:      "speech",
		},
	}
	model := &SherpaOfflineModel{
		name:       "fake",
		language:   "auto",
		recognizer: &fakeOfflineRecognizer{stream: stream},
	}

	result, err := model.Transcribe(context.Background(), []float32{0.1, 0.2, 0.3, 0.4}, 2)
	if err != nil {
		t.Fatalf("transcription failed: %v", err)
	}
	if result.Text != "hello world" {
		t.Fatalf("unexpected text %q", result.Text)
	}
	if result.Language != "en" {
		t.Fatalf("unexpected language %q", result.Language)
	}
	if result.Emotion != "neutral" || result.Event != "speech" {
		t.Fatalf("expected emotion/event metadata to be mapped")
	}
	if result.EndTime != 2*time.Second {
		t.Fatalf("unexpected end time %s", result.EndTime)
	}
	if len(result.Tokens) != 2 {
		t.Fatalf("expected two model-native tokens, got %d", len(result.Tokens))
	}
	if len(result.Words) != 0 {
		t.Fatalf("tokens must not be exposed as segmented words, got %d words", len(result.Words))
	}
	if result.Tokens[0].Text != "hello" || !result.Tokens[0].HasTiming || result.Tokens[1].StartTime != 500*time.Millisecond || result.Tokens[1].Duration != 300*time.Millisecond {
		t.Fatalf("unexpected token mapping: %+v", result.Tokens)
	}
	if stream.acceptedRate != 2 || len(stream.acceptedSamples) != 4 {
		t.Fatalf("offline stream did not receive audio")
	}
}

func TestSherpaOfflineModelRejectsEmptyAudio(t *testing.T) {
	model := &SherpaOfflineModel{recognizer: &fakeOfflineRecognizer{stream: &fakeOfflineStream{}}}
	if _, err := model.Transcribe(context.Background(), nil, 16000); err == nil {
		t.Fatal("empty audio should fail before native calls")
	}
}

func TestSherpaOfflineModelCloseIdempotent(t *testing.T) {
	recognizer := &fakeOfflineRecognizer{stream: &fakeOfflineStream{}}
	model := &SherpaOfflineModel{recognizer: recognizer}

	if err := model.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}
	if err := model.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
	if recognizer.closeCount != 1 {
		t.Fatalf("expected one native close, got %d", recognizer.closeCount)
	}
}

func TestVADProviderSelectionEnergy(t *testing.T) {
	service, err := NewVADService(&VADConfig{
		Provider:   VADProviderEnergy,
		Threshold:  0.01,
		SampleRate: 16000,
	})
	if err != nil {
		t.Fatalf("failed to create energy VAD: %v", err)
	}
	defer service.Close()

	result, err := service.Process([]float32{0.5, -0.5, 0.25}, nil)
	if err != nil {
		t.Fatalf("energy VAD failed: %v", err)
	}
	if !result.IsSpeech {
		t.Fatal("expected energy VAD to detect speech")
	}
}

func TestVADProviderSelectionSherpaRequiresModelPath(t *testing.T) {
	_, err := NewVADService(&VADConfig{
		Provider:   VADProviderSilero,
		SampleRate: 16000,
	})
	if err == nil {
		t.Fatal("Sherpa VAD without model path should fail")
	}
}

func TestASRServiceModelRegistrationAndSelection(t *testing.T) {
	service := newTestService()
	defer service.Close()

	fastModel := &testModel{name: "fast_model", language: "en", quantization: "int8", latency: 50 * time.Millisecond}
	accurateModel := &testModel{name: "accurate_model", language: "en", quantization: "float32", latency: 500 * time.Millisecond}

	if err := service.RegisterModel(fastModel); err != nil {
		t.Fatalf("failed to register fast model: %v", err)
	}
	if err := service.RegisterModel(accurateModel); err != nil {
		t.Fatalf("failed to register accurate model: %v", err)
	}

	selected, err := service.SelectModel("en", &types.ModelRequirements{PreferSpeed: true})
	if err != nil {
		t.Fatalf("failed to select model: %v", err)
	}
	if selected.Name() != "fast_model" {
		t.Fatalf("expected fast model, got %s", selected.Name())
	}
}

func TestASRServiceProcessAudioWithExplicitFakeModel(t *testing.T) {
	service := newTestService()
	defer service.Close()

	model := &testModel{name: "fake", language: "en", quantization: "int8", latency: 0}
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register fake model: %v", err)
	}

	audio := make([]float32, 16000)
	result, err := service.ProcessAudioChunk(context.Background(), "test-session", audio)
	if err != nil {
		t.Fatalf("failed to process audio chunk: %v", err)
	}
	if result == nil {
		t.Fatal("expected transcription result")
	}
	if result.Text != "test transcript" {
		t.Fatalf("expected fake transcript, got %q", result.Text)
	}
}

func TestASRServiceStreamsEachSampleExactlyOnce(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)

	service := newTestService()
	service.config.DefaultModel = model.Name()
	defer service.Close()
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	chunks := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	for _, chunk := range chunks {
		if _, err := service.ProcessAudioChunk(context.Background(), "exactly-once", chunk); err != nil {
			t.Fatalf("process chunk %v: %v", chunk, err)
		}
	}
	if _, err := service.FinishStream(context.Background(), "exactly-once"); err != nil {
		t.Fatalf("finish stream: %v", err)
	}

	if len(recognizer.streams) != 1 {
		t.Fatalf("native streams = %d, want 1", len(recognizer.streams))
	}
	stream := recognizer.streams[0]
	if want := []float32{1, 2, 3, 4, 5, 6}; !slices.Equal(stream.samples, want) {
		t.Fatalf("native samples = %v, want %v", stream.samples, want)
	}
	if stream.acceptCount != len(chunks) {
		t.Fatalf("AcceptWaveform calls = %d, want %d", stream.acceptCount, len(chunks))
	}
	if stream.inputFinishedCount != 1 {
		t.Fatalf("InputFinished calls = %d, want 1", stream.inputFinishedCount)
	}
}

func TestASRServiceDoesNotTruncateChunkLargerThanRollingBuffer(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)

	service := newTestService()
	service.config.DefaultModel = model.Name()
	defer service.Close()
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	audio := make([]float32, service.streaming.config.BufferSize+17)
	for i := range audio {
		audio[i] = float32(i + 1)
	}
	if _, err := service.ProcessAudioChunk(context.Background(), "long-chunk", audio); err != nil {
		t.Fatalf("process long chunk: %v", err)
	}
	if _, err := service.FinishStream(context.Background(), "long-chunk"); err != nil {
		t.Fatalf("finish stream: %v", err)
	}

	if got := recognizer.streams[0].samples; !slices.Equal(got, audio) {
		t.Fatalf("native recognizer received %d samples, want the complete %d-sample chunk", len(got), len(audio))
	}
}

func TestASRServiceUsesOneVADPerSessionAndFeedsIncrementalAudio(t *testing.T) {
	service := newTestService()
	defer service.Close()
	if err := service.RegisterModel(&testModel{name: "fake", language: "en", quantization: "int8"}); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	config := &VADConfig{Provider: VADProviderEnergy, Threshold: 0.01, SampleRate: 16000}
	prototypeDetector := &recordingVADDetector{}
	service.vadConfig = config
	service.vadPrototype = &VADService{config: config, detector: prototypeDetector}

	firstChunk := []float32{0.5, 0.5}
	secondChunk := []float32{0.6, 0.6}
	if _, err := service.ProcessAudioChunk(context.Background(), "session-a", firstChunk); err != nil {
		t.Fatalf("first session chunk: %v", err)
	}
	if _, err := service.ProcessAudioChunk(context.Background(), "session-a", secondChunk); err != nil {
		t.Fatalf("second session chunk: %v", err)
	}
	if _, err := service.ProcessAudioChunk(context.Background(), "session-b", firstChunk); err != nil {
		t.Fatalf("other session chunk: %v", err)
	}

	service.streaming.mu.RLock()
	sessionA := service.streaming.sessions["session-a"]
	sessionB := service.streaming.sessions["session-b"]
	service.streaming.mu.RUnlock()
	if sessionA == nil || sessionB == nil {
		t.Fatal("expected both streaming sessions")
	}
	if sessionA.vadService == sessionB.vadService {
		t.Fatal("sessions must not share a stateful VAD service")
	}
	if len(prototypeDetector.calls) != 2 || !slices.Equal(prototypeDetector.calls[0], firstChunk) || !slices.Equal(prototypeDetector.calls[1], secondChunk) {
		t.Fatalf("VAD calls = %v, want the two caller chunks without rolling replay", prototypeDetector.calls)
	}
}

func TestASRServicePreservesVADPreRollUntilSpeechActivation(t *testing.T) {
	service := newTestService()
	defer service.Close()
	model := &finalizableTestModel{
		testModel: testModel{name: "fake", language: "en", quantization: "int8"},
	}
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	config := &VADConfig{Provider: VADProviderEnergy, SampleRate: 16000}
	detector := &sequenceVADDetector{speech: []bool{false, false, true}}
	service.vadConfig = config
	service.vadPrototype = &VADService{config: config, detector: detector}

	for _, chunk := range [][]float32{{1}, {2}, {3}} {
		if _, err := service.ProcessAudioChunk(context.Background(), "delayed-speech", chunk); err != nil {
			t.Fatalf("process chunk %v: %v", chunk, err)
		}
	}

	if model.processCalls != 1 {
		t.Fatalf("model process calls = %d, want 1 after VAD activation", model.processCalls)
	}
	if want := []float32{1, 2, 3}; !slices.Equal(model.processedAudio, want) {
		t.Fatalf("ASR samples = %v, want VAD pre-roll %v", model.processedAudio, want)
	}
}

func TestASRServicePreservesCompleteLargeVADActivationChunk(t *testing.T) {
	service := newTestService()
	defer service.Close()
	model := &finalizableTestModel{
		testModel: testModel{name: "fake", language: "en", quantization: "int8"},
	}
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	config := &VADConfig{Provider: VADProviderEnergy, SampleRate: 16000}
	service.vadConfig = config
	service.vadPrototype = &VADService{
		config:   config,
		detector: &sequenceVADDetector{speech: []bool{true}},
	}

	audio := make([]float32, service.streaming.config.BufferSize+17)
	for i := range audio {
		audio[i] = float32(i + 1)
	}
	if _, err := service.ProcessAudioChunk(context.Background(), "large-vad-activation", audio); err != nil {
		t.Fatalf("process activation chunk: %v", err)
	}
	if !slices.Equal(model.processedAudio, audio) {
		t.Fatalf("ASR received %d samples, want complete %d-sample activation chunk", len(model.processedAudio), len(audio))
	}
}

func TestASRServiceDiscardsNativeStateWhenProcessingIsCanceledAfterAcceptance(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	recognizer := &fakeOnlineRecognizer{ready: true, onAccept: cancel}
	model := newSherpaOnlineModelForTest(recognizer)

	service := newTestServiceWithCapacity(1)
	service.config.DefaultModel = model.Name()
	defer service.Close()
	if err := service.RegisterModel(model); err != nil {
		t.Fatalf("failed to register model: %v", err)
	}

	if _, err := service.ProcessAudioChunk(ctx, "canceled", []float32{1, 2}); !errors.Is(err, context.Canceled) {
		t.Fatalf("processing error = %v, want context cancellation", err)
	}
	if len(recognizer.streams) != 1 || recognizer.streams[0].closeCount != 1 {
		t.Fatalf("canceled native stream was not closed exactly once: %+v", recognizer.streams)
	}
	if len(model.sessions) != 0 {
		t.Fatalf("canceled native session remains registered: %d", len(model.sessions))
	}
	service.streaming.mu.RLock()
	session := service.streaming.sessions["canceled"]
	service.streaming.mu.RUnlock()
	session.mu.Lock()
	state := session.State.ASRState
	session.mu.Unlock()
	if state != nil {
		t.Fatalf("canceled ASR state was retained: %T", state)
	}
	recognizer.ready = false
	if _, err := service.ProcessAudioChunk(context.Background(), "replacement", []float32{3}); err != nil {
		t.Fatalf("canceled operation must release capacity after cleanup: %v", err)
	}
}

func TestSherpaOnlineModelReusesNativeStreamAcrossChunks(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)
	defer model.Close()

	state := &types.StreamingState{Language: "en"}
	if _, err := model.ProcessAudio(context.Background(), []float32{0.1, 0.2}, state); err != nil {
		t.Fatalf("first chunk failed: %v", err)
	}
	if _, err := model.ProcessAudio(context.Background(), []float32{0.3, 0.4}, state); err != nil {
		t.Fatalf("second chunk failed: %v", err)
	}

	if recognizer.newStreamCount != 1 {
		t.Fatalf("expected one native stream, got %d", recognizer.newStreamCount)
	}
	stream := recognizer.streams[0]
	if stream.acceptCount != 2 {
		t.Fatalf("expected two accepted chunks, got %d", stream.acceptCount)
	}
	if stream.inputFinishedCount != 0 {
		t.Fatalf("ordinary chunks must not finish the stream")
	}
	if stream.closeCount != 0 {
		t.Fatalf("stream should stay open across chunks")
	}
}

func TestSherpaOnlineModelFinishAudioClosesNativeStream(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)
	defer model.Close()

	state := &types.StreamingState{Language: "en"}
	if _, err := model.ProcessAudio(context.Background(), []float32{0.1}, state); err != nil {
		t.Fatalf("first chunk failed: %v", err)
	}
	result, err := model.FinishAudio(context.Background(), nil, state)
	if err != nil {
		t.Fatalf("finish failed: %v", err)
	}
	if result.IsPartial {
		t.Fatalf("finished result should be final")
	}
	if state.ASRState != nil {
		t.Fatalf("finished stream should be removed from state")
	}

	stream := recognizer.streams[0]
	if stream.inputFinishedCount != 1 {
		t.Fatalf("expected one InputFinished call, got %d", stream.inputFinishedCount)
	}
	if stream.closeCount != 1 {
		t.Fatalf("expected stream close after finish, got %d", stream.closeCount)
	}
	if !slices.Equal(stream.samples, []float32{0.1}) {
		t.Fatalf("finish replayed audio: got native samples %v", stream.samples)
	}
}

func TestSherpaOnlineModelReportsCumulativeStreamDuration(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{}
	model := newSherpaOnlineModelForTest(recognizer)
	defer model.Close()

	state := &types.StreamingState{Language: "en"}
	first, err := model.ProcessAudio(context.Background(), make([]float32, 8000), state)
	if err != nil {
		t.Fatalf("first chunk: %v", err)
	}
	if first.EndTime != 500*time.Millisecond {
		t.Fatalf("first end time = %s, want 500ms", first.EndTime)
	}
	second, err := model.ProcessAudio(context.Background(), make([]float32, 8000), state)
	if err != nil {
		t.Fatalf("second chunk: %v", err)
	}
	if second.EndTime != time.Second {
		t.Fatalf("second end time = %s, want 1s cumulative", second.EndTime)
	}
}

func TestSherpaOnlineModelExposesTokensWithoutPretendingTheyAreWords(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{
		result: &sherpa.OnlineRecognizerResult{
			Text:       "hello",
			Tokens:     []string{"▁hel", "lo", "!"},
			Timestamps: []float32{0, 0.5},
		},
	}
	model := newSherpaOnlineModelForTest(recognizer)
	defer model.Close()

	result, err := model.ProcessAudio(context.Background(), []float32{0.1}, &types.StreamingState{Language: "en"})
	if err != nil {
		t.Fatalf("process audio: %v", err)
	}
	if len(result.Tokens) != 3 {
		t.Fatalf("tokens = %d, want 3", len(result.Tokens))
	}
	if result.Tokens[0].Text != "▁hel" || !result.Tokens[0].HasTiming || result.Tokens[1].Text != "lo" || result.Tokens[1].StartTime != 500*time.Millisecond || result.Tokens[2].Text != "!" || result.Tokens[2].HasTiming {
		t.Fatalf("unexpected tokens: %+v", result.Tokens)
	}
	if len(result.Words) != 0 {
		t.Fatalf("online model-native tokens must not populate Words: %+v", result.Words)
	}
}

func TestSherpaOnlineModelEndpointResetsWithoutFinishing(t *testing.T) {
	recognizer := &fakeOnlineRecognizer{endpoint: true}
	model := newSherpaOnlineModelForTest(recognizer)
	defer model.Close()

	result, err := model.ProcessAudio(context.Background(), []float32{0.1}, &types.StreamingState{Language: "en"})
	if err != nil {
		t.Fatalf("process failed: %v", err)
	}
	if result.IsPartial {
		t.Fatalf("endpoint result should be final")
	}
	if recognizer.resetCount != 1 {
		t.Fatalf("expected one endpoint reset, got %d", recognizer.resetCount)
	}
	stream := recognizer.streams[0]
	if stream.inputFinishedCount != 0 {
		t.Fatalf("endpoint reset must not call InputFinished")
	}
	if stream.closeCount != 0 {
		t.Fatalf("endpoint reset should keep the stream open")
	}
}

func TestProcessFinalAudioPrefersFinalizableModel(t *testing.T) {
	model := &finalizableTestModel{
		testModel: testModel{name: "finalizable", language: "en", quantization: "int8"},
	}

	result, err := processFinalAudio(context.Background(), model, []float32{0.1}, &types.StreamingState{Language: "en"})
	if err != nil {
		t.Fatalf("final processing failed: %v", err)
	}
	if result.IsPartial {
		t.Fatalf("expected final result")
	}
	if model.finishCalls != 1 || model.processCalls != 0 {
		t.Fatalf("expected FinishAudio only, got finish=%d process=%d", model.finishCalls, model.processCalls)
	}
}

func TestStreamingManagerClosesASRStateOnSessionRemoval(t *testing.T) {
	manager := NewStreamingManager(nil)
	defer manager.Close()

	state, err := manager.GetState("session")
	if err != nil {
		t.Fatalf("failed to get state: %v", err)
	}
	closer := &countingCloser{}
	state.ASRState = closer
	vadDetector := &recordingVADDetector{}
	manager.mu.RLock()
	session := manager.sessions["session"]
	manager.mu.RUnlock()
	session.mu.Lock()
	session.vadService = &VADService{detector: vadDetector}
	session.mu.Unlock()

	if err := manager.RemoveSession("session"); err != nil {
		t.Fatalf("failed to remove session: %v", err)
	}
	if closer.closeCount != 1 {
		t.Fatalf("expected ASR state close on removal, got %d", closer.closeCount)
	}
	if state.ASRState != nil {
		t.Fatalf("ASR state should be cleared after removal")
	}
	if !vadDetector.closed {
		t.Fatal("expected session VAD service to close on removal")
	}
}

func TestStreamingManagerCloseClosesASRStates(t *testing.T) {
	manager := NewStreamingManager(nil)
	state, err := manager.GetState("session")
	if err != nil {
		t.Fatalf("failed to get state: %v", err)
	}
	closer := &countingCloser{}
	state.ASRState = closer
	vadDetector := &recordingVADDetector{}
	manager.mu.RLock()
	session := manager.sessions["session"]
	manager.mu.RUnlock()
	session.mu.Lock()
	session.vadService = &VADService{detector: vadDetector}
	session.mu.Unlock()

	if err := manager.Close(); err != nil {
		t.Fatalf("failed to close manager: %v", err)
	}
	if closer.closeCount != 1 {
		t.Fatalf("expected ASR state close, got %d", closer.closeCount)
	}
	if state.ASRState != nil {
		t.Fatalf("ASR state should be cleared after close")
	}
	if !vadDetector.closed {
		t.Fatal("expected session VAD service to close with manager")
	}
}

func TestStreamingManagerRejectsSessionsAfterClose(t *testing.T) {
	manager := NewStreamingManager(nil)
	if err := manager.Close(); err != nil {
		t.Fatalf("close manager: %v", err)
	}
	if _, err := manager.GetState("late-session"); err == nil {
		t.Fatal("closed streaming manager must reject new sessions")
	}
	if err := manager.Close(); err != nil {
		t.Fatalf("second close must remain idempotent: %v", err)
	}
}

func TestASRServiceEnforcesStreamCapacityAndReleasesFinalizedSlot(t *testing.T) {
	service := newTestServiceWithCapacity(2)
	defer service.Close()
	if err := service.RegisterModel(&testModel{name: "fake", language: "en", quantization: "int8"}); err != nil {
		t.Fatalf("register model: %v", err)
	}

	ctx := context.Background()
	for _, sessionID := range []string{"one", "two"} {
		if _, err := service.ProcessAudioChunk(ctx, sessionID, []float32{0.1}); err != nil {
			t.Fatalf("admit %s: %v", sessionID, err)
		}
	}
	if _, err := service.ProcessAudioChunk(ctx, "two", []float32{0.2}); err != nil {
		t.Fatalf("existing admitted stream must continue at capacity: %v", err)
	}

	_, err := service.ProcessAudioChunk(ctx, "three", []float32{0.3})
	if err == nil {
		t.Fatal("expected the third concurrent stream to be rejected")
	}
	var capacityErr *StreamCapacityError
	if !errors.As(err, &capacityErr) {
		t.Fatalf("expected a typed StreamCapacityError, got %T: %v", err, err)
	}
	if capacityErr.Limit != 2 || capacityErr.Active != 2 {
		t.Fatalf("unexpected capacity error: %+v", capacityErr)
	}

	if _, err := service.FinishStream(ctx, "one"); err != nil {
		t.Fatalf("finish first stream: %v", err)
	}
	if _, err := service.ProcessAudioChunk(ctx, "three", []float32{0.3}); err != nil {
		t.Fatalf("finalization should release one admission slot: %v", err)
	}
	if got := service.streaming.GetActiveSessions(); got != 2 {
		t.Fatalf("active streams = %d, want 2", got)
	}
}

func TestStreamingManagerConcurrentAdmissionNeverExceedsLimit(t *testing.T) {
	config := &types.StreamingConfig{
		MaxConcurrentStreams: 3,
		ChunkSize:            16,
		BufferSize:           48,
		SampleRate:           16000,
		StreamTimeout:        time.Minute,
		IdleTimeout:          time.Minute,
	}
	manager := NewStreamingManager(config)
	defer manager.Close()

	var admitted atomic.Int32
	var rejected atomic.Int32
	var invalid atomic.Int32
	var wg sync.WaitGroup
	for i := range 24 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := manager.getOrCreateSession(string(rune('a' + i))); err != nil {
				var capacityErr *StreamCapacityError
				if errors.As(err, &capacityErr) {
					rejected.Add(1)
					return
				}
				invalid.Add(1)
				return
			}
			admitted.Add(1)
		}()
	}
	wg.Wait()

	if got := admitted.Load(); got != 3 {
		t.Fatalf("admitted = %d, want 3", got)
	}
	if got := rejected.Load(); got != 21 {
		t.Fatalf("capacity rejections = %d, want 21", got)
	}
	if got := invalid.Load(); got != 0 {
		t.Fatalf("unexpected non-capacity errors = %d", got)
	}
	if got := manager.GetActiveSessions(); got != 3 {
		t.Fatalf("active streams = %d, want 3", got)
	}
}

func TestStreamingManagerKeepsCapacityForQueuedSameSessionOperation(t *testing.T) {
	config := &types.StreamingConfig{
		MaxConcurrentStreams: 1,
		ChunkSize:            16,
		BufferSize:           48,
		SampleRate:           16000,
		StreamTimeout:        time.Minute,
		IdleTimeout:          time.Minute,
	}
	manager := NewStreamingManager(config)
	defer manager.Close()

	first, err := manager.acquireSession("one")
	if err != nil {
		t.Fatalf("acquire first operation: %v", err)
	}
	second, err := manager.acquireSession("one")
	if err != nil {
		t.Fatalf("queue same-session operation: %v", err)
	}
	if first != second {
		t.Fatal("same session ID must reuse one session")
	}

	first.mu.Lock()
	manager.completeSessionOperation(first, true)
	first.mu.Unlock()
	if _, err := manager.getOrCreateSession("two"); err == nil {
		t.Fatal("a queued same-session operation must retain the admission slot")
	} else {
		var capacityErr *StreamCapacityError
		if !errors.As(err, &capacityErr) {
			t.Fatalf("expected capacity error while queued work remains, got %v", err)
		}
	}

	second.mu.Lock()
	manager.completeSessionOperation(second, false)
	second.mu.Unlock()
	if _, err := manager.getOrCreateSession("two"); err == nil {
		t.Fatal("a partial operation after finalization must retain the admission slot")
	}

	last, err := manager.acquireSession("one")
	if err != nil {
		t.Fatalf("acquire final operation: %v", err)
	}
	last.mu.Lock()
	manager.completeSessionOperation(last, true)
	last.mu.Unlock()
	if _, err := manager.getOrCreateSession("two"); err != nil {
		t.Fatalf("a final operation with no queued successor should release the slot: %v", err)
	}
}

func TestStreamingManagerRetiresAcquiredSessionBeforeRemovalCleanup(t *testing.T) {
	manager := NewStreamingManager(nil)
	defer manager.Close()

	session, err := manager.acquireSession("retired")
	if err != nil {
		t.Fatalf("acquire session: %v", err)
	}
	if err := manager.RemoveSession("retired"); err != nil {
		t.Fatalf("remove session: %v", err)
	}

	session.mu.Lock()
	retired := session.retired
	manager.completeSessionOperation(session, false)
	session.mu.Unlock()
	if !retired {
		t.Fatal("an unlinked session must be retired before queued work resumes")
	}
	if got := manager.GetActiveSessions(); got != 0 {
		t.Fatalf("active streams = %d, want 0 after removal", got)
	}
}

func BenchmarkASRService_ProcessAudioChunk(b *testing.B) {
	service := newTestService()
	defer service.Close()

	if err := service.RegisterModel(&testModel{name: "fake", language: "en", quantization: "int8"}); err != nil {
		b.Fatalf("failed to register fake model: %v", err)
	}

	audio := make([]float32, 16000)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := service.ProcessAudioChunk(ctx, "bench-session", audio)
		if err != nil {
			b.Fatalf("failed to process audio chunk: %v", err)
		}
	}
}

func newTestService() *Service {
	return newTestServiceWithCapacity(10)
}

func newTestServiceWithCapacity(maxConcurrentStreams int) *Service {
	config := DefaultConfig()
	config.Enabled = true
	config.Backend = BackendSherpaOnline
	config.DefaultModel = "fake"
	config.VADProvider = VADProviderNone
	config.MaxConcurrentStreams = maxConcurrentStreams

	streamingConfig := &types.StreamingConfig{
		MaxConcurrentStreams: config.MaxConcurrentStreams,
		ChunkSize:            config.ChunkSize,
		OverlapSize:          config.ChunkSize / 10,
		BufferSize:           config.ChunkSize * 3,
		SampleRate:           config.SampleRate,
		StreamTimeout:        time.Duration(config.StreamTimeout) * time.Second,
		IdleTimeout:          30 * time.Second,
		FlushInterval:        100 * time.Millisecond,
		PartialResults:       true,
		StabilityThreshold:   0.8,
		MinConfidence:        0.5,
	}

	return &Service{
		config:    &config,
		models:    make(map[string]Model),
		streaming: NewStreamingManager(streamingConfig),
		metrics:   &ASRMetrics{},
	}
}

func tempModelFile(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
		t.Fatalf("failed to create temp model file: %v", err)
	}
	return path
}

type fakeOfflineRecognizer struct {
	stream     *fakeOfflineStream
	closeCount int
}

func (r *fakeOfflineRecognizer) NewStream() (offlineStream, error) {
	return r.stream, nil
}

func (r *fakeOfflineRecognizer) Decode(stream offlineStream) error {
	return nil
}

func (r *fakeOfflineRecognizer) Close() error {
	r.closeCount++
	return nil
}

type fakeOfflineStream struct {
	result          *sherpa.OfflineRecognizerResult
	acceptedRate    int
	acceptedSamples []float32
	closeCount      int
}

func (s *fakeOfflineStream) AcceptWaveform(sampleRate int, samples []float32) error {
	s.acceptedRate = sampleRate
	s.acceptedSamples = append([]float32(nil), samples...)
	return nil
}

func (s *fakeOfflineStream) Result() *sherpa.OfflineRecognizerResult {
	return s.result
}

func (s *fakeOfflineStream) Close() error {
	s.closeCount++
	return nil
}

type testModel struct {
	name         string
	language     string
	quantization string
	latency      time.Duration
}

func (m *testModel) Name() string         { return m.name }
func (m *testModel) Language() string     { return m.language }
func (m *testModel) Quantization() string { return m.quantization }
func (m *testModel) SupportsLanguage(lang string) bool {
	return lang == "" || m.language == lang
}
func (m *testModel) Latency() time.Duration { return m.latency }
func (m *testModel) Close() error           { return nil }
func (m *testModel) ProcessAudio(ctx context.Context, audio []float32, state *types.StreamingState) (*types.Transcription, error) {
	return &types.Transcription{
		Text:      "test transcript",
		IsPartial: true,
		Language:  m.language,
		Timestamp: time.Now(),
	}, nil
}

type finalizableTestModel struct {
	testModel
	processCalls   int
	finishCalls    int
	processedAudio []float32
	finishAudio    []float32
}

func (m *finalizableTestModel) ProcessAudio(ctx context.Context, audio []float32, state *types.StreamingState) (*types.Transcription, error) {
	m.processCalls++
	m.processedAudio = append(m.processedAudio, audio...)
	return m.testModel.ProcessAudio(ctx, audio, state)
}

func (m *finalizableTestModel) FinishAudio(ctx context.Context, audio []float32, state *types.StreamingState) (*types.Transcription, error) {
	m.finishCalls++
	m.finishAudio = append([]float32(nil), audio...)
	return &types.Transcription{
		Text:      "final transcript",
		IsPartial: false,
		Language:  m.language,
		Timestamp: time.Now(),
	}, nil
}

type recordingVADDetector struct {
	calls  [][]float32
	closed bool
}

type sequenceVADDetector struct {
	speech []bool
	calls  int
}

func (d *sequenceVADDetector) Process(_ []float32, state any) (*types.VADResult, error) {
	index := d.calls
	d.calls++
	isSpeech := index < len(d.speech) && d.speech[index]
	return &types.VADResult{IsSpeech: isSpeech, State: state}, nil
}

func (d *sequenceVADDetector) Close() error { return nil }

func (d *recordingVADDetector) Process(audio []float32, state any) (*types.VADResult, error) {
	d.calls = append(d.calls, append([]float32(nil), audio...))
	return &types.VADResult{IsSpeech: true, State: state}, nil
}

func (d *recordingVADDetector) Close() error {
	d.closed = true
	return nil
}

type countingCloser struct {
	closeCount int
}

func (c *countingCloser) Close() error {
	c.closeCount++
	return nil
}

func newSherpaOnlineModelForTest(recognizer *fakeOnlineRecognizer) *SherpaOnlineModel {
	return &SherpaOnlineModel{
		name:         "fake-sherpa-online",
		language:     "en",
		quantization: "int8",
		sampleRate:   16000,
		recognizer:   recognizer,
		sessions:     make(map[*onlineSession]struct{}),
	}
}

type fakeOnlineRecognizer struct {
	streams        []*fakeOnlineStream
	newStreamCount int
	decodeCount    int
	resetCount     int
	closeCount     int
	endpoint       bool
	ready          bool
	onAccept       func()
	result         *sherpa.OnlineRecognizerResult
}

func (r *fakeOnlineRecognizer) NewStream() (onlineStream, error) {
	r.newStreamCount++
	stream := &fakeOnlineStream{onAccept: r.onAccept}
	r.streams = append(r.streams, stream)
	return stream, nil
}

func (r *fakeOnlineRecognizer) Decode(stream onlineStream) error {
	r.decodeCount++
	return nil
}

func (r *fakeOnlineRecognizer) IsReady(stream onlineStream) bool {
	return r.ready
}

func (r *fakeOnlineRecognizer) IsEndpoint(stream onlineStream) bool {
	return r.endpoint
}

func (r *fakeOnlineRecognizer) Reset(stream onlineStream) error {
	r.resetCount++
	r.endpoint = false
	return nil
}

func (r *fakeOnlineRecognizer) Result(stream onlineStream) (*sherpa.OnlineRecognizerResult, error) {
	if r.result != nil {
		return r.result, nil
	}
	return &sherpa.OnlineRecognizerResult{Text: "stream result"}, nil
}

func (r *fakeOnlineRecognizer) Close() error {
	r.closeCount++
	return nil
}

type fakeOnlineStream struct {
	acceptCount        int
	inputFinishedCount int
	closeCount         int
	sampleRate         int
	samples            []float32
	onAccept           func()
}

func (s *fakeOnlineStream) AcceptWaveform(sampleRate int, samples []float32) error {
	if s.inputFinishedCount > 0 {
		return nil
	}
	s.acceptCount++
	s.sampleRate = sampleRate
	s.samples = append(s.samples, samples...)
	if s.onAccept != nil {
		s.onAccept()
		s.onAccept = nil
	}
	return nil
}

func (s *fakeOnlineStream) InputFinished() error {
	s.inputFinishedCount++
	return nil
}

func (s *fakeOnlineStream) Close() error {
	s.closeCount++
	return nil
}

func TestSherpaOfflineModelRefusesAudioOverItsLimit(t *testing.T) {
	stream := &fakeOfflineStream{result: &sherpa.OfflineRecognizerResult{Text: "ok"}}
	model := &SherpaOfflineModel{recognizer: &fakeOfflineRecognizer{stream: stream}, maxDecode: 30 * time.Second}

	_, err := model.Transcribe(context.Background(), make([]float32, 31*16000), 16000)
	var tooLong *AudioTooLongError
	if !errors.As(err, &tooLong) || tooLong.Limit != 30*time.Second || tooLong.Duration != 31*time.Second {
		t.Fatalf("got %v, want *AudioTooLongError 31s > 30s", err)
	}
	if !strings.Contains(err.Error(), "at most 30s") || stream.acceptedSamples != nil {
		t.Fatalf("refused audio still reached the decoder or message unclear: %v", err)
	}
	if _, err := model.Transcribe(context.Background(), make([]float32, 30*16000), 16000); err != nil {
		t.Fatalf("audio at the limit: %v", err)
	}
}

func TestSherpaOfflineModelChecksContextAfterQueueing(t *testing.T) {
	stream := &fakeOfflineStream{}
	model := &SherpaOfflineModel{recognizer: &fakeOfflineRecognizer{stream: stream}}
	model.mu.Lock()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := model.Transcribe(ctx, []float32{0.1}, 16000)
		done <- err
	}()
	time.Sleep(20 * time.Millisecond) // let the call queue on the lock
	cancel()
	model.mu.Unlock()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want context.Canceled", err)
	}
	if stream.acceptedSamples != nil {
		t.Fatal("a canceled caller still decoded")
	}
}

func TestOfflineConfigDecodeLimitAndChunkOptions(t *testing.T) {
	if got := (OfflineConfig{ModelFamily: OfflineFamilyWhisper}).maxDecodeSeconds(); got != 30 {
		t.Fatalf("whisper default limit %v, want 30", got)
	}
	if got := (OfflineConfig{ModelFamily: OfflineFamilyTransducer}).maxDecodeSeconds(); got != 0 {
		t.Fatalf("transducer default limit %v, want none", got)
	}
	if got := (OfflineConfig{ModelFamily: OfflineFamilyWhisper, MaxDecodeSeconds: 12}).maxDecodeSeconds(); got != 12 {
		t.Fatalf("explicit limit %v, want 12", got)
	}

	if got := (OfflineConfig{}).ChunkOptions(); got != DefaultChunkOptions() {
		t.Fatalf("zero config chunking %+v, want defaults", got)
	}
	got := OfflineConfig{MaxChunkSeconds: 20, ChunkOverlapSeconds: 1.5, SpeechPaddingSeconds: 0.1, SpeechMergeGapSeconds: 0.5}.ChunkOptions()
	want := ChunkOptions{MaxChunk: 20 * time.Second, Overlap: 1500 * time.Millisecond, Padding: 100 * time.Millisecond, MergeGap: 500 * time.Millisecond, CutSearch: DefaultChunkOptions().CutSearch}
	if got != want {
		t.Fatalf("ChunkOptions() = %+v, want %+v", got, want)
	}
}

func TestValidateOfflineChunkingAndLimit(t *testing.T) {
	for name, mutate := range map[string]func(*Config){
		"chunk over cap":   func(c *Config) { c.Offline.MaxChunkSeconds = 40 },
		"negative limit":   func(c *Config) { c.Offline.MaxDecodeSeconds = -1 },
		"overlap too wide": func(c *Config) { c.Offline.MaxChunkSeconds = 10; c.Offline.ChunkOverlapSeconds = 6 },
	} {
		t.Run(name, func(t *testing.T) {
			config := DefaultConfig()
			config.Enabled = true
			config.Offline.ModelFamily = OfflineFamilySenseVoice
			config.Offline.ModelPath = tempModelFile(t, "model.onnx")
			config.Offline.TokensPath = tempModelFile(t, "tokens.txt")
			mutate(&config)
			if err := config.Validate(); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestNewOfflineTranscriberRejectsInvalidConfig(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	if _, err := NewOfflineTranscriber(&config); err == nil {
		t.Fatal("expected error without model paths")
	}
	if _, err := NewOfflineTranscriber(nil); err == nil {
		t.Fatal("expected error for nil config")
	}
}
