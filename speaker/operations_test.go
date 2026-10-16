package speaker

import (
	"context"
	"errors"
	"testing"
)

func TestContextAwareMethodsRejectNilContext(t *testing.T) {
	var manager Manager

	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if err := manager.RegisterSpeakerContext(nil, "id", "name", []float32{0.1}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from RegisterSpeakerContext, got %v", err)
	}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := manager.IdentifySpeakerContext(nil, []float32{0.1}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from IdentifySpeakerContext, got %v", err)
	}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := manager.VerifySpeakerContext(nil, "id", []float32{0.1}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from VerifySpeakerContext, got %v", err)
	}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if speakers := manager.GetAllSpeakersContext(nil); len(speakers) != 0 {
		t.Fatalf("expected empty speaker list for nil context, got %d", len(speakers))
	}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if err := manager.DeleteSpeakerContext(nil, "id"); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from DeleteSpeakerContext, got %v", err)
	}
	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := manager.ExtractEmbedding(nil, []float32{0.1}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from ExtractEmbedding, got %v", err)
	}
}

func TestSherpaEmbeddingExtractorAdapterCreatesFreshStreamPerExtraction(t *testing.T) {
	extractor := &fakeSpeakerEmbeddingExtractor{
		ready:     true,
		embedding: []float32{0.1, 0.2, 0.3},
	}
	adapter := NewSherpaEmbeddingExtractorAdapter(extractor)
	defer adapter.Close()

	for i := 0; i < 2; i++ {
		embedding, err := adapter.ExtractEmbedding(context.Background(), []float32{0.1, 0.2}, 16000)
		if err != nil {
			t.Fatalf("extract %d failed: %v", i, err)
		}
		if embedding.Dimension() != 3 {
			t.Fatalf("unexpected embedding dimension: %d", embedding.Dimension())
		}
	}

	if extractor.createCount != 2 {
		t.Fatalf("expected fresh stream per extraction, got %d streams", extractor.createCount)
	}
	for i, stream := range extractor.streams {
		if stream.acceptCount != 1 {
			t.Fatalf("stream %d expected one accept, got %d", i, stream.acceptCount)
		}
		if stream.inputFinishedCount != 1 {
			t.Fatalf("stream %d expected one finish, got %d", i, stream.inputFinishedCount)
		}
		if stream.deleteCount != 1 {
			t.Fatalf("stream %d expected delete after extraction, got %d", i, stream.deleteCount)
		}
	}
}

func TestSherpaEmbeddingExtractorAdapterCloseIdempotent(t *testing.T) {
	extractor := &fakeSpeakerEmbeddingExtractor{ready: true, embedding: []float32{0.1}}
	adapter := NewSherpaEmbeddingExtractorAdapter(extractor)

	if err := adapter.Close(); err != nil {
		t.Fatalf("first close failed: %v", err)
	}
	if err := adapter.Close(); err != nil {
		t.Fatalf("second close failed: %v", err)
	}
	if extractor.deleteCount != 1 {
		t.Fatalf("expected one extractor delete, got %d", extractor.deleteCount)
	}
	if adapter.IsReady() {
		t.Fatalf("closed adapter should not be ready")
	}
}

func TestSherpaEmbeddingExtractorAdapterRejectsNilContext(t *testing.T) {
	adapter := NewSherpaEmbeddingExtractorAdapter(&fakeSpeakerEmbeddingExtractor{})
	defer adapter.Close()

	//lint:ignore SA1012 deliberately nil to assert the nil-context check
	if _, err := adapter.ExtractEmbedding(nil, []float32{0.1}, 16000); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestStreamPoolDeletesInsteadOfReusingFinishedStreams(t *testing.T) {
	extractor := &fakeSpeakerEmbeddingExtractor{}
	pool := NewStreamPool(extractor)

	first := pool.Get()
	pool.Put(first)
	second := pool.Get()
	pool.Put(second)

	if first == second {
		t.Fatalf("finished speaker streams must not be reused")
	}
	if extractor.createCount != 2 {
		t.Fatalf("expected two stream creations, got %d", extractor.createCount)
	}
	if extractor.streams[0].deleteCount != 1 || extractor.streams[1].deleteCount != 1 {
		t.Fatalf("expected each stream to be deleted exactly once")
	}
}

type fakeSpeakerEmbeddingExtractor struct {
	ready       bool
	embedding   []float32
	createCount int
	deleteCount int
	streams     []*fakeSpeakerStream
}

func (e *fakeSpeakerEmbeddingExtractor) CreateStream() SpeakerStream {
	e.createCount++
	stream := &fakeSpeakerStream{}
	e.streams = append(e.streams, stream)
	return stream
}

func (e *fakeSpeakerEmbeddingExtractor) IsReady(stream SpeakerStream) bool {
	return e.ready
}

func (e *fakeSpeakerEmbeddingExtractor) Compute(stream SpeakerStream) []float32 {
	return append([]float32(nil), e.embedding...)
}

func (e *fakeSpeakerEmbeddingExtractor) Dim() int {
	return len(e.embedding)
}

func (e *fakeSpeakerEmbeddingExtractor) Delete() {
	e.deleteCount++
}

type fakeSpeakerStream struct {
	acceptCount        int
	inputFinishedCount int
	deleteCount        int
}

func (s *fakeSpeakerStream) AcceptWaveform(sampleRate int, samples []float32) {
	s.acceptCount++
}

func (s *fakeSpeakerStream) InputFinished() {
	s.inputFinishedCount++
}

func (s *fakeSpeakerStream) Delete() {
	s.deleteCount++
}
