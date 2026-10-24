package speaker

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// BatchIdentifySpeakers performs batch speaker identification for improved GPU utilization
func (m *Manager) BatchIdentifySpeakers(audioSegments [][]float32, sampleRate int) ([]*IdentifyResult, error) {
	atomic.AddInt64(&m.identifyRequests, int64(len(audioSegments)))

	if len(audioSegments) == 0 {
		return []*IdentifyResult{}, nil
	}

	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Extract embeddings in batch (if supported by the underlying library)
	embeddings, err := m.extractEmbeddingsBatch(audioSegments, sampleRate)
	if err != nil {
		atomic.AddInt64(&m.errorCount, int64(len(audioSegments)))
		return nil, fmt.Errorf("failed to extract embeddings batch: %v", err)
	}

	results := make([]*IdentifyResult, len(audioSegments))

	for i, embedding := range embeddings {
		result := &IdentifyResult{
			Identified:  false,
			SpeakerID:   "",
			SpeakerName: "",
			Confidence:  0.0,
			Threshold:   m.threshold,
		}

		// Search for best match
		speakerID := m.manager.Search(embedding, m.threshold)

		if speakerID != "" {
			// Found matching speaker
			speakerData, err := m.database.GetSpeaker(speakerID)
			if err == nil {
				result.Identified = true
				result.SpeakerID = speakerID
				result.SpeakerName = speakerData.Name
				atomic.AddInt64(&m.identifySuccesses, 1)

				// Calculate precise similarity score
				confidence := m.calculateSimilarity(embedding, speakerData.Embeddings)
				result.Confidence = confidence
			}
		}

		results[i] = result
	}

	return results, nil
}

// BatchVerifySpeakers performs batch speaker verification
func (m *Manager) BatchVerifySpeakers(speakerIDs []string, audioSegments [][]float32, sampleRate int) ([]*VerifyResult, error) {
	atomic.AddInt64(&m.verifyRequests, int64(len(audioSegments)))

	if len(audioSegments) == 0 || len(speakerIDs) != len(audioSegments) {
		return nil, fmt.Errorf("mismatched speaker IDs and audio segments")
	}

	m.mutex.RLock()
	defer m.mutex.RUnlock()

	// Extract embeddings in batch
	embeddings, err := m.extractEmbeddingsBatch(audioSegments, sampleRate)
	if err != nil {
		atomic.AddInt64(&m.errorCount, int64(len(audioSegments)))
		return nil, fmt.Errorf("failed to extract embeddings batch: %v", err)
	}

	results := make([]*VerifyResult, len(audioSegments))

	for i, embedding := range embeddings {
		speakerID := speakerIDs[i]

		// Check if speaker exists
		speakerData, err := m.database.GetSpeaker(speakerID)
		if err != nil {
			atomic.AddInt64(&m.errorCount, 1)
			results[i] = &VerifyResult{
				SpeakerID:   speakerID,
				SpeakerName: "",
				Verified:    false,
				Confidence:  0.0,
				Threshold:   m.threshold,
			}
			continue
		}

		// Calculate similarity score
		confidence := m.calculateSimilarity(embedding, speakerData.Embeddings)
		verified := confidence >= m.threshold

		if verified {
			atomic.AddInt64(&m.verifySuccesses, 1)
		}

		results[i] = &VerifyResult{
			SpeakerID:   speakerID,
			SpeakerName: speakerData.Name,
			Verified:    verified,
			Confidence:  confidence,
			Threshold:   m.threshold,
		}
	}

	return results, nil
}

// extractEmbeddingsBatch extracts embeddings for multiple audio segments using concurrent processing
func (m *Manager) extractEmbeddingsBatch(audioSegments [][]float32, sampleRate int) ([][]float32, error) {
	if len(audioSegments) == 0 {
		return [][]float32{}, nil
	}

	// For small batches, process sequentially to avoid goroutine overhead
	if len(audioSegments) <= 4 {
		return m.extractEmbeddingsSequential(audioSegments, sampleRate)
	}

	// For larger batches, use concurrent processing
	return m.extractEmbeddingsConcurrent(audioSegments, sampleRate)
}

// extractEmbeddingsSequential processes embeddings sequentially for small batches
func (m *Manager) extractEmbeddingsSequential(audioSegments [][]float32, sampleRate int) ([][]float32, error) {
	embeddings := make([][]float32, len(audioSegments))

	for i, audioData := range audioSegments {
		embedding, err := m.extractEmbedding(context.Background(), audioData, sampleRate)
		if err != nil {
			return nil, fmt.Errorf("failed to extract embedding for segment %d: %v", i, err)
		}
		embeddings[i] = embedding
	}

	return embeddings, nil
}

// extractEmbeddingsConcurrent processes embeddings concurrently for better performance
func (m *Manager) extractEmbeddingsConcurrent(audioSegments [][]float32, sampleRate int) ([][]float32, error) {
	numSegments := len(audioSegments)
	embeddings := make([][]float32, numSegments)

	// Use a reasonable number of goroutines (up to 8 to avoid overwhelming the system)
	maxGoroutines := min(8, numSegments)
	semaphore := make(chan struct{}, maxGoroutines)

	var wg sync.WaitGroup
	errChan := make(chan error, numSegments)

	// Process segments concurrently. The semaphore is acquired before the
	// spawn, so at most maxGoroutines goroutines exist at once; spawning
	// first and acquiring inside would park one goroutine per segment for a
	// large batch before any of them runs.
	for i, audioData := range audioSegments {
		semaphore <- struct{}{}
		wg.Add(1)
		go func(index int, data []float32) {
			defer wg.Done()
			defer func() { <-semaphore }()

			embedding, err := m.extractEmbedding(context.Background(), data, sampleRate)
			if err != nil {
				errChan <- fmt.Errorf("failed to extract embedding for segment %d: %v", index, err)
				return
			}

			embeddings[index] = embedding
		}(i, audioData)
	}

	// Wait for all goroutines to complete
	wg.Wait()
	close(errChan)

	// Check for errors
	if err := <-errChan; err != nil {
		return nil, err
	}

	return embeddings, nil
}
