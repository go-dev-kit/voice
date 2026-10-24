package indexing

import (
	"context"
	"fmt"
	"sync"

	"github.com/viktordanov/go-hnswlib"
)

// HNSWIndex implements VectorIndex using Hierarchical Navigable Small World
type HNSWIndex struct {
	index  *hnswlib.HNSW
	mapper *IDMapper
	config *Config
	mu     sync.RWMutex
}

// NewHNSWIndex creates a new HNSW-based vector index
func NewHNSWIndex(config *Config) (*HNSWIndex, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	// Create ID mapper
	mapper := NewIDMapper()

	// Determine HNSW distance type based on our DistanceMetric
	var distanceType byte
	switch config.DistanceMetric {
	case Cosine:
		distanceType = 2 // Cosine
	case L2:
		distanceType = 0 // L2
	case InnerProduct:
		distanceType = 1 // Inner Product
	default:
		return nil, fmt.Errorf("unsupported distance metric: %s", config.DistanceMetric)
	}

	// Create HNSW index
	// Parameters: dim, max_elements, M, ef_construction, random_seed, distance_type
	index := hnswlib.InitHNSW(
		int32(config.Dimension),
		uint64(config.MaxElements),
		int32(config.M),
		int32(config.EfConstruction),
		42, // Fixed seed for reproducibility
		distanceType,
	)

	return &HNSWIndex{
		index:  index,
		mapper: mapper,
		config: config,
	}, nil
}

// Add adds a vector with an ID to the index
func (h *HNSWIndex) Add(ctx context.Context, id string, vector []float32) error {
	// Check for context cancellation
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.index == nil {
		return fmt.Errorf("index is closed")
	}

	// Validate vector dimensions
	if len(vector) != h.config.Dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d",
			h.config.Dimension, len(vector))
	}

	// Get integer ID for the string ID
	intID := h.mapper.GetIntID(id)

	// Add to HNSW index. The Safe variant reports failure (bad handle, or
	// the C++ addPoint threw — e.g. the index is at MaxElements) instead of
	// silently dropping the point.
	if code := hnswlib.AddPointSafe(h.index, vector, uint64(intID)); code != 0 {
		return fmt.Errorf("hnsw: add %q failed (status %d; index may be at its max elements)", id, code)
	}

	return nil
}

// Remove removes a vector from the index by ID
func (h *HNSWIndex) Remove(ctx context.Context, id string) error {
	// Check for context cancellation
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.index == nil {
		return fmt.Errorf("index is closed")
	}

	// Get the integer ID
	intID, exists := h.mapper.LookupIntID(id)
	if !exists {
		return nil
	}

	// Mark as deleted in hnswlib
	hnswlib.MarkDeleted(h.index, uint64(intID))

	// Remove from mapper
	h.mapper.RemoveID(id)

	return nil
}

// Search finds the k most similar vectors to the query vector
func (h *HNSWIndex) Search(ctx context.Context, query []float32, k int, threshold float32) ([]SearchResult, error) {
	// Check for context cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.index == nil {
		return nil, fmt.Errorf("index is closed")
	}

	// Validate query vector dimensions
	if len(query) != h.config.Dimension {
		return nil, fmt.Errorf("query vector dimension mismatch: expected %d, got %d",
			h.config.Dimension, len(query))
	}

	liveSize := h.mapper.Size()
	if liveSize == 0 {
		return nil, nil
	}
	k = normalizeSearchLimit(k, liveSize)

	// Prepare result arrays
	labels := make([]uint64, k)
	distances := make([]float32, k)

	// Search
	hnswlib.SearchKnn(h.index, query, int32(k), labels, distances)

	// Convert results
	var results []SearchResult
	for i := 0; i < k; i++ {
		if labels[i] == 0 && distances[i] == 0 {
			// No more results
			break
		}

		// Get string ID
		stringID, exists := h.mapper.GetStringID(int(labels[i]))
		if !exists {
			continue // Skip deleted or invalid entries
		}

		similarity := similarityFromDistance(h.config.DistanceMetric, distances[i])
		if similarity >= threshold {
			results = append(results, SearchResult{
				ID:         stringID,
				Similarity: similarity,
				Distance:   distances[i],
			})
		}
	}

	return results, nil
}

func normalizeSearchLimit(k, liveSize int) int {
	if k <= 0 {
		k = 1
	}
	if k > liveSize {
		k = liveSize
	}
	if k > 100 {
		return 100
	}
	return k
}

func similarityFromDistance(metric DistanceMetric, distance float32) float32 {
	switch metric {
	case Cosine:
		return 1.0 - distance
	case L2:
		if distance < 0 {
			return 0
		}
		return 1.0 / (1.0 + distance)
	case InnerProduct:
		return clampSimilarity(distance)
	default:
		return 0
	}
}

func clampSimilarity(value float32) float32 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

// Size returns the number of vectors currently in the index
func (h *HNSWIndex) Size(ctx context.Context) (int, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if h.index == nil {
		return 0, fmt.Errorf("index is closed")
	}
	return h.mapper.Size(), nil
}

// Close releases resources used by the index
func (h *HNSWIndex) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.index != nil {
		// Free the HNSW index
		hnswlib.FreeHNSW(h.index)
		h.index = nil
	}

	// Clear the mapper
	if h.mapper != nil {
		h.mapper.Clear()
	}

	return nil
}
