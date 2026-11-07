package speaker

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"time"
)

// ShardedSpeakerDatabase implements SpeakerDatabase with multiple shards for reduced lock contention
type ShardedSpeakerDatabase struct {
	shards       []*PebbleSpeakerDatabase
	shardCount   int
	shardMutexes []sync.RWMutex
}

// NewShardedSpeakerDatabase creates a new sharded speaker database
func NewShardedSpeakerDatabase(dataDir string, shardCount int) (*ShardedSpeakerDatabase, error) {
	if shardCount <= 0 {
		shardCount = 8 // Default to 8 shards
	}

	// Limit shard count to prevent excessive resource usage
	if shardCount > 64 {
		shardCount = 64
	}

	shards := make([]*PebbleSpeakerDatabase, shardCount)
	shardMutexes := make([]sync.RWMutex, shardCount)

	for i := 0; i < shardCount; i++ {
		shardDataDir := fmt.Sprintf("%s/shard_%02d", dataDir, i)
		shard, err := NewPebbleSpeakerDatabase(shardDataDir)
		if err != nil {
			// Close already created shards on error
			var closeErr error
			for j := 0; j < i; j++ {
				if err := shards[j].Close(); err != nil {
					closeErr = errors.Join(closeErr, err)
				}
			}
			if closeErr != nil {
				return nil, fmt.Errorf("failed to create shard %d: %v; failed to close initialized shards: %w", i, err, closeErr)
			}
			return nil, fmt.Errorf("failed to create shard %d: %v", i, err)
		}
		shards[i] = shard
	}

	return &ShardedSpeakerDatabase{
		shards:       shards,
		shardCount:   shardCount,
		shardMutexes: shardMutexes,
	}, nil
}

// Close closes all shards
func (s *ShardedSpeakerDatabase) Close() error {
	var lastErr error
	for _, shard := range s.shards {
		if err := shard.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

// getShard returns the shard index for a speaker ID
func (s *ShardedSpeakerDatabase) getShard(speakerID string) int {
	// Use SHA256 hash of speaker ID for even distribution
	hash := sha256.Sum256([]byte(speakerID))
	hashValue := binary.BigEndian.Uint64(hash[:8])
	// NewShardedSpeakerDatabase bounds shardCount to [1, 64], so the modulo
	// result is always representable as int on every supported architecture.
	return int(hashValue % uint64(s.shardCount)) // #nosec G115 -- bounded as described above.
}

// getShardForRead gets the shard for reading operations
func (s *ShardedSpeakerDatabase) getShardForRead(speakerID string) (*PebbleSpeakerDatabase, *sync.RWMutex) {
	shardIndex := s.getShard(speakerID)
	return s.shards[shardIndex], &s.shardMutexes[shardIndex]
}

// getShardForWrite gets the shard for writing operations
func (s *ShardedSpeakerDatabase) getShardForWrite(speakerID string) (*PebbleSpeakerDatabase, *sync.RWMutex) {
	shardIndex := s.getShard(speakerID)
	return s.shards[shardIndex], &s.shardMutexes[shardIndex]
}

// RegisterSpeaker registers a new speaker in the appropriate shard
func (s *ShardedSpeakerDatabase) RegisterSpeaker(speakerID, speakerName string, embeddings [][]float32) error {
	shard, mu := s.getShardForWrite(speakerID)
	mu.Lock()
	defer mu.Unlock()

	return shard.RegisterSpeaker(speakerID, speakerName, embeddings)
}

// GetSpeaker retrieves speaker data from the appropriate shard
func (s *ShardedSpeakerDatabase) GetSpeaker(speakerID string) (*SpeakerData, error) {
	shard, mu := s.getShardForRead(speakerID)
	mu.RLock()
	defer mu.RUnlock()

	return shard.GetSpeaker(speakerID)
}

// UpdateSpeaker updates speaker data in the appropriate shard
func (s *ShardedSpeakerDatabase) UpdateSpeaker(speakerID string, speakerData *SpeakerData) error {
	shard, mu := s.getShardForWrite(speakerID)
	mu.Lock()
	defer mu.Unlock()

	return shard.UpdateSpeaker(speakerID, speakerData)
}

// DeleteSpeaker deletes a speaker from the appropriate shard
func (s *ShardedSpeakerDatabase) DeleteSpeaker(speakerID string) error {
	shard, mu := s.getShardForWrite(speakerID)
	mu.Lock()
	defer mu.Unlock()

	return shard.DeleteSpeaker(speakerID)
}

// ListSpeakers returns all speaker IDs from all shards
func (s *ShardedSpeakerDatabase) ListSpeakers() ([]string, error) {
	var allSpeakers []string

	// Collect speakers from all shards concurrently
	type result struct {
		speakers []string
		err      error
	}

	results := make(chan result, s.shardCount)

	for i := 0; i < s.shardCount; i++ {
		go func(shardIndex int) {
			shard := s.shards[shardIndex]
			mu := &s.shardMutexes[shardIndex]

			mu.RLock()
			speakers, err := shard.ListSpeakers()
			mu.RUnlock()

			results <- result{speakers: speakers, err: err}
		}(i)
	}

	// Collect results
	for i := 0; i < s.shardCount; i++ {
		res := <-results
		if res.err != nil {
			return nil, res.err
		}
		allSpeakers = append(allSpeakers, res.speakers...)
	}

	return allSpeakers, nil
}

// GetAllSpeakers returns all speakers with their embeddings from all shards
func (s *ShardedSpeakerDatabase) GetAllSpeakers() (map[string][]float32, error) {
	allSpeakers := make(map[string][]float32)

	// Collect speakers from all shards concurrently
	type result struct {
		speakers map[string][]float32
		err      error
	}

	results := make(chan result, s.shardCount)

	for i := 0; i < s.shardCount; i++ {
		go func(shardIndex int) {
			shard := s.shards[shardIndex]
			mu := &s.shardMutexes[shardIndex]

			mu.RLock()
			speakers, err := shard.GetAllSpeakers()
			mu.RUnlock()

			results <- result{speakers: speakers, err: err}
		}(i)
	}

	// Collect results and merge
	for i := 0; i < s.shardCount; i++ {
		res := <-results
		if res.err != nil {
			return nil, res.err
		}
		for speakerID, embedding := range res.speakers {
			allSpeakers[speakerID] = embedding
		}
	}

	return allSpeakers, nil
}

// GetSpeakerEmbedding gets a single embedding for a speaker from the appropriate shard
func (s *ShardedSpeakerDatabase) GetSpeakerEmbedding(speakerID string) ([]float32, error) {
	shard, mu := s.getShardForRead(speakerID)
	mu.RLock()
	defer mu.RUnlock()

	return shard.GetSpeakerEmbedding(speakerID)
}

// BatchRegisterSpeakers registers multiple speakers across shards
func (s *ShardedSpeakerDatabase) BatchRegisterSpeakers(speakers map[string]*SpeakerData) error {
	// Group speakers by shard
	shardGroups := make([]map[string]*SpeakerData, s.shardCount)
	for i := 0; i < s.shardCount; i++ {
		shardGroups[i] = make(map[string]*SpeakerData)
	}

	for speakerID, speakerData := range speakers {
		shardIndex := s.getShard(speakerID)
		shardGroups[shardIndex][speakerID] = speakerData
	}

	// Register speakers in each shard concurrently
	type batchResult struct {
		shardIndex int
		err        error
	}

	results := make(chan batchResult, s.shardCount)

	for shardIndex, shardSpeakers := range shardGroups {
		if len(shardSpeakers) == 0 {
			results <- batchResult{shardIndex: shardIndex, err: nil}
			continue
		}

		go func(idx int, speakers map[string]*SpeakerData) {
			shard := s.shards[idx]
			mu := &s.shardMutexes[idx]

			mu.Lock()
			err := shard.BatchRegisterSpeakers(speakers)
			mu.Unlock()

			results <- batchResult{shardIndex: idx, err: err}
		}(shardIndex, shardSpeakers)
	}

	// Collect results
	for i := 0; i < s.shardCount; i++ {
		res := <-results
		if res.err != nil {
			return fmt.Errorf("batch register failed for shard %d: %v", res.shardIndex, res.err)
		}
	}

	return nil
}

// GetStats returns aggregated statistics from all shards
func (s *ShardedSpeakerDatabase) GetStats() (*DatabaseStats, error) {
	totalSpeakers := 0
	totalSamples := 0
	maxUpdatedAt := time.Time{}

	// Collect stats from all shards
	for i := 0; i < s.shardCount; i++ {
		shard := s.shards[i]
		mu := &s.shardMutexes[i]

		mu.RLock()
		stats, err := shard.GetStats()
		mu.RUnlock()

		if err != nil {
			return nil, fmt.Errorf("failed to get stats for shard %d: %v", i, err)
		}

		totalSpeakers += stats.TotalSpeakers
		totalSamples += stats.TotalSamples

		if stats.UpdatedAt.After(maxUpdatedAt) {
			maxUpdatedAt = stats.UpdatedAt
		}
	}

	return &DatabaseStats{
		TotalSpeakers: totalSpeakers,
		TotalSamples:  totalSamples,
		EmbeddingDim:  192, // Default embedding dimension
		Threshold:     0.5, // Default threshold
		Version:       "2.0.0-sharded",
		UpdatedAt:     maxUpdatedAt,
	}, nil
}

// Compact performs compaction on all shards
func (s *ShardedSpeakerDatabase) Compact() error {
	// Compact shards concurrently
	type compactResult struct {
		shardIndex int
		err        error
	}

	results := make(chan compactResult, s.shardCount)

	for i := 0; i < s.shardCount; i++ {
		go func(shardIndex int) {
			shard := s.shards[shardIndex]
			mu := &s.shardMutexes[shardIndex]

			mu.Lock()
			err := shard.Compact()
			mu.Unlock()

			results <- compactResult{shardIndex: shardIndex, err: err}
		}(i)
	}

	// Collect results
	for i := 0; i < s.shardCount; i++ {
		res := <-results
		if res.err != nil {
			return fmt.Errorf("compaction failed for shard %d: %v", res.shardIndex, res.err)
		}
	}

	return nil
}

// GetShardCount returns the number of shards
func (s *ShardedSpeakerDatabase) GetShardCount() int {
	return s.shardCount
}

// GetShardDistribution returns the distribution of speakers across shards
func (s *ShardedSpeakerDatabase) GetShardDistribution() (map[int]int, error) {
	distribution := make(map[int]int)

	for i := 0; i < s.shardCount; i++ {
		shard := s.shards[i]
		mu := &s.shardMutexes[i]

		mu.RLock()
		speakers, err := shard.ListSpeakers()
		mu.RUnlock()

		if err != nil {
			return nil, fmt.Errorf("failed to get speaker count for shard %d: %v", i, err)
		}

		distribution[i] = len(speakers)
	}

	return distribution, nil
}
