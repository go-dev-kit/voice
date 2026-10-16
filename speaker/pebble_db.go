package speaker

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/cockroachdb/pebble"
)

// PebbleSpeakerDatabase implements SpeakerDatabase interface using Pebble
type PebbleSpeakerDatabase struct {
	db     *pebble.DB
	dbPath string
	mu     sync.RWMutex
}

// NewPebbleSpeakerDatabase creates a new Pebble-based speaker database
func NewPebbleSpeakerDatabase(dataDir string) (*PebbleSpeakerDatabase, error) {
	dbPath := filepath.Join(dataDir, "pebble_speakers")

	// Ensure directory exists
	if err := os.MkdirAll(dbPath, 0o750); err != nil {
		return nil, fmt.Errorf("failed to create database directory: %v", err)
	}

	// Open Pebble database with optimized settings for speaker data
	cache := pebble.NewCache(64 << 20) // 64MB cache
	defer cache.Unref()

	opts := &pebble.Options{
		Cache: cache,
	}

	// Configure level options for better performance
	for i := 0; i < len(opts.Levels); i++ {
		l := &opts.Levels[i]
		l.BlockSize = 64 << 10       // 64KB blocks
		l.IndexBlockSize = 256 << 10 // 256KB index blocks
		l.TargetFileSize = 2 << 30   // 2GB target file size
		l.Compression = pebble.ZstdCompression
	}

	db, err := pebble.Open(dbPath, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to open Pebble database: %v", err)
	}

	database := &PebbleSpeakerDatabase{
		db:     db,
		dbPath: dbPath,
	}

	// Migrate from JSON if it exists
	if err := database.migrateFromJSON(dataDir); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to migrate from JSON: %v", err)
	}

	return database, nil
}

// Close closes the database
func (p *PebbleSpeakerDatabase) Close() error {
	return p.db.Close()
}

// migrateFromJSON migrates data from JSON file to Pebble database
func (p *PebbleSpeakerDatabase) migrateFromJSON(dataDir string) error {
	jsonPath := filepath.Join(dataDir, "speaker.json")

	// Check if JSON file exists
	if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
		return nil // No migration needed
	}

	// Read JSON file
	data, err := os.ReadFile(filepath.Clean(jsonPath))
	if err != nil {
		return fmt.Errorf("failed to read JSON file: %v", err)
	}

	var jsonDB SpeakerDatabaseData
	if err := json.Unmarshal(data, &jsonDB); err != nil {
		return fmt.Errorf("failed to unmarshal JSON: %v", err)
	}

	// Migrate speakers to Pebble
	batch := p.db.NewBatch()
	defer batch.Close()

	for speakerID, speakerData := range jsonDB.Speakers {
		key := []byte("speaker:" + speakerID)
		value, err := json.Marshal(speakerData)
		if err != nil {
			return fmt.Errorf("failed to marshal speaker %s: %v", speakerID, err)
		}

		if err := batch.Set(key, value, pebble.Sync); err != nil {
			return fmt.Errorf("failed to set speaker %s: %v", speakerID, err)
		}
	}

	// Store metadata
	metadata := map[string]any{
		"version":       jsonDB.Version,
		"updated_at":    jsonDB.UpdatedAt,
		"migrated_at":   time.Now(),
		"migrated_from": "json",
	}

	metadataKey := []byte("metadata")
	metadataValue, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	if err := batch.Set(metadataKey, metadataValue, pebble.Sync); err != nil {
		return fmt.Errorf("failed to set metadata: %v", err)
	}

	// Commit batch
	if err := batch.Commit(pebble.Sync); err != nil {
		return fmt.Errorf("failed to commit migration batch: %v", err)
	}

	// Backup and remove old JSON file
	backupPath := jsonPath + ".backup"
	if err := os.Rename(jsonPath, backupPath); err != nil {
		return fmt.Errorf("failed to backup JSON file: %v", err)
	}

	return nil
}

// RegisterSpeaker registers a new speaker
func (p *PebbleSpeakerDatabase) RegisterSpeaker(speakerID, speakerName string, embeddings [][]float32) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := time.Now()
	speakerData := &SpeakerData{
		ID:          speakerID,
		Name:        speakerName,
		Embeddings:  embeddings,
		CreatedAt:   now,
		UpdatedAt:   now,
		LastUsedAt:  now,
		SampleCount: len(embeddings),
	}

	key := []byte("speaker:" + speakerID)
	value, err := json.Marshal(speakerData)
	if err != nil {
		return fmt.Errorf("failed to marshal speaker data: %v", err)
	}

	return p.db.Set(key, value, pebble.Sync)
}

// GetSpeaker retrieves speaker data by ID
func (p *PebbleSpeakerDatabase) GetSpeaker(speakerID string) (*SpeakerData, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	key := []byte("speaker:" + speakerID)
	value, closer, err := p.db.Get(key)
	if err != nil {
		if err == pebble.ErrNotFound {
			return nil, fmt.Errorf("speaker %s not found", speakerID)
		}
		return nil, fmt.Errorf("failed to get speaker: %v", err)
	}
	defer closer.Close()

	var speakerData SpeakerData
	if err := json.Unmarshal(value, &speakerData); err != nil {
		return nil, fmt.Errorf("failed to unmarshal speaker data: %v", err)
	}

	return &speakerData, nil
}

// UpdateSpeaker updates speaker data
func (p *PebbleSpeakerDatabase) UpdateSpeaker(speakerID string, speakerData *SpeakerData) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	speakerData.UpdatedAt = time.Now()

	key := []byte("speaker:" + speakerID)
	value, err := json.Marshal(speakerData)
	if err != nil {
		return fmt.Errorf("failed to marshal speaker data: %v", err)
	}

	return p.db.Set(key, value, pebble.Sync)
}

// DeleteSpeaker deletes a speaker
func (p *PebbleSpeakerDatabase) DeleteSpeaker(speakerID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	key := []byte("speaker:" + speakerID)
	return p.db.Delete(key, pebble.Sync)
}

// ListSpeakers returns all speaker IDs
func (p *PebbleSpeakerDatabase) ListSpeakers() ([]string, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var speakers []string

	prefix := []byte("speaker:")
	iter, err := p.db.NewIter(&pebble.IterOptions{
		LowerBound: prefix,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create iterator: %v", err)
	}
	defer iter.Close()

	for iter.First(); iter.Valid(); iter.Next() {
		key := iter.Key()
		if len(key) > len(prefix) {
			speakerID := string(key[len(prefix):])
			speakers = append(speakers, speakerID)
		}
	}

	if err := iter.Error(); err != nil {
		return nil, fmt.Errorf("iterator error: %v", err)
	}

	return speakers, nil
}

// GetAllSpeakers returns all speakers with their embeddings
func (p *PebbleSpeakerDatabase) GetAllSpeakers() (map[string][]float32, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	speakers := make(map[string][]float32)

	speakerIDs, err := p.ListSpeakers()
	if err != nil {
		return nil, err
	}

	for _, speakerID := range speakerIDs {
		speakerData, err := p.GetSpeaker(speakerID)
		if err != nil {
			continue // Skip speakers that can't be loaded
		}

		if len(speakerData.Embeddings) > 0 {
			// Use the first embedding as representative
			speakers[speakerID] = speakerData.Embeddings[0]
		}
	}

	return speakers, nil
}

// GetSpeakerEmbedding gets a single embedding for a speaker
func (p *PebbleSpeakerDatabase) GetSpeakerEmbedding(speakerID string) ([]float32, error) {
	speakerData, err := p.GetSpeaker(speakerID)
	if err != nil {
		return nil, err
	}

	if len(speakerData.Embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings for speaker %s", speakerID)
	}

	// Return first embedding (could implement averaging or selection logic)
	return speakerData.Embeddings[0], nil
}

// BatchRegisterSpeakers registers multiple speakers in a batch
func (p *PebbleSpeakerDatabase) BatchRegisterSpeakers(speakers map[string]*SpeakerData) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	batch := p.db.NewBatch()
	defer batch.Close()

	for speakerID, speakerData := range speakers {
		key := []byte("speaker:" + speakerID)
		value, err := json.Marshal(speakerData)
		if err != nil {
			return fmt.Errorf("failed to marshal speaker %s: %v", speakerID, err)
		}

		if err := batch.Set(key, value, nil); err != nil {
			return fmt.Errorf("failed to batch set speaker %s: %v", speakerID, err)
		}
	}

	return batch.Commit(pebble.Sync)
}

// GetStats returns database statistics
func (p *PebbleSpeakerDatabase) GetStats() (*DatabaseStats, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	speakerIDs, err := p.ListSpeakers()
	if err != nil {
		return nil, err
	}

	totalSamples := 0
	for _, speakerID := range speakerIDs {
		speakerData, err := p.GetSpeaker(speakerID)
		if err != nil {
			continue
		}
		totalSamples += speakerData.SampleCount
	}

	// Get metadata
	metadataKey := []byte("metadata")
	metadataValue, closer, err := p.db.Get(metadataKey)
	version := "1.0.0"
	var updatedAt time.Time

	if err == nil {
		defer closer.Close()
		var metadata map[string]any
		if err := json.Unmarshal(metadataValue, &metadata); err == nil {
			if v, ok := metadata["version"].(string); ok {
				version = v
			}
			if t, ok := metadata["updated_at"].(string); ok {
				if parsed, err := time.Parse(time.RFC3339, t); err == nil {
					updatedAt = parsed
				}
			}
		}
	} else if err != pebble.ErrNotFound {
		return nil, fmt.Errorf("failed to get metadata: %v", err)
	}

	return &DatabaseStats{
		TotalSpeakers: len(speakerIDs),
		TotalSamples:  totalSamples,
		EmbeddingDim:  192, // Default embedding dimension
		Threshold:     0.5, // Default threshold
		Version:       version,
		UpdatedAt:     updatedAt,
	}, nil
}

// Compact performs database compaction for better performance
func (p *PebbleSpeakerDatabase) Compact() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Compact the entire database
	return p.db.Compact([]byte("metadata"), []byte("speaker:~"), false)
}
