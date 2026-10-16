package speaker

import (
	"context"
	"fmt"
	"os"

	"github.com/k2-fsa/sherpa-onnx-go/sherpa_onnx"
	"go.glpx.pro/gdk/voice/indexing"
	"go.glpx.pro/gdk/voice/speaker/application"
	"go.glpx.pro/gdk/voice/speaker/domain"
)

// NewManager creates speaker recognition manager
func NewManager(config *Config) (*Manager, error) {
	if config == nil {
		config = &Config{
			NumThreads: 1,
			Provider:   "cpu",
			Threshold:  0.5,
			Logger:     &NoOpLogger{},
		}
	}

	if config.Logger == nil {
		config.Logger = &NoOpLogger{}
	}

	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid speaker configuration: %w", err)
	}

	// Ensure data directory exists
	if err := os.MkdirAll(config.DataDir, 0o750); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %v", err)
	}

	// Create speaker embedding extractor config
	extractorConfig := &sherpa_onnx.SpeakerEmbeddingExtractorConfig{
		Model:      config.ModelPath,
		NumThreads: config.NumThreads,
		Debug:      0,
		Provider:   config.Provider,
	}

	// Create speaker embedding extractor
	extractor := NewSherpaExtractorAdapter(extractorConfig)
	if extractor == nil {
		return nil, fmt.Errorf("failed to create speaker embedding extractor")
	}

	// Get embedding dimension
	dim := extractor.Dim()
	config.Logger.Infof("Speaker embedding dimension: %d", dim)

	// Create speaker embedding manager
	embeddingManager := NewSherpaManagerAdapter(dim)
	if embeddingManager == nil {
		extractor.Delete()
		return nil, fmt.Errorf("failed to create speaker embedding manager")
	}

	// Initialize sharded Pebble database (8 shards by default)
	shardCount := 8
	if config.NumThreads > 1 {
		// Scale shard count with thread count, but cap at 16
		shardCount = min(16, max(4, config.NumThreads*2))
	}

	shardedDB, err := NewShardedSpeakerDatabase(config.DataDir, shardCount)
	if err != nil {
		extractor.Delete()
		embeddingManager.Delete()
		return nil, fmt.Errorf("failed to initialize sharded database: %v", err)
	}

	// Create HNSW indexer for fast similarity search
	indexConfig := &indexing.Config{
		Dimension:      dim,
		MaxElements:    10000,
		M:              16,
		EfConstruction: 200,
		EfSearch:       100,
		DistanceMetric: indexing.Cosine,
	}
	vectorIndex, err := indexing.NewHNSWIndex(indexConfig)
	if err != nil {
		extractor.Delete()
		embeddingManager.Delete()
		if closeErr := shardedDB.Close(); closeErr != nil {
			config.Logger.Errorf("failed to close speaker database after vector index initialization failure: %v", closeErr)
		}
		return nil, fmt.Errorf("failed to initialize vector index: %v", err)
	}

	// Create adapter for speaker domain
	vectorIndexAdapter := NewVectorIndexAdapter(vectorIndex)

	// Create domain service
	repositoryAdapter := NewSpeakerDatabaseAdapter(shardedDB)
	embeddingAdapter := NewSherpaEmbeddingExtractorAdapter(extractor)
	similarityCalculator := NewCosineSimilarityCalculator()

	thresholdScore, _ := domain.NewSimilarityScore(config.Threshold)

	speakerService := domain.NewSpeakerService(
		repositoryAdapter,
		embeddingAdapter,
		vectorIndexAdapter,
		similarityCalculator,
		thresholdScore,
	)

	// Create application layer use cases
	metricsRecorder := application.NewSimpleMetricsRecorder()
	recognitionUseCase := application.NewSpeakerRecognitionUseCase(
		*speakerService,
		config.Logger,
		metricsRecorder,
	)
	managementUseCase := application.NewSpeakerManagementUseCase(
		*speakerService,
		config.Logger,
		metricsRecorder,
	)

	retentionPolicy := config.RetentionPolicy
	if retentionPolicy == "" {
		retentionPolicy = RetentionFIFO
	}

	manager := &Manager{
		// Clean architecture components
		recognitionUseCase: recognitionUseCase,
		managementUseCase:  managementUseCase,

		// Legacy infrastructure (for backward compatibility)
		extractor:   extractor,
		manager:     embeddingManager,
		database:    shardedDB,
		vectorIndex: vectorIndexAdapter,

		// Configuration
		threshold:       config.Threshold,
		embeddingDim:    dim,
		dataDir:         config.DataDir,
		logger:          config.Logger,
		maxSpeakers:     config.MaxSpeakers,
		retentionPolicy: retentionPolicy,
		maxAge:          config.MaxAge,
	}

	// Load speakers into memory manager
	if err := manager.loadSpeakersToMemory(); err != nil {
		config.Logger.Infof("Warning: failed to load speakers to memory: %v", err)
	}

	return manager, nil
}

// Close closes manager and releases resources
func (m *Manager) Close() {
	if m.extractor != nil {
		m.extractor.Delete()
	}
	if m.manager != nil {
		m.manager.Delete()
	}
	if m.database != nil {
		if err := m.database.Close(); err != nil && m.logger != nil {
			m.logger.Errorf("failed to close speaker database: %v", err)
		}
	}
	if m.vectorIndex != nil {
		if err := m.vectorIndex.Close(); err != nil && m.logger != nil {
			m.logger.Errorf("failed to close speaker vector index: %v", err)
		}
	}
}

// loadSpeakersToMemory loads speakers from database into memory manager and vector index
func (m *Manager) loadSpeakersToMemory() error {
	speakerList, err := m.database.ListSpeakers()
	if err != nil {
		return fmt.Errorf("failed to list speakers from database: %v", err)
	}

	loadedCount := 0
	totalEmbeddings := 0

	// Load into both Sherpa memory manager and vector index
	for _, speakerID := range speakerList {
		speakerData, err := m.database.GetSpeaker(speakerID)
		if err != nil {
			m.logger.Errorf("Failed to get speaker %s: %v", speakerID, err)
			continue
		}

		if len(speakerData.Embeddings) == 0 {
			continue
		}

		// Register to Sherpa memory manager (for backward compatibility)
		success := m.manager.RegisterV(speakerID, speakerData.Embeddings)
		if !success {
			m.logger.Infof("Warning: failed to register speaker %s to Sherpa memory", speakerID)
		}

		// Create domain speaker for indexing
		domainID, _ := domain.NewSpeakerID(speakerID)
		domainName, _ := domain.NewSpeakerName(speakerData.Name)

		// Convert embeddings
		embeddings := make([]domain.SpeakerEmbedding, len(speakerData.Embeddings))
		for i, emb := range speakerData.Embeddings {
			embedding, _ := domain.NewSpeakerEmbedding(emb)
			embeddings[i] = embedding
		}

		// Create domain speaker
		speaker := domain.NewSpeaker(domainID, domainName, embeddings[0])
		for i := 1; i < len(embeddings); i++ {
			speaker.AddEmbedding(embeddings[i])
		}

		// Add to vector index
		err = m.vectorIndex.AddSpeaker(context.Background(), speaker)
		if err != nil {
			m.logger.Errorf("Warning: failed to add speaker %s to vector index: %v", speakerID, err)
		} else {
			loadedCount++
			totalEmbeddings += len(speakerData.Embeddings)
		}
	}

	m.logger.Infof("✅ Loaded %d speakers with %d total embeddings to memory and vector index",
		loadedCount, totalEmbeddings)
	return nil
}
