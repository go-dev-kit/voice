package indexing

import (
	"fmt"
	"math"
)

// Config holds configuration for vector indexing
type Config struct {
	// Dimension is the dimensionality of vectors (e.g., 192, 512)
	Dimension int

	// MaxElements is the maximum number of vectors that can be indexed
	MaxElements int

	// M is the number of bidirectional links created for every new element
	// Higher values increase accuracy but use more memory (16-64 typical)
	M int

	// EfConstruction is the size of the dynamic candidate list during construction
	// Higher values improve index quality but increase build time (200-800 typical)
	EfConstruction int

	// EfSearch is the size of the dynamic candidate list during search
	// Higher values improve recall but slow down queries (50-400 typical)
	EfSearch int

	// DistanceMetric specifies the distance/similarity metric to use
	DistanceMetric DistanceMetric
}

// DefaultConfig returns a default configuration for typical embedding use cases
func DefaultConfig() *Config {
	return &Config{
		Dimension:      192,   // Common for many speaker models
		MaxElements:    10000, // Start with reasonable capacity
		M:              16,    // Good balance between memory and accuracy
		EfConstruction: 200,   // Higher for better index quality
		EfSearch:       100,   // Good for 95%+ recall
		DistanceMetric: Cosine,
	}
}

// Validate validates the configuration parameters
func (c *Config) Validate() error {
	if c.Dimension <= 0 {
		return fmt.Errorf("dimension must be positive, got %d", c.Dimension)
	}
	if c.Dimension > math.MaxInt32 {
		return fmt.Errorf("dimension must not exceed %d, got %d", math.MaxInt32, c.Dimension)
	}
	if c.MaxElements <= 0 {
		return fmt.Errorf("maxElements must be positive, got %d", c.MaxElements)
	}
	if c.M <= 0 {
		return fmt.Errorf("m must be positive, got %d", c.M)
	}
	if c.M > math.MaxInt32 {
		return fmt.Errorf("m must not exceed %d, got %d", math.MaxInt32, c.M)
	}
	if c.EfConstruction < c.M {
		return fmt.Errorf("efConstruction must be >= M, got %d < %d", c.EfConstruction, c.M)
	}
	if c.EfConstruction > math.MaxInt32 {
		return fmt.Errorf("efConstruction must not exceed %d, got %d", math.MaxInt32, c.EfConstruction)
	}
	if c.EfSearch <= 0 {
		return fmt.Errorf("efSearch must be positive, got %d", c.EfSearch)
	}

	// Validate distance metric
	switch c.DistanceMetric {
	case Cosine, L2, InnerProduct:
		// Valid
	default:
		return fmt.Errorf("unsupported distance metric: %s", c.DistanceMetric)
	}

	return nil
}
