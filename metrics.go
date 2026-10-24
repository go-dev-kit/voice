package voicekit

import (
	"sync/atomic"
	"time"
)

// MetricsCollector collects performance metrics for VoiceKit operations
type MetricsCollector struct {
	// Audio processing metrics
	audioProcessCount   int64
	audioProcessLatency int64 // nanoseconds
	audioErrors         int64

	// Speaker recognition metrics
	speakerIdentifyCount   int64
	speakerIdentifyLatency int64 // nanoseconds
	speakerVerifyCount     int64
	speakerVerifyLatency   int64 // nanoseconds
	speakerRegisterCount   int64
	speakerRegisterLatency int64 // nanoseconds
	speakerErrors          int64

	// Diarization metrics
	diarizationProcessCount   int64
	diarizationProcessLatency int64 // nanoseconds
	diarizationErrors         int64

	// Overall system metrics
	totalRequests int64
	totalErrors   int64
	// uptime is the collector's start as Unix nanoseconds; atomic because
	// Reset can run while GetMetrics takes a snapshot.
	uptime atomic.Int64
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	m := &MetricsCollector{}
	m.uptime.Store(time.Now().UnixNano())
	return m
}

// RecordAudioProcessing records audio processing metrics
func (m *MetricsCollector) RecordAudioProcessing(duration time.Duration, success bool) {
	atomic.AddInt64(&m.audioProcessCount, 1)
	atomic.AddInt64(&m.audioProcessLatency, duration.Nanoseconds())
	atomic.AddInt64(&m.totalRequests, 1)

	if !success {
		atomic.AddInt64(&m.audioErrors, 1)
		atomic.AddInt64(&m.totalErrors, 1)
	}
}

// RecordSpeakerIdentify records speaker identification metrics
func (m *MetricsCollector) RecordSpeakerIdentify(duration time.Duration, success bool) {
	atomic.AddInt64(&m.speakerIdentifyCount, 1)
	atomic.AddInt64(&m.speakerIdentifyLatency, duration.Nanoseconds())
	atomic.AddInt64(&m.totalRequests, 1)

	if !success {
		atomic.AddInt64(&m.speakerErrors, 1)
		atomic.AddInt64(&m.totalErrors, 1)
	}
}

// RecordSpeakerVerify records speaker verification metrics
func (m *MetricsCollector) RecordSpeakerVerify(duration time.Duration, success bool) {
	atomic.AddInt64(&m.speakerVerifyCount, 1)
	atomic.AddInt64(&m.speakerVerifyLatency, duration.Nanoseconds())
	atomic.AddInt64(&m.totalRequests, 1)

	if !success {
		atomic.AddInt64(&m.speakerErrors, 1)
		atomic.AddInt64(&m.totalErrors, 1)
	}
}

// RecordSpeakerRegister records speaker registration metrics
func (m *MetricsCollector) RecordSpeakerRegister(duration time.Duration, success bool) {
	atomic.AddInt64(&m.speakerRegisterCount, 1)
	atomic.AddInt64(&m.speakerRegisterLatency, duration.Nanoseconds())
	atomic.AddInt64(&m.totalRequests, 1)

	if !success {
		atomic.AddInt64(&m.speakerErrors, 1)
		atomic.AddInt64(&m.totalErrors, 1)
	}
}

// RecordDiarization records diarization processing metrics
func (m *MetricsCollector) RecordDiarization(duration time.Duration, success bool) {
	atomic.AddInt64(&m.diarizationProcessCount, 1)
	atomic.AddInt64(&m.diarizationProcessLatency, duration.Nanoseconds())
	atomic.AddInt64(&m.totalRequests, 1)

	if !success {
		atomic.AddInt64(&m.diarizationErrors, 1)
		atomic.AddInt64(&m.totalErrors, 1)
	}
}

// GetMetrics returns a point-in-time snapshot of the collected metrics. Each call
// returns a freshly allocated snapshot that the caller fully owns; callers may
// retain it for as long as they like.
func (m *MetricsCollector) GetMetrics() *MetricsSnapshot {
	snapshot := &MetricsSnapshot{}
	snapshot.AudioProcessCount = atomic.LoadInt64(&m.audioProcessCount)
	snapshot.AudioProcessLatency = time.Duration(atomic.LoadInt64(&m.audioProcessLatency))
	snapshot.AudioErrors = atomic.LoadInt64(&m.audioErrors)

	snapshot.SpeakerIdentifyCount = atomic.LoadInt64(&m.speakerIdentifyCount)
	snapshot.SpeakerIdentifyLatency = time.Duration(atomic.LoadInt64(&m.speakerIdentifyLatency))
	snapshot.SpeakerVerifyCount = atomic.LoadInt64(&m.speakerVerifyCount)
	snapshot.SpeakerVerifyLatency = time.Duration(atomic.LoadInt64(&m.speakerVerifyLatency))
	snapshot.SpeakerRegisterCount = atomic.LoadInt64(&m.speakerRegisterCount)
	snapshot.SpeakerRegisterLatency = time.Duration(atomic.LoadInt64(&m.speakerRegisterLatency))
	snapshot.SpeakerErrors = atomic.LoadInt64(&m.speakerErrors)

	snapshot.DiarizationProcessCount = atomic.LoadInt64(&m.diarizationProcessCount)
	snapshot.DiarizationProcessLatency = time.Duration(atomic.LoadInt64(&m.diarizationProcessLatency))
	snapshot.DiarizationErrors = atomic.LoadInt64(&m.diarizationErrors)

	snapshot.TotalRequests = atomic.LoadInt64(&m.totalRequests)
	snapshot.TotalErrors = atomic.LoadInt64(&m.totalErrors)
	snapshot.Uptime = time.Since(time.Unix(0, m.uptime.Load()))

	return snapshot
}

// MetricsSnapshot represents a point-in-time snapshot of metrics. Each snapshot is
// independently owned by the caller of GetMetrics.
type MetricsSnapshot struct {
	// Audio processing
	AudioProcessCount   int64         `json:"audio_process_count"`
	AudioProcessLatency time.Duration `json:"audio_process_latency"`
	AudioErrors         int64         `json:"audio_errors"`

	// Speaker recognition
	SpeakerIdentifyCount   int64         `json:"speaker_identify_count"`
	SpeakerIdentifyLatency time.Duration `json:"speaker_identify_latency"`
	SpeakerVerifyCount     int64         `json:"speaker_verify_count"`
	SpeakerVerifyLatency   time.Duration `json:"speaker_verify_latency"`
	SpeakerRegisterCount   int64         `json:"speaker_register_count"`
	SpeakerRegisterLatency time.Duration `json:"speaker_register_latency"`
	SpeakerErrors          int64         `json:"speaker_errors"`

	// Diarization
	DiarizationProcessCount   int64         `json:"diarization_process_count"`
	DiarizationProcessLatency time.Duration `json:"diarization_process_latency"`
	DiarizationErrors         int64         `json:"diarization_errors"`

	// Overall
	TotalRequests int64         `json:"total_requests"`
	TotalErrors   int64         `json:"total_errors"`
	Uptime        time.Duration `json:"uptime"`
}

// GetAudioProcessLatencyAvg returns average audio processing latency
func (s *MetricsSnapshot) GetAudioProcessLatencyAvg() time.Duration {
	if s.AudioProcessCount == 0 {
		return 0
	}
	return s.AudioProcessLatency / time.Duration(s.AudioProcessCount)
}

// GetSpeakerIdentifyLatencyAvg returns average speaker identification latency
func (s *MetricsSnapshot) GetSpeakerIdentifyLatencyAvg() time.Duration {
	if s.SpeakerIdentifyCount == 0 {
		return 0
	}
	return s.SpeakerIdentifyLatency / time.Duration(s.SpeakerIdentifyCount)
}

// GetSpeakerVerifyLatencyAvg returns average speaker verification latency
func (s *MetricsSnapshot) GetSpeakerVerifyLatencyAvg() time.Duration {
	if s.SpeakerVerifyCount == 0 {
		return 0
	}
	return s.SpeakerVerifyLatency / time.Duration(s.SpeakerVerifyCount)
}

// GetSpeakerRegisterLatencyAvg returns average speaker registration latency
func (s *MetricsSnapshot) GetSpeakerRegisterLatencyAvg() time.Duration {
	if s.SpeakerRegisterCount == 0 {
		return 0
	}
	return s.SpeakerRegisterLatency / time.Duration(s.SpeakerRegisterCount)
}

// GetDiarizationLatencyAvg returns average diarization processing latency
func (s *MetricsSnapshot) GetDiarizationLatencyAvg() time.Duration {
	if s.DiarizationProcessCount == 0 {
		return 0
	}
	return s.DiarizationProcessLatency / time.Duration(s.DiarizationProcessCount)
}

// GetErrorRate returns overall error rate as percentage
func (s *MetricsSnapshot) GetErrorRate() float64 {
	if s.TotalRequests == 0 {
		return 0.0
	}
	return float64(s.TotalErrors) / float64(s.TotalRequests) * 100.0
}

// GetAudioErrorRate returns audio processing error rate as percentage
func (s *MetricsSnapshot) GetAudioErrorRate() float64 {
	if s.AudioProcessCount == 0 {
		return 0.0
	}
	return float64(s.AudioErrors) / float64(s.AudioProcessCount) * 100.0
}

// GetSpeakerErrorRate returns speaker recognition error rate as percentage
func (s *MetricsSnapshot) GetSpeakerErrorRate() float64 {
	totalSpeakerOps := s.SpeakerIdentifyCount + s.SpeakerVerifyCount + s.SpeakerRegisterCount
	if totalSpeakerOps == 0 {
		return 0.0
	}
	return float64(s.SpeakerErrors) / float64(totalSpeakerOps) * 100.0
}

// GetDiarizationErrorRate returns diarization error rate as percentage
func (s *MetricsSnapshot) GetDiarizationErrorRate() float64 {
	if s.DiarizationProcessCount == 0 {
		return 0.0
	}
	return float64(s.DiarizationErrors) / float64(s.DiarizationProcessCount) * 100.0
}

// GetThroughput returns requests per second
func (s *MetricsSnapshot) GetThroughput() float64 {
	uptimeSeconds := s.Uptime.Seconds()
	if uptimeSeconds == 0 {
		return 0.0
	}
	return float64(s.TotalRequests) / uptimeSeconds
}

// Reset resets all metrics (useful for testing)
func (m *MetricsCollector) Reset() {
	atomic.StoreInt64(&m.audioProcessCount, 0)
	atomic.StoreInt64(&m.audioProcessLatency, 0)
	atomic.StoreInt64(&m.audioErrors, 0)

	atomic.StoreInt64(&m.speakerIdentifyCount, 0)
	atomic.StoreInt64(&m.speakerIdentifyLatency, 0)
	atomic.StoreInt64(&m.speakerVerifyCount, 0)
	atomic.StoreInt64(&m.speakerVerifyLatency, 0)
	atomic.StoreInt64(&m.speakerRegisterCount, 0)
	atomic.StoreInt64(&m.speakerRegisterLatency, 0)
	atomic.StoreInt64(&m.speakerErrors, 0)

	atomic.StoreInt64(&m.diarizationProcessCount, 0)
	atomic.StoreInt64(&m.diarizationProcessLatency, 0)
	atomic.StoreInt64(&m.diarizationErrors, 0)

	atomic.StoreInt64(&m.totalRequests, 0)
	atomic.StoreInt64(&m.totalErrors, 0)

	m.uptime.Store(time.Now().UnixNano())
}
