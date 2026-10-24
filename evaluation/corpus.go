package evaluation

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// MaxCorpusLineBytes is the longest corpus line the loaders accept. A longer
// line fails with ErrCorpusLineTooLong instead of growing memory unbounded.
const MaxCorpusLineBytes = 1 << 20

// ErrCorpusLineTooLong reports a corpus line above MaxCorpusLineBytes.
var ErrCorpusLineTooLong = errors.New("corpus line too long")

// CorpusError is a strict-parse failure tied to a 1-based input line.
type CorpusError struct {
	Line int
	Err  error
}

func (e *CorpusError) Error() string { return fmt.Sprintf("corpus line %d: %v", e.Line, e.Err) }
func (e *CorpusError) Unwrap() error { return e.Err }

// FLEURSRow is one row of a FLEURS split TSV file (no header). Columns are
// id, file, raw transcription, normalized transcription, characters,
// num_samples and gender.
type FLEURSRow struct {
	ID string
	// File is the audio file name as written in the TSV.
	File string
	// AudioPath is File resolved against the audio directory.
	AudioPath string
	// Transcription is the raw, cased and punctuated reference text.
	Transcription string
	// Normalized is the dataset's own normalized transcription.
	Normalized string
	// Chars is the character-tokenized transcription column as written,
	// with "|" marking word boundaries.
	Chars      string
	NumSamples int
	Gender     string
}

// LoadFLEURSTSV parses a FLEURS split TSV. Audio paths are File joined to
// audioDir; an absolute File is kept. A leading UTF-8 byte order mark is
// ignored. Blank lines are skipped; any other
// line that does not have exactly seven tab-separated columns, has an empty
// id or file, or has a non-integer num_samples fails with a *CorpusError.
func LoadFLEURSTSV(r io.Reader, audioDir string) ([]FLEURSRow, error) {
	var rows []FLEURSRow
	err := scanCorpusLines(r, func(line int, text string) error {
		cols := strings.Split(text, "\t")
		if len(cols) != 7 {
			return fmt.Errorf("want 7 tab-separated columns, got %d", len(cols))
		}
		if cols[0] == "" || cols[1] == "" {
			return errors.New("id and file are required")
		}
		samples, err := strconv.Atoi(cols[5])
		if err != nil || samples < 0 {
			return fmt.Errorf("invalid num_samples %q", cols[5])
		}
		rows = append(rows, FLEURSRow{
			ID:            cols[0],
			File:          cols[1],
			AudioPath:     resolveCorpusPath(audioDir, cols[1]),
			Transcription: cols[2],
			Normalized:    cols[3],
			Chars:         cols[4],
			NumSamples:    samples,
			Gender:        cols[6],
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return rows, nil
}

// LoadFLEURSTSVFile reads a FLEURS TSV. Audio is looked up in a "wav"
// directory next to the file when one exists (the layout FLEURS dumps use),
// otherwise next to the file.
func LoadFLEURSTSVFile(path string) ([]FLEURSRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	dir := filepath.Dir(path)
	if info, statErr := os.Stat(filepath.Join(dir, "wav")); statErr == nil && info.IsDir() {
		dir = filepath.Join(dir, "wav")
	}
	return LoadFLEURSTSV(f, dir)
}

// ManifestRow is one entry of a JSONL manifest.
type ManifestRow struct {
	ID string `json:"id"`
	// Audio is the audio path as written; AudioPath is it resolved against
	// the manifest directory.
	Audio     string `json:"audio"`
	AudioPath string `json:"-"`
	Reference string `json:"reference"`
	// DurationS is the audio length in seconds; zero when absent.
	DurationS float64 `json:"duration_s,omitempty"`
	Language  string  `json:"language,omitempty"`
	Speaker   string  `json:"speaker,omitempty"`
}

// LoadManifestJSONL parses a JSONL manifest with fields id, audio, reference
// and optional duration_s, language, speaker. Relative audio paths resolve
// against baseDir. A leading UTF-8 byte order mark is ignored. Each line must
// be exactly one JSON object followed only by whitespace. Blank lines are
// skipped. Invalid JSON, unknown fields,
// trailing data, missing id/audio/reference, a negative or non-finite
// duration_s and duplicate ids fail with a *CorpusError.
func LoadManifestJSONL(r io.Reader, baseDir string) ([]ManifestRow, error) {
	var rows []ManifestRow
	seen := make(map[string]struct{})
	err := scanCorpusLines(r, func(line int, text string) error {
		dec := json.NewDecoder(strings.NewReader(text))
		dec.DisallowUnknownFields()
		var row ManifestRow
		if err := dec.Decode(&row); err != nil {
			return err
		}
		// Exactly one object per line: anything but whitespace after it
		// (including a stray "}" that dec.More would accept) is an error.
		if err := dec.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
			return errors.New("unexpected data after JSON object")
		}
		switch {
		case row.ID == "":
			return errors.New("id is required")
		case row.Audio == "":
			return errors.New("audio is required")
		case row.Reference == "":
			return errors.New("reference is required")
		case row.DurationS < 0:
			return fmt.Errorf("invalid duration_s %v", row.DurationS)
		}
		if _, dup := seen[row.ID]; dup {
			return fmt.Errorf("duplicate id %q", row.ID)
		}
		seen[row.ID] = struct{}{}
		row.AudioPath = resolveCorpusPath(baseDir, row.Audio)
		rows = append(rows, row)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return rows, nil
}

// LoadManifestJSONLFile reads a JSONL manifest, resolving relative audio
// paths against the manifest's directory.
func LoadManifestJSONLFile(path string) ([]ManifestRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	return LoadManifestJSONL(f, filepath.Dir(path))
}

// resolveCorpusPath joins a relative p to base. An absolute p, or an empty
// base, returns p unchanged, and a relative p containing ".." is not
// rejected, so the result is not confined to base. AudioPath values built
// from it are not confined either: a consumer must confine them (for example
// with os.Root or a prefix check) before opening audio named by an untrusted
// manifest.
func resolveCorpusPath(base string, p string) string {
	if filepath.IsAbs(p) || base == "" {
		return p
	}
	return filepath.Join(base, p)
}

var utf8BOM = []byte{0xEF, 0xBB, 0xBF}

// scanCorpusLines feeds each non-blank line (without the line terminator) to
// fn, skipping a leading UTF-8 BOM, enforcing MaxCorpusLineBytes and wrapping failures with the line number.
func scanCorpusLines(r io.Reader, fn func(line int, text string) error) error {
	br := bufio.NewReaderSize(r, 64<<10)
	if bom, err := br.Peek(len(utf8BOM)); err == nil && bytes.Equal(bom, utf8BOM) {
		_, _ = br.Discard(len(utf8BOM))
	}
	for line := 1; ; line++ {
		text, err := readBoundedLine(br)
		if err != nil && !errors.Is(err, io.EOF) {
			return &CorpusError{Line: line, Err: err}
		}
		eof := errors.Is(err, io.EOF)
		if len(bytes.TrimSpace(text)) > 0 {
			if ferr := fn(line, string(bytes.TrimSuffix(text, []byte("\r")))); ferr != nil {
				return &CorpusError{Line: line, Err: ferr}
			}
		}
		if eof {
			return nil
		}
	}
}

// readBoundedLine reads one line without its "\n". It returns io.EOF with
// the final unterminated text, or ErrCorpusLineTooLong past the bound.
func readBoundedLine(br *bufio.Reader) ([]byte, error) {
	var line []byte
	for {
		chunk, err := br.ReadSlice('\n')
		line = append(line, chunk...)
		if len(line) > MaxCorpusLineBytes+1 {
			return nil, ErrCorpusLineTooLong
		}
		switch {
		case err == nil:
			line = line[:len(line)-1]
			if len(line) > MaxCorpusLineBytes {
				return nil, ErrCorpusLineTooLong
			}
			return line, nil
		case errors.Is(err, bufio.ErrBufferFull):
			continue
		default:
			return line, err
		}
	}
}
