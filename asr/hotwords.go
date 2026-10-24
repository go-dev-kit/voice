package asr

import (
	"fmt"
	"math"
	"strings"
)

const decodingModifiedBeamSearch = "modified_beam_search"

// Sherpa modeling units that tokenize hotwords. cjkchar is Sherpa's default
// when none is given.
const (
	modelingUnitCJKChar    = "cjkchar"
	modelingUnitBPE        = "bpe"
	modelingUnitByteBPE    = "bbpe"
	modelingUnitCJKCharBPE = "cjkchar+bpe"
)

// hotwordsSettings is the contextual-biasing slice of OnlineConfig and
// OfflineConfig, so both validate through one rule set.
type hotwordsSettings struct {
	file         string
	text         string
	score        float32
	modelingUnit string
	bpeVocab     string
}

func (h hotwordsSettings) active() bool {
	return h.file != "" || h.text != ""
}

// validateHotwords refuses every hotwords setup Sherpa would ignore silently
// or fail on. Sherpa reads hotwords only in the transducer recognizers
// (icefall and NeMo, online and offline), only under modified_beam_search, and
// encodes them with ModelingUnit: an unknown unit exits the process, and a BPE
// unit without a vocabulary dereferences a null encoder. Greedy search, CTC and
// seq2seq families ignore the settings without a word.
func validateHotwords(label, decodingMethod string, transducer bool, hw hotwordsSettings) []error {
	if !hw.active() {
		var errs []error
		if hw.score != 0 {
			errs = append(errs, fmt.Errorf("%s hotwords score is set without a hotwords file or text", label))
		}
		if hw.modelingUnit != "" || hw.bpeVocab != "" {
			errs = append(errs, fmt.Errorf("%s modeling unit and BPE vocab apply only to hotwords, which are not set", label))
		}
		return errs
	}
	if !transducer {
		return []error{fmt.Errorf("%s hotwords require a transducer model; Sherpa ignores them for other model types", label)}
	}

	var errs []error
	if hw.file != "" && hw.text != "" {
		errs = append(errs, fmt.Errorf("%s sets both a hotwords file and hotwords text; Sherpa would ignore the file", label))
	}
	if hw.text != "" && strings.TrimSpace(hw.text) == "" {
		errs = append(errs, fmt.Errorf("%s hotwords text is blank", label))
	}
	if decodingMethod != decodingModifiedBeamSearch {
		errs = append(errs, fmt.Errorf("%s hotwords require decoding method %q, got %q", label, decodingModifiedBeamSearch, decodingMethod))
	}
	if hw.file != "" {
		if err := requireFile(hw.file); err != nil {
			errs = append(errs, fmt.Errorf("%s hotwords file: %w", label, err))
		}
	}
	if s := float64(hw.score); math.IsNaN(s) || math.IsInf(s, 0) || s < 0 {
		errs = append(errs, fmt.Errorf("%s hotwords score must be a finite number >= 0 (0 keeps the Sherpa default 1.5), got %g", label, hw.score))
	}

	needsVocab := false
	switch hw.modelingUnit {
	case "", modelingUnitCJKChar:
	case modelingUnitBPE, modelingUnitByteBPE, modelingUnitCJKCharBPE:
		needsVocab = true
	default:
		errs = append(errs, fmt.Errorf("%s modeling unit %q is not one of %s, %s, %s, %s", label, hw.modelingUnit,
			modelingUnitCJKChar, modelingUnitBPE, modelingUnitByteBPE, modelingUnitCJKCharBPE))
	}
	switch {
	case hw.bpeVocab != "":
		if err := requireFile(hw.bpeVocab); err != nil {
			errs = append(errs, fmt.Errorf("%s BPE vocab: %w", label, err))
		}
	case needsVocab:
		errs = append(errs, fmt.Errorf("%s modeling unit %q requires a BPE vocab file", label, hw.modelingUnit))
	}
	return errs
}

func (o OnlineConfig) hotwords() hotwordsSettings {
	return hotwordsSettings{file: o.HotwordsFile, text: o.HotwordsText, score: o.HotwordsScore, modelingUnit: o.ModelingUnit, bpeVocab: o.BpeVocab}
}

func (o OfflineConfig) hotwords() hotwordsSettings {
	return hotwordsSettings{file: o.HotwordsFile, score: o.HotwordsScore, modelingUnit: o.ModelingUnit, bpeVocab: o.BpeVocab}
}

// hasTransducerPaths reports whether the online entry is a transducer.
func (o OnlineConfig) hasTransducerPaths() bool {
	return o.EncoderPath != "" || o.DecoderPath != "" || o.JoinerPath != ""
}
