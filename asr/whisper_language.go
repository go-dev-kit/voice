package asr

import (
	"fmt"
	"strings"
)

// whisperLanguages is the openai/whisper tokenizer LANGUAGES set (100 codes).
// sherpa-onnx exits the host process at the first decode when the configured
// language is outside its lang2id map, so anything else must be refused at
// config time. An empty language is the only way to request auto-detection.
var whisperLanguages = map[string]struct{}{
	"en": {}, "zh": {}, "de": {}, "es": {}, "ru": {}, "ko": {}, "fr": {}, "ja": {}, "pt": {}, "tr": {},
	"pl": {}, "ca": {}, "nl": {}, "ar": {}, "sv": {}, "it": {}, "id": {}, "hi": {}, "fi": {}, "vi": {},
	"he": {}, "uk": {}, "el": {}, "ms": {}, "cs": {}, "ro": {}, "da": {}, "hu": {}, "ta": {}, "no": {},
	"th": {}, "ur": {}, "hr": {}, "bg": {}, "lt": {}, "la": {}, "mi": {}, "ml": {}, "cy": {}, "sk": {},
	"te": {}, "fa": {}, "lv": {}, "bn": {}, "sr": {}, "az": {}, "sl": {}, "kn": {}, "et": {}, "mk": {},
	"br": {}, "eu": {}, "is": {}, "hy": {}, "ne": {}, "mn": {}, "bs": {}, "kk": {}, "sq": {}, "sw": {},
	"gl": {}, "mr": {}, "pa": {}, "si": {}, "km": {}, "sn": {}, "yo": {}, "so": {}, "af": {}, "oc": {},
	"ka": {}, "be": {}, "tg": {}, "sd": {}, "gu": {}, "am": {}, "yi": {}, "lo": {}, "uz": {}, "fo": {},
	"ht": {}, "ps": {}, "tk": {}, "nn": {}, "mt": {}, "sa": {}, "lb": {}, "my": {}, "bo": {}, "tl": {},
	"mg": {}, "as": {}, "tt": {}, "haw": {}, "ln": {}, "ha": {}, "ba": {}, "jw": {}, "su": {}, "yue": {},
}

// whisperLanguageCode maps a configured language to the value sherpa receives:
// "" and "auto" (the SenseVoice convention) both mean auto-detect.
func whisperLanguageCode(language string) string {
	language = strings.TrimSpace(language)
	if strings.EqualFold(language, "auto") {
		return ""
	}
	return language
}

func validateWhisperOptions(offline OfflineConfig) []error {
	var errs []error
	if code := whisperLanguageCode(offline.Language); code != "" {
		if _, ok := whisperLanguages[code]; !ok {
			errs = append(errs, fmt.Errorf("offline whisper language %q is not a Whisper language code (use an openai/whisper code such as \"en\", \"de\" or \"yue\", or leave it empty or \"auto\" to auto-detect)", offline.Language))
		}
	}
	switch offline.Task {
	case "", "transcribe", "translate":
	default:
		errs = append(errs, fmt.Errorf("offline whisper task %q must be \"transcribe\" or \"translate\"", offline.Task))
	}
	return errs
}
