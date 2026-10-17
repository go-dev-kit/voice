package voicekit

import "testing"

func TestVoiceKitCloseIsIdempotent(t *testing.T) {
	vk, err := NewVoiceKit(nil)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		if err := vk.Close(); err != nil {
			t.Fatalf("close %d: %v", i, err)
		}
	}
}
