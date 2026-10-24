#!/usr/bin/env bash
#
# Fetch the native test models listed in testdata/model_matrix.yaml into
# $VK_TEST_MODEL_DIR for the env-gated native smoke tests. VoiceKit ships NO
# model weights; this script only downloads entries that carry an explicit,
# pinned `url` in the matrix. Entries with only an `hf` (HuggingFace) pointer and
# no `url` are reported as skipped — this script never fabricates a download URL.
#
# Digest policy: if a matrix entry has an empty `sha256`, the script prints the
# computed digest so a maintainer can pin it in the matrix (it does not rewrite
# the tracked file). If `sha256` is set, the download is verified and a mismatch
# fails the run.
#
# Archives (.tar.bz2 / .tar.gz) are extracted into the entry directory after the
# digest check and the archive file is removed. GNU tar and bsdtar both refuse
# absolute paths and ".." members by default. Go consumers that need streaming
# size limits and symlink checks should use the go.glpx.pro/gdk/voice/modelstore
# package instead of this test-only script.
#
# Usage:
#   make fetch-test-models
#   VK_TEST_MODEL_DIR=/path/to/models scripts/fetch-test-models.sh
#   FETCH_DRY_RUN=1 scripts/fetch-test-models.sh   # list actions, no network
#
set -euo pipefail

MATRIX="${VK_MODEL_MATRIX:-testdata/model_matrix.yaml}"
DEST="${VK_TEST_MODEL_DIR:-/tmp/voicekit-models}"
DRY_RUN="${FETCH_DRY_RUN:-0}"

if ! command -v yq >/dev/null 2>&1; then
	echo "error: yq (mikefarah/yq v4) is required to parse ${MATRIX}." >&2
	echo "       install: https://github.com/mikefarah/yq/#install" >&2
	exit 1
fi

if [ ! -f "$MATRIX" ]; then
	echo "error: model matrix not found: $MATRIX" >&2
	exit 1
fi

sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum "$1" | awk '{print $1}'
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 "$1" | awk '{print $1}'
	else
		echo "error: neither sha256sum nor shasum is available" >&2
		exit 1
	fi
}

# Flatten every model entry (including the nested diarization/speaker lists) into
# TSV: section, id, hf, url, sha256. Empty fields are emitted as "-" because
# `read` with a tab IFS collapses consecutive tabs and would shift the columns.
entries="$(yq '
  [
    (.asr_offline.models[] | .section = "asr_offline"),
    (.asr_online.models[]  | .section = "asr_online"),
    (.vad.models[]         | .section = "vad"),
    (.tts.models[]         | .section = "tts"),
    (.diarization.segmentation[] | .section = "diarization.segmentation"),
    (.diarization.embedding[]    | .section = "diarization.embedding"),
    (.speaker.embedding[]        | .section = "speaker.embedding")
  ] | .[]
  | [.section, (.key // .family // .provider // .hf // "unknown"), ((.hf | select(length > 0)) // "-"), ((.url | select(length > 0)) // "-"), ((.sha256 | select(length > 0)) // "-")] | @tsv
' "$MATRIX")"

fetched=0
skipped=0
pinned=0
verified=0
failed=0

echo "Model matrix: $MATRIX"
echo "Destination:  $DEST"
[ "$DRY_RUN" = "1" ] && echo "Mode:         dry-run (no downloads)"
echo

while IFS=$'\t' read -r section id hf url want_sha; do
	[ -z "${section:-}" ] && continue
	label="${section}/${id}"
	# yq emits "" as "-" (see above); a literal "null" is also treated as empty.
	[ "$hf" = "-" ] && hf=""
	[ "$url" = "-" ] && url=""
	[ "$want_sha" = "-" ] && want_sha=""

	if [ -z "$url" ] || [ "$url" = "null" ]; then
		echo "SKIP  $label — no url pinned in matrix (hf: ${hf:-none})"
		skipped=$((skipped + 1))
		continue
	fi

	target_dir="$DEST/$section/$id"
	filename="$(basename "$url")"
	target="$target_dir/$filename"

	if [ "$DRY_RUN" = "1" ]; then
		echo "FETCH $label — would download $url -> $target"
		fetched=$((fetched + 1))
		continue
	fi

	mkdir -p "$target_dir"
	tmp="$(mktemp)"
	if ! curl -fL --retry 3 --retry-delay 2 -o "$tmp" "$url"; then
		echo "FAIL  $label — download failed: $url" >&2
		rm -f "$tmp"
		failed=$((failed + 1))
		continue
	fi

	got_sha="$(sha256_of "$tmp")"
	if [ -z "$want_sha" ] || [ "$want_sha" = "null" ]; then
		echo "PIN   $label — sha256=$got_sha (add to $MATRIX to pin)"
		pinned=$((pinned + 1))
	elif [ "$got_sha" != "$want_sha" ]; then
		echo "FAIL  $label — sha256 mismatch: want $want_sha got $got_sha" >&2
		rm -f "$tmp"
		failed=$((failed + 1))
		continue
	else
		echo "OK    $label — sha256 verified"
		verified=$((verified + 1))
	fi

	mv "$tmp" "$target"
	case "$filename" in
	*.tar.bz2 | *.tar.gz | *.tgz)
		if tar -xf "$target" -C "$target_dir"; then
			rm -f "$target"
		else
			echo "FAIL  $label — could not extract $filename" >&2
			failed=$((failed + 1))
			continue
		fi
		;;
	esac
	fetched=$((fetched + 1))
done <<<"$entries"

echo
echo "Summary: fetched=$fetched verified=$verified pinned=$pinned skipped=$skipped failed=$failed"
[ "$failed" -eq 0 ]
