# VoiceKit developer targets.

GO_PACKAGES := ./...
BENCH_PACKAGES ?= ./audio ./diarization ./speaker ./indexing ./asr ./meeting ./evaluation .
BENCH_COUNT ?= 6
BENCH_DIR ?= /tmp/voicekit-benchmarks
PROFILE_DIR ?= /tmp/voicekit-profiles
GOLANGCI_LINT_VERSION ?= v2.14.0
GOLANGCI_LINT_VERSION_RAW := $(patsubst v%,%,$(GOLANGCI_LINT_VERSION))
GO_BIN := $(shell go env GOPATH)/bin

.PHONY: verify forbidden-files mod-verify fmt-check tidy-check tools lint test vet race \
	fetch-test-models \
	bench bench-all bench-audio bench-speaker bench-diarization bench-memory bench-load bench-asr-init \
	bench-report bench-compare bench-profile bench-memprofile bench-clean bench-comprehensive \
	bench-help

verify: forbidden-files mod-verify fmt-check tidy-check lint vet test race

# Fetch native test models listed in testdata/model_matrix.yaml into
# $VK_TEST_MODEL_DIR (default /tmp/voicekit-models) for the env-gated native
# smoke tests. Only entries with a pinned url are downloaded; set FETCH_DRY_RUN=1
# to list actions without downloading.
fetch-test-models:
	@scripts/fetch-test-models.sh

forbidden-files:
	@scripts/check-forbidden-files.sh

mod-verify:
	@go mod verify

fmt-check:
	@unformatted="$$(gofmt -l $$(git ls-files '*.go'))"; \
	if [ -n "$$unformatted" ]; then \
		echo "The following Go files are not gofmt-formatted:"; \
		echo "$$unformatted"; \
		exit 1; \
	fi

tidy-check:
	@go mod tidy
	@git diff --exit-code go.mod go.sum

tools:
	@installed="$$(golangci-lint version 2>/dev/null | awk '{print $$4}' || true)"; \
	if [ "$$installed" != "$(GOLANGCI_LINT_VERSION_RAW)" ]; then \
		echo "Installing golangci-lint $(GOLANGCI_LINT_VERSION)"; \
		go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION); \
	fi

lint: tools
	@PATH="$(GO_BIN):$$PATH" golangci-lint run ./...

test:
	@go test $(GO_PACKAGES)

vet:
	@go vet $(GO_PACKAGES)

race:
	@if [ "$$(go env GOARCH)" = "amd64" ] || [ "$$(go env GOOS)" = "darwin" ]; then \
		go test -race $(GO_PACKAGES); \
	else \
		echo "Skipping -race on $$(go env GOOS)/$$(go env GOARCH); running plain tests instead"; \
		go test $(GO_PACKAGES); \
	fi

bench: bench-all

bench-all:
	@mkdir -p "$(BENCH_DIR)"
	@go test -bench=. -benchmem -run=^$$ -count=$(BENCH_COUNT) $(BENCH_PACKAGES) | tee "$(BENCH_DIR)/raw_results_$$(date +%Y%m%d_%H%M%S).txt"

bench-audio:
	@go test -bench=. -benchmem -run=^$$ -count=$(BENCH_COUNT) ./audio

bench-speaker:
	@go test -bench=. -benchmem -run=^$$ -count=$(BENCH_COUNT) ./speaker

bench-diarization:
	@go test -bench=. -benchmem -run=^$$ -count=$(BENCH_COUNT) ./diarization

bench-memory:
	@go test -bench=BufferPool -benchmem -run=^$$ -count=$(BENCH_COUNT) ./audio

# Recognizer init-cost / first-chunk benchmarks. Requires real model files via
# VOICEKIT_ONLINE_ASR_* env vars (see asr/init_benchmark_test.go); skips otherwise.
bench-asr-init:
	@go test -bench='NewSherpaOnlineModel|FirstChunk' -benchmem -run=^$$ -count=$(BENCH_COUNT) ./asr

bench-load:
	@go test -bench=Concurrent -benchmem -run=^$$ -count=$(BENCH_COUNT) .

bench-report:
	@mkdir -p "$(BENCH_DIR)"
	@go test -bench=. -benchmem -run=^$$ -count=$(BENCH_COUNT) $(BENCH_PACKAGES) | tee "$(BENCH_DIR)/benchmark_report_$$(date +%Y%m%d_%H%M%S).txt"

bench-compare:
	@if ! command -v benchstat >/dev/null 2>&1; then \
		echo "benchstat is required. Install with: go install golang.org/x/perf/cmd/benchstat@latest"; \
		exit 1; \
	fi
	@if [ -z "$(BEFORE)" ] || [ -z "$(AFTER)" ]; then \
		echo "Usage: make bench-compare BEFORE=/tmp/voicekit-benchmarks/before.txt AFTER=/tmp/voicekit-benchmarks/after.txt"; \
		exit 1; \
	fi
	@benchstat "$(BEFORE)" "$(AFTER)"

bench-profile:
	@mkdir -p "$(PROFILE_DIR)"
	@for pkg in ./audio ./speaker ./diarization .; do \
		name=$$(echo "$$pkg" | sed 's|^\.$$|root|; s|^\./||; s|/|_|g'); \
		go test -bench=. -benchmem -run='^$$' -cpuprofile="$(PROFILE_DIR)/cpu_$$name.prof" "$$pkg" || exit 1; \
	done
	@echo "CPU profiles written to $(PROFILE_DIR)/cpu_*.prof"

bench-memprofile:
	@mkdir -p "$(PROFILE_DIR)"
	@for pkg in ./audio ./speaker ./diarization .; do \
		name=$$(echo "$$pkg" | sed 's|^\.$$|root|; s|^\./||; s|/|_|g'); \
		go test -bench=. -benchmem -run='^$$' -memprofile="$(PROFILE_DIR)/mem_$$name.prof" "$$pkg" || exit 1; \
	done
	@echo "Memory profiles written to $(PROFILE_DIR)/mem_*.prof"

bench-clean:
	@for dir in "$(BENCH_DIR)" "$(PROFILE_DIR)"; do \
		case "$$dir" in \
			/tmp/voicekit-*) rm -rf "$$dir" ;; \
			*) echo "Refusing to remove non-VoiceKit temp directory: $$dir"; exit 1 ;; \
		esac; \
	done

bench-comprehensive: bench-report

bench-help:
	@echo "VoiceKit targets:"
	@echo "  verify            - Run repository hygiene, formatting, lint, vet, tests, and race gate"
	@echo "  lint              - Run golangci-lint v2"
	@echo "  test              - Run go test ./..."
	@echo "  vet               - Run go vet ./..."
	@echo "  race              - Run go test -race ./... when supported"
	@echo "  bench-all         - Run broad benchmarks into $(BENCH_DIR)"
	@echo "  bench-compare     - Compare two benchmark files with benchstat"
	@echo "  bench-profile     - Write CPU profile into $(PROFILE_DIR)"
	@echo "  bench-memprofile  - Write memory profile into $(PROFILE_DIR)"
	@echo "  bench-clean       - Remove VoiceKit benchmark/profile temp directories"
	@echo "  fetch-test-models - Download pinned native test models into \$$VK_TEST_MODEL_DIR"
