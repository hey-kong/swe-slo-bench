# SWE SLO Bench

Search for the highest tested client concurrency satisfying **P90 TTFT < 10 s** and **mean decode TPS > 15 tokens/s**, using local LongSWE-Bench or SWE-Bench prompts and an already running SGLang server. Both entry points share the same requests, metrics, search, and time budget.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/hey-kong/swe-slo-bench.git
cd swe-slo-bench
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Configure and run

Replace the placeholders with your local paths and server URL:

```bash
DATASET="<local-LongSWE-Bench-128K-directory>"
TOKENIZER="<local-model-or-tokenizer-directory>"
BASE_URL="<SGLang-server-URL>"
OUTPUT_DIR="<new-results-directory>"
LABEL="hicache"

# Inspect local data before benchmarking.
python longswe_slo.py \
  --dataset "$DATASET" --tokenizer "$TOKENIZER" \
  --output-len 1024 --output-dir "$OUTPUT_DIR" --inspect-only

# Start the target server separately, then benchmark it.
python longswe_slo.py \
  --dataset "$DATASET" --tokenizer "$TOKENIZER" --base-url "$BASE_URL" \
  --label "$LABEL" --output-len 1024 --output-dir "$OUTPUT_DIR" \
  --confirm-requests 100 --budget-seconds 1200
```

Local inputs can be a `save_to_disk` dataset, a Parquet/JSON/JSONL file, or a downloaded dataset directory with split files such as `data/test-*.parquet`. Directory loading selects only `--split test` by default, never train/dev shards. The server URL must include its scheme, host, and port.

For other server configurations, change `LABEL` to `hisparse`, `adaptive_hisparse`, or `kvduo`, and choose a new `OUTPUT_DIR`. Run one configuration at a time. Output directories must not already exist.

All data and tokenizer files are loaded locally. Use `--prompt-column` if field detection is ambiguous.

### SWE-Bench

Use `swebench_slo.py` with the same options and server configurations:

```bash
DATASET="<local-SWE-Bench-directory-or-parquet-file>"
OUTPUT_DIR="<new-SWE-Bench-results-directory>"

python swebench_slo.py \
  --dataset "$DATASET" --tokenizer "$TOKENIZER" \
  --output-len 1024 --output-dir "$OUTPUT_DIR" --inspect-only

python swebench_slo.py \
  --dataset "$DATASET" --tokenizer "$TOKENIZER" --base-url "$BASE_URL" \
  --label "$LABEL" --output-len 1024 --output-dir "$OUTPUT_DIR" \
  --confirm-requests 100 --budget-seconds 1200
```

For the approximately 40K workload, use a locally downloaded [SWE-bench_bm25_40K](https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_40K): its `text` column already includes the issue and retrieved code. Its 40K limit concerns code context under the source tokenizer; actual full-prompt lengths vary with the serving tokenizer.

Original SWE-Bench task data, including the original ModelScope mirror, may only contain `problem_statement` and metadata, without prepared code context. Such data is rejected by default. Add `--issue-only` to **both** commands above to explicitly test issue text alone; this is a short-input workload, not SWE-Bench 40K. The mode and warning are recorded in `config.json`. Reference patches and test answers are never automatically included in prompts. No code retrieval or dataset download runs automatically.

Keep SWE-Bench and LongSWE-Bench results in separate output directories. `--inspect-only` reports fields, selected input mode, source files, and local token counts before sending requests. `longswe_slo.py --dataset-kind swebench` is equivalent to the SWE-Bench entry point.

## Test settings

- Search uses fixed-time probes; final confirmation targets the same **100 requests** by default (`--confirm-requests`). Every stage starts from the same seeded prompt order, so complete confirmations use the same source samples across configurations. This target applies to the final confirmation/continuation stage, not each search probe or the whole run.
- A **1,200-second maximum global send window** starts after local data preparation and warmup. Search, cache flushes, inter-stage draining, and confirmation share this budget. Confirmation stops admitting when it reaches 100 requests or the global deadline, whichever comes first. All admitted requests drain normally; wall time can exceed 20 minutes. A complete confirmation may finish earlier than 20 minutes.
- Concurrency starts at 8 and is capped at 64. Doubling and binary search use 180-second probes; `--confirm-seconds` reserves a target amount of remaining time, not an extra window or required minimum. If no probe passed, the final stage continues the last candidate with the same request target.
- Confirmation requires all target requests to complete successfully and meet the SLO; a partial workload is `INSUFFICIENT` even if its observed metrics pass. Errors remain `ERROR` and are not replaced with extra requests. `--min-completed` (default 20) is a sample gate, not a request-count stopping rule; `--confirm-requests` must be at least this value.
- Requests drain completely between stages without resetting the global clock. Errors stop the run early; existing per-request timeouts still apply. The former `--drain-reserve` option has been removed.
- Fixed output length, temperature 0, and EOS ignored. Keep output length identical across configurations.
- Mean decode TPS is the arithmetic mean of `(output_tokens - 1) / (E2E - TTFT)` per request, excluding TTFT.
- Unique request prefixes prevent full-prefix reuse by default. Add `--allow-prefix-reuse` to replay unchanged prompts. Cache is flushed between stages unless `--keep-cache` is set.
- Full dataset prompts are used without truncation or a chat template. Actual token counts depend on the tokenizer.

## Results

See `summary.json` for `highest_passing_concurrency`, `confirmed_concurrency`, `smallest_observed_fail`, `confirmation_target_requests`, `confirmation_completed`, `confirmation_status`, and per-stage metrics. A passing probe and a successful independent confirmation are reported separately; incomplete confirmation never produces a confirmed result, and inconsistent PASS/FAIL observations are flagged. Stages record `target_requests`, `request_target_met`, and `stop_reason`. `requests.jsonl` contains individual request measurements; `config.json` records settings and dataset details.

Errors and insufficient samples do not count as passing. Results are observations within the send window, not proof of the global maximum or a full 20-minute run at one concurrency. Timing fields include `send_window_seconds` (the configured cap), `last_request_started_elapsed_s`, `final_drain_seconds`, and `elapsed_wall_seconds`. Per-stage `admission_limit_seconds` is the available window; `admission_seconds` ends at the target/deadline and `drain_seconds` covers the remaining in-flight work.

Run `python longswe_slo.py --help` for all options.

## Timing checks

CPU-only tests use temporary local fixtures; no GPU or running SGLang server is required:

```bash
python -m unittest discover -v
```


