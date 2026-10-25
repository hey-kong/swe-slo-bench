#!/usr/bin/env python3
"""Budgeted LongSWE-Bench / SWE-Bench concurrency search against SGLang /generate.

Local dataset/tokenizer only. No model hub downloads. See --help and README.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import uuid
import warnings


def percentile(values, q):
    if not values:
        return None
    vals = sorted(values)
    index = (len(vals) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return vals[lo] + (vals[hi] - vals[lo]) * (index - lo)


def summarize(rows, args):
    ok = [r for r in rows if r['ok']]
    ttfts = [r['ttft_s'] for r in ok]
    speeds = [r['decode_tps'] for r in ok]
    enough = len(ok) >= args.min_completed
    p90 = percentile(ttfts, .9)
    mean = statistics.mean(speeds) if speeds else None
    if len(ok) != len(rows):
        status = 'ERROR'  # Never hide failed requests by removing them from percentiles.
    elif not enough:
        status = 'INSUFFICIENT'
    elif p90 < args.ttft_slo and mean > args.tps_slo:
        status = 'PASS'
    else:
        status = 'FAIL'
    return dict(status=status, completed=len(ok), sent=len(rows),
                errors=len(rows)-len(ok), p90_ttft_s=p90, mean_decode_tps=mean,
                p90_e2e_s=percentile([r['e2e_s'] for r in ok], .9),
                mean_e2e_tps=statistics.mean([r['e2e_tps'] for r in ok]) if ok else None,
                mean_cached_tokens=statistics.mean([r.get('cached_tokens', 0) for r in ok]) if ok else None,
                coalesced_first_chunks=sum(r.get('first_chunk_tokens', 0) > 1 for r in ok),
                error_examples=[r.get('error') for r in rows if not r['ok']][:3])


async def request(session, args, prompt, rid, max_tokens=None,
                  admission_deadline=None, benchmark_start=None):
    length = args.output_len if max_tokens is None else max_tokens
    start = time.perf_counter()
    record = dict(rid=rid, ok=False, ttft_s=None, e2e_s=None,
                  decode_tps=None, completion_tokens=0, started_unix=time.time())
    first = None
    tokens = 0
    done = False
    finish_type = None
    try:
        payload = dict(text=prompt, rid=rid, stream=True,
                       sampling_params=dict(max_new_tokens=length, temperature=0,
                                            ignore_eos=True, stop=[], stop_token_ids=[]))
        import aiohttp
        timeout = aiohttp.ClientTimeout(total=args.request_timeout)
        admitted_at = time.perf_counter()
        if admission_deadline is not None and admitted_at >= admission_deadline:
            return None  # Not sent; do not record this as a failed request.
        if benchmark_start is not None:
            record['admitted_elapsed_s'] = admitted_at - benchmark_start
        async with session.post(args.base_url+'/generate', json=payload, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f'HTTP {response.status}: {(await response.text())[:300]}')
            async for raw in response.content:
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    done = True
                    break
                event = json.loads(data)
                if 'error' in event:
                    raise RuntimeError(str(event['error'])[:300])
                meta = event.get('meta_info', {})
                count = meta.get('completion_tokens')
                if count is None:
                    continue
                count = int(count)
                if count < tokens:
                    raise RuntimeError('completion_tokens counter decreased')
                tokens = count
                if first is None and tokens > 0:
                    first = time.perf_counter()
                    record['first_chunk_tokens'] = tokens
                for key in ('prompt_tokens', 'cached_tokens'):
                    if key in meta:
                        record[key] = meta[key]
                reason = meta.get('finish_reason')
                if reason:
                    finish_type = reason.get('type') if isinstance(reason, dict) else str(reason)
        end = time.perf_counter()
        if not done and finish_type is None:
            raise RuntimeError('Stream ended without DONE or finish_reason')
        if finish_type in ('abort', 'error'):
            raise RuntimeError(f'Server finish_reason={finish_type}')
        if tokens != length:
            raise RuntimeError(f'Expected fixed output {length}; got {tokens}')
        if first is None or tokens <= 1 or end <= first:
            raise RuntimeError('Not enough streamed tokens to calculate decode speed')
        record.update(ok=True, ttft_s=first-start, e2e_s=end-start,
                      decode_tps=(tokens-1)/(end-first),
                      e2e_tps=tokens/(end-start), completion_tokens=tokens)
    except Exception as exc:
        record.update(error=f'{type(exc).__name__}: {exc}',
                      completion_tokens=tokens, e2e_s=time.perf_counter()-start)
    return record


async def flush_cache(session, args):
    import aiohttp

    # Wait for full scheduler idleness, including outstanding async operations.
    # The HTTP deadline must exceed the server's wait; the session default is 30s.
    server_wait_seconds = 60
    http_timeout = aiohttp.ClientTimeout(total=server_wait_seconds + 15)
    url = args.base_url + f'/flush_cache?timeout={server_wait_seconds}'
    # Older SGLang releases use GET, newer releases also support POST.
    async with session.post(url, timeout=http_timeout) as response:
        code, text = response.status, await response.text()
    if code == 405:
        async with session.get(url, timeout=http_timeout) as response:
            code, text = response.status, await response.text()
    if code != 200:
        raise RuntimeError(f'Cache flush failed: HTTP {code}: {text[:300]}')
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    if value is False or (isinstance(value, dict) and value.get('success') is False):
        raise RuntimeError(f'Cache flush rejected: {text[:300]}')


async def confirmation_stage(session, args, prompts, c, kind, raw_file, run_id,
                             start, admission_end, request_limit, benchmark_start):
    """Score a fixed cohort while same-workload background traffic fills slots."""
    order = list(range(len(prompts)))
    random.Random(args.seed).shuffle(order)
    rows = []
    measured_assigned = measured_finished = warmup_completed = background_assigned = 0
    measured_started_at = measured_finished_at = error_at = None
    print(f'[{kind}] concurrency={c}, measured target={request_limit}, '
          f'submit limit={admission_end-start:.1f}s, background refill enabled', flush=True)

    async def worker(worker_id):
        nonlocal measured_assigned, measured_finished, warmup_completed, background_assigned
        nonlocal measured_started_at, measured_finished_at, error_at
        first_request = True
        while (time.perf_counter() < admission_end and error_at is None
               and measured_finished_at is None):
            # No await between cohort assignment and request initiation. Early
            # warmup finishers refill too, instead of waiting at an idle barrier.
            if first_request:
                role, index, source_index = 'warmup', worker_id, worker_id
                first_request = False
            elif warmup_completed == c and measured_assigned < request_limit:
                role = 'measured'
                index = source_index = measured_assigned
                measured_assigned += 1
                if measured_started_at is None:
                    measured_started_at = time.perf_counter()
            else:
                role, index = 'background', background_assigned
                source_index = request_limit + background_assigned
                background_assigned += 1
            sample = order[source_index % len(order)]
            rid = f'{run_id}-{kind}-{c}-{role}-{index}'
            prompt = prompts[sample]
            if not args.allow_prefix_reuse:
                salt = hashlib.sha256(rid.encode()).hexdigest()
                prompt = f'<!-- benchmark-request {salt} -->\n' + prompt
            row = await request(session, args, prompt, rid,
                                admission_deadline=admission_end,
                                benchmark_start=benchmark_start)
            if row is None:
                break
            finished_at = time.perf_counter()
            row.update(kind=kind, concurrency=c, sample_index=sample,
                       measurement_role=role, request_index=index,
                       stage_elapsed_s=finished_at-start)
            rows.append(row)
            if not row['ok'] and error_at is None:
                error_at = finished_at
            if role == 'warmup' and row['ok']:
                warmup_completed += 1
            elif role == 'measured':
                measured_finished += 1
                if measured_finished == request_limit:
                    measured_finished_at = finished_at
            raw_file.write(json.dumps(row, ensure_ascii=False)+'\n')
            raw_file.flush()

    # Once the cohort is done (or the deadline/error occurs), no worker refills.
    # Pending background and measured requests finish normally and remain logged.
    await asyncio.gather(*(worker(index) for index in range(c)))
    finished_at = time.perf_counter()
    stopped_at = min(value for value in
                     (admission_end, finished_at, measured_finished_at, error_at)
                     if value is not None)
    measured = [row for row in rows if row['measurement_role'] == 'measured']
    warmup = [row for row in rows if row['measurement_role'] == 'warmup']
    background = [row for row in rows if row['measurement_role'] == 'background']
    errors = [row for row in rows if not row['ok']]
    result = summarize(measured, args)
    target_met = result['completed'] == request_limit
    steady_state_valid = (target_met and warmup_completed == c and not errors
                          and measured_finished_at is not None
                          and measured_finished_at < admission_end)
    if errors:
        result['status'] = 'ERROR'
        result['error_examples'] = [row.get('error') for row in errors][:3]
    elif not steady_state_valid:
        result['status'] = 'INSUFFICIENT'
        result['insufficient_reason'] = 'cohort_not_completed_under_load_before_deadline'
    result.update(
        kind=kind, concurrency=c, target_requests=request_limit,
        request_target_met=target_met, steady_state_valid=steady_state_valid,
        measured_assigned=measured_assigned,
        warmup_sent=len(warmup), warmup_completed=warmup_completed,
        background_sent=len(background), total_sent=len(rows),
        load_errors=sum(not row['ok'] for row in warmup + background),
        total_errors=len(errors),
        stop_reason=('request_error' if errors else 'request_target_reached'
                     if steady_state_valid else 'admission_deadline'),
        measurement_start_seconds=(measured_started_at-start
                                   if measured_started_at is not None else None),
        measurement_end_seconds=(measured_finished_at-start
                                 if measured_finished_at is not None else None),
        admission_seconds=stopped_at-start,
        admission_limit_seconds=admission_end-start,
        wall_seconds=finished_at-start,
        drain_seconds=max(0.0, finished_at-stopped_at),
        last_request_started_elapsed_s=max(
            (row.get('admitted_elapsed_s', 0.0) for row in rows), default=None))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


async def stage(session, args, prompts, c, seconds, kind, raw_file, run_id, deadline,
                benchmark_start=None, request_limit=None):
    if time.perf_counter() >= deadline:
        return None
    if not args.keep_cache:
        await flush_cache(session, args)
    start = time.perf_counter()
    admission_end = min(start + seconds, deadline)
    if admission_end <= start:
        return None
    if request_limit is not None:
        return await confirmation_stage(
            session, args, prompts, c, kind, raw_file, run_id, start,
            admission_end, request_limit, benchmark_start)
    rows = []
    counter = 0
    # Same shuffled order per stage across all server configurations.
    order = list(range(len(prompts)))
    random.Random(args.seed).shuffle(order)
    print(f'[{kind}] concurrency={c}, submit window={admission_end-start:.1f}s', flush=True)

    async def worker():
        nonlocal counter
        while time.perf_counter() < admission_end:
            seq = counter
            counter += 1
            sample = order[seq % len(order)]
            rid = f'{run_id}-{kind}-{c}-{seq}'
            prompt = prompts[sample]
            if not args.allow_prefix_reuse:
                # Distinct FIRST tokens prevent repeated full-context cache hits.
                salt = hashlib.sha256(rid.encode()).hexdigest()
                prompt = f'<!-- benchmark-request {salt} -->\n' + prompt
            row = await request(session, args, prompt, rid,
                                admission_deadline=admission_end,
                                benchmark_start=benchmark_start)
            if row is None:
                break
            row.update(kind=kind, concurrency=c, sample_index=sample,
                       measurement_role='probe', request_index=seq,
                       stage_elapsed_s=time.perf_counter()-start)
            rows.append(row)
            raw_file.write(json.dumps(row, ensure_ascii=False)+'\n')
            raw_file.flush()
            if not row['ok']:
                # Do not repeatedly hammer an invalid/failed endpoint.
                return

    # Time-based probes stop at their deadline and drain admitted requests.
    await asyncio.gather(*(worker() for _ in range(c)))
    finished_at = time.perf_counter()
    stopped_at = min(admission_end, finished_at)
    result = summarize(rows, args)
    stop_reason = 'request_error' if result['errors'] else 'admission_deadline'
    result.update(kind=kind, concurrency=c,
                  target_requests=None, request_target_met=None,
                  steady_state_valid=None,
                  stop_reason=stop_reason,
                  admission_seconds=stopped_at-start,
                  admission_limit_seconds=admission_end-start,
                  wall_seconds=finished_at-start,
                  drain_seconds=max(0.0, finished_at-stopped_at),
                  last_request_started_elapsed_s=max(
                      (r.get('admitted_elapsed_s', 0.0) for r in rows), default=None))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def load_local_split(dataset, split):
    """Read saved datasets or local data files without resolving a hub dataset."""
    from datasets import Dataset, DatasetDict, load_from_disk

    path = Path(dataset).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f'Local dataset does not exist: {path}')
    if path.is_dir() and any((path / name).is_file() for name in
                             ('dataset_dict.json', 'state.json')):
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f'Split {split!r} not found; available: {list(ds)}')
            ds = ds[split]
        return ds, dict(format='save_to_disk', path=str(path), split=split)

    formats = {'.parquet': 'parquet', '.json': 'json', '.jsonl': 'json'}
    if path.is_file():
        files = [path]
    else:
        # Downloaded dataset repositories commonly use data/test-*.parquet.
        # Never fall back to all shards: that could mix test, train, and dev.
        files = []
        for root in (path, path / 'data'):
            for suffix in formats:
                files.extend(root.glob(f'{split}{suffix}'))
                files.extend(root.glob(f'{split}-*{suffix}'))
                files.extend(root.glob(f'{split}_*{suffix}'))
                files.extend((root / split).glob(f'*{suffix}'))
        files = sorted({file.resolve() for file in files if file.is_file()})
    if not files:
        raise ValueError(
            f'No local {split!r} Parquet/JSON/JSONL files in {path}; '
            'pass a dataset repository directory, a saved dataset, or a data file.'
        )
    kinds = {formats.get(file.suffix.lower()) for file in files}
    if None in kinds or len(kinds) != 1:
        raise ValueError('Use one local data format: Parquet or JSON/JSONL.')
    kind = kinds.pop()
    filenames = [str(file) for file in files]
    ds = (Dataset.from_parquet(filenames) if kind == 'parquet'
          else Dataset.from_json(filenames))
    return ds, dict(format=kind, path=str(path), split=split, files=filenames)


def load_prompts(args):
    ds, source = load_local_split(args.dataset, args.split)
    dataset_kind = getattr(args, 'dataset_kind', 'longswebench')
    issue_only = getattr(args, 'issue_only', False)
    if issue_only and (dataset_kind != 'swebench' or args.prompt_column):
        raise ValueError('--issue-only requires SWE-Bench and no --prompt-column')
    column = 'problem_statement' if issue_only else args.prompt_column
    if column is None:
        present = [key for key in ('text', 'prompt') if key in ds.column_names]
        if len(present) != 1:
            if dataset_kind == 'swebench' and not present and 'problem_statement' in ds.column_names:
                raise ValueError(
                    'Raw SWE-Bench has issue text but no prepared code-context prompt. '
                    'For the approximately 40K workload, use a local '
                    'SWE-bench_bm25_40K dataset with a text column. '
                    'Use --issue-only explicitly for short issue-only testing; '
                    'this is not the 40K workload.'
                )
            raise ValueError(f'Specify --prompt-column; available fields: {ds.column_names}')
        column = present[0]
    if column not in ds.column_names:
        raise ValueError(f'Column {column!r} not found; available fields: {ds.column_names}')
    prompts = list(ds[column])
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError(f'{column} must contain nonempty strings')
    input_mode = 'issue_only' if column == 'problem_statement' else 'prepared_prompt'
    info = dict(dataset_kind=dataset_kind, input_mode=input_mode, source=source,
                samples=len(prompts), column=column, fields=ds.column_names,
                sha256=hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
                chat_template='none; dataset prompt sent verbatim, apart from optional unique prefix')
    if dataset_kind == 'swebench' and input_mode == 'issue_only':
        message = ('Issue-only SWE-Bench: no repository code context is included. '
                   'These prompts are not the approximately 40K BM25 workload.')
        info['warnings'] = [message]
        warnings.warn(message, UserWarning, stacklevel=2)
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True,
                                           trust_remote_code=args.trust_remote_code)
        counts = [len(tok.encode(p, add_special_tokens=True)) for p in prompts]
        info['base_input_tokens'] = dict(min=min(counts), mean=statistics.mean(counts),
                                        p90=percentile(counts, .9), max=max(counts))
        info['length_note'] = 'Source prompts are not truncated or padded. Dataset length labels use the source tokenizer; local tokenizer counts can differ. Unique prefixes add a few tokens; requests.jsonl records server prompt_tokens.'
    return prompts, info


async def main_async(args, prompts, data_info):
    import aiohttp
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    settings = vars(args).copy()
    settings['confirmation_mode'] = 'scored_cohort_with_background'
    (out/'config.json').write_text(json.dumps(dict(settings=settings, data=data_info), indent=2, ensure_ascii=False))
    headers = {}
    if os.environ.get('SGLANG_API_KEY'):
        headers['Authorization'] = 'Bearer '+os.environ['SGLANG_API_KEY']
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                    timeout=aiohttp.ClientTimeout(total=30),
                                    headers=headers, read_bufsize=4*1024*1024) as session:
        async with session.get(args.base_url+'/health') as response:
            if response.status != 200:
                raise RuntimeError(f'Server health failed: {response.status}')
        warm = await request(session, args, 'Write a short Python comment.', 'warmup-'+uuid.uuid4().hex, 8)
        if not warm['ok']:
            raise RuntimeError(f'Warmup failed: {warm}')
        start = time.perf_counter()
        deadline = start + args.budget_seconds
        results = []
        run_id = uuid.uuid4().hex
        low, high, candidate = 0, None, args.start_concurrency
        confirmed = None
        confirmation = None
        stop_reason = 'budget_exhausted'
        with (out/'requests.jsonl').open('w', buffering=1) as raw:
            while True:
                remaining = deadline-time.perf_counter()
                # Reserve a target confirmation window, not time for final draining.
                # Each probe drains before the next stage, within the send-window clock.
                if remaining < args.confirm_seconds + args.probe_seconds:
                    break
                r = await stage(session, args, prompts, candidate, args.probe_seconds,
                                'probe', raw, run_id, deadline, benchmark_start=start)
                if r is None:
                    break
                results.append(r)
                (out/'stages.json').write_text(json.dumps(results, indent=2))
                if r['status'] == 'ERROR':
                    stop_reason = 'invalid_probe_error'
                    break
                if r['status'] == 'INSUFFICIENT':
                    # Use the remaining window at this concurrency for more evidence.
                    stop_reason = 'probe_insufficient_samples'
                    break
                if r['status'] == 'PASS':
                    low = max(low, candidate)
                    if candidate == args.max_concurrency:
                        stop_reason = 'all_tested_up_to_cap_passed'
                        break
                else:
                    high = candidate if high is None else min(high, candidate)
                if high is not None:
                    if high-low <= 1:
                        stop_reason = 'adjacent_boundary_observed'
                        break
                    candidate = (low+high)//2
                else:
                    candidate = min(args.max_concurrency, candidate*2)
            if stop_reason != 'invalid_probe_error':
                remaining = deadline-time.perf_counter()
                if remaining > 0:
                    # Confirm a fixed workload, bounded by the remaining global
                    # send window. A partial workload cannot confirm a candidate.
                    final_c = low or candidate
                    final_kind = 'confirm' if low else 'continue'
                    r = await stage(session, args, prompts, final_c, remaining,
                                    final_kind, raw, run_id, deadline,
                                    benchmark_start=start,
                                    request_limit=args.confirm_requests)
                    if r:
                        confirmation = r
                        results.append(r)
                        (out/'stages.json').write_text(json.dumps(results, indent=2))
                        if (final_kind == 'confirm' and r['status'] == 'PASS'
                                and r['steady_state_valid']
                                and r['completed'] == args.confirm_requests):
                            confirmed = low
        passed = {r['concurrency'] for r in results if r['status'] == 'PASS'}
        failed = {r['concurrency'] for r in results if r['status'] == 'FAIL'}
        inconsistent = sorted(passed & failed)
        has_errors = any(r['status'] == 'ERROR' for r in results)
        if inconsistent:
            confirmed = None
            status = 'INCONSISTENT'
        elif has_errors:
            confirmed = None
            status = 'INCONCLUSIVE'
        elif confirmed is not None:
            status = 'CONFIRMED_CANDIDATE'
        elif passed:
            status = 'PASSING_CANDIDATE'
        else:
            status = 'INCONCLUSIVE'
        elapsed = time.perf_counter()-start
        last_starts = [r['last_request_started_elapsed_s'] for r in results
                       if r.get('last_request_started_elapsed_s') is not None]
        report = dict(label=args.label, best_probe_concurrency=low or None,
                      highest_passing_concurrency=max(passed) if passed else None,
                      confirmed_concurrency=confirmed,
                      smallest_observed_fail=min(failed) if failed else None,
                      inconsistent_concurrencies=inconsistent,
                      tested_upper_cap=args.max_concurrency,
                      status=status,
                      confirmation_target_requests=args.confirm_requests,
                      confirmation_mode='scored_cohort_with_background',
                      confirmation_completed=confirmation['completed'] if confirmation else None,
                      confirmation_status=confirmation['status'] if confirmation else 'NOT_RUN',
                      confirmation_steady_state_valid=(confirmation.get('steady_state_valid')
                                                       if confirmation else None),
                      confirmation_total_sent=confirmation.get('total_sent') if confirmation else None,
                      confirmation_background_sent=confirmation.get('background_sent') if confirmation else None,
                      stop_reason=stop_reason, budget_seconds=args.budget_seconds,
                      send_window_seconds=args.budget_seconds,
                      last_request_started_elapsed_s=max(last_starts) if last_starts else None,
                      final_drain_seconds=max(0.0, elapsed-args.budget_seconds),
                      elapsed_wall_seconds=elapsed,
                      note='Search probes are time-bounded. Confirmation scores a fixed cohort after concurrent warmup, with same-workload background refill until all scored requests finish or the global deadline. All traffic is logged; only scored requests enter SLO metrics. A cohort finishing after the deadline or any load error cannot confirm a candidate. Pending requests drain normally. Passing candidates are observations, not proof of the global maximum.',
                      stages=results)
        (out/'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print('\nSUMMARY\n'+json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def parse_args(argv=None, *, dataset_kind='longswebench'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-kind', choices=('longswebench', 'swebench'), default=dataset_kind)
    p.add_argument('--dataset', help='Local saved dataset, Parquet/JSON/JSONL file, or dataset repository directory')
    p.add_argument('--split', default='test')
    p.add_argument('--prompt-column')
    p.add_argument('--issue-only', action='store_true', help='Explicitly benchmark raw SWE-Bench problem_statement without code context; not a 40K workload')
    p.add_argument('--tokenizer', default='/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731')
    p.add_argument('--trust-remote-code', action='store_true', help='Allow custom tokenizer Python from the local model directory')
    p.add_argument('--base-url', default='http://127.0.0.1:32073')
    p.add_argument('--label', default='run')
    p.add_argument('--output-dir', required=True, help='Must not exist')
    p.add_argument('--output-len', type=int, required=True, help='Agree on this before comparing results; fixed output, ignore EOS')
    p.add_argument('--budget-seconds', type=float, default=1200, help='Maximum global send-window seconds after preparation and warmup; confirmation can finish early, final draining is extra')
    p.add_argument('--probe-seconds', type=float, default=180)
    p.add_argument('--confirm-seconds', type=float, default=300, help='Target time to reserve for confirmation; not a required minimum')
    p.add_argument('--confirm-requests', type=int, default=100, help='Scored requests in final confirmation; background refill maintains load until all finish or the global send deadline')
    p.add_argument('--request-timeout', type=float, default=300)
    p.add_argument('--start-concurrency', type=int, default=8)
    p.add_argument('--max-concurrency', type=int, default=64)
    p.add_argument('--min-completed', type=int, default=20)
    p.add_argument('--ttft-slo', type=float, default=10)
    p.add_argument('--tps-slo', type=float, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--allow-prefix-reuse', action='store_true', help='Replay unchanged prompts; possible full-prefix hits on repeats')
    p.add_argument('--keep-cache', action='store_true', help='Do not flush between stages; changes cache-state policy')
    p.add_argument('--inspect-only', action='store_true', help='Read local data/tokenizer and exit without HTTP or output files')
    a = p.parse_args(argv)
    if a.dataset is None:
        relative = ('swebench' if a.dataset_kind == 'swebench'
                    else 'longswebench/LongSWE_Bench/128K')
        a.dataset = str(Path.home() / 'datasets' / relative)
    if a.issue_only and (a.dataset_kind != 'swebench' or a.prompt_column):
        p.error('--issue-only requires SWE-Bench and no --prompt-column')
    a.base_url = a.base_url.rstrip('/')
    if not 1 <= a.start_concurrency <= a.max_concurrency:
        p.error('Require 1 <= start concurrency <= max concurrency')
    if a.output_len < 2 or a.min_completed < 1:
        p.error('Require output_len >= 2 and min_completed >= 1')
    if a.confirm_requests < a.min_completed:
        p.error('Require confirm-requests >= min-completed >= 1')
    for key in ('budget_seconds', 'probe_seconds', 'confirm_seconds', 'request_timeout', 'ttft_slo', 'tps_slo'):
        if getattr(a, key) <= 0:
            p.error(f'{key} must be positive')
    return a


def cli_main(argv=None, *, dataset_kind='longswebench'):
    args = parse_args(argv, dataset_kind=dataset_kind)
    prompts, data_info = load_prompts(args)
    print(json.dumps(data_info, indent=2, ensure_ascii=False), flush=True)
    if not args.inspect_only:
        asyncio.run(main_async(args, prompts, data_info))


if __name__ == '__main__':
    cli_main()



