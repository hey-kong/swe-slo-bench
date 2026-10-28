"""Full-workload warmup, cache boundaries, logging, and admission-budget checks."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

import longswe_slo as bench
import test_longswe_slo as fixture


class WorkloadWarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_full_wave_uses_seeded_prompts_and_drains_without_refilling(self):
        calls = []
        active = maximum = 0
        clock = fixture.Clock()
        args = fixture.options(output_len=1024, workload_warmup=True)

        async def generate(session, args, prompt, rid, **kwargs):
            nonlocal active, maximum
            self.assertNotIn('max_tokens', kwargs)
            self.assertEqual(args.output_len, 1024)
            self.assertLess(clock.read(), kwargs['admission_deadline'])
            calls.append((prompt, rid))
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0)
            active -= 1
            # A slow warmup does not trigger retries or the measured SLO gate.
            return {**fixture.row(), 'rid': rid, 'ttft_s': 145,
                    'completion_tokens': 1024, 'admitted_elapsed_s': 0}

        raw = io.StringIO()
        flush = AsyncMock()
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'request', side_effect=generate),
            patch.object(bench, 'flush_cache', new=flush),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = await bench.warmup_workload(
                None, args, ['alpha', 'beta', 'gamma'], 8, 'probe', raw,
                'run', 100, 0)
        self.assertEqual(len(calls), 8)
        self.assertEqual(maximum, 8)
        self.assertEqual(active, 0)
        self.assertEqual(result['status'], 'COMPLETE')
        self.assertEqual(result['mean_ttft_s'], 145)
        self.assertEqual(result['p90_ttft_s'], 145)
        self.assertEqual(result['completed'], 8)
        flush.assert_awaited_once()
        rows = [json.loads(line) for line in raw.getvalue().splitlines()]
        expected = list(range(3))
        random.Random(args.seed).shuffle(expected)
        self.assertEqual([r['sample_index'] for r in rows],
                         [expected[i % 3] for i in range(8)])
        self.assertTrue(all(r['kind'] == 'warmup' for r in rows))
        self.assertTrue(all(r['for_stage'] == 'probe' for r in rows))
        self.assertEqual(len({prompt.splitlines()[0] for prompt, _ in calls}), 8)

    async def test_expired_or_flush_exhausted_warmup_never_sends(self):
        for initial in (0, 10):
            with self.subTest(initial=initial):
                clock = fixture.Clock()
                clock.now = initial
                generate = AsyncMock()

                async def flush(session, args):
                    clock.advance(11)

                with (
                    patch.object(bench.time, 'perf_counter', clock.read),
                    patch.object(bench, 'request', new=generate),
                    patch.object(bench, 'flush_cache', side_effect=flush),
                ):
                    result = await bench.warmup_workload(
                        None, fixture.options(), ['prompt'], 8, 'probe',
                        io.StringIO(), 'expired', 10, 0)
                self.assertIsNone(result)
                generate.assert_not_awaited()

    async def run_main(self, generate, *, budget=10, concurrency=8,
                       stage_impl=None, clock=None):
        clock = clock or fixture.Clock()
        stage_mock = AsyncMock(side_effect=stage_impl)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            args = fixture.options(str(out), budget_seconds=budget,
                                   start_concurrency=concurrency,
                                   max_concurrency=concurrency,
                                   output_len=1024, workload_warmup=True)
            with (
                patch.object(bench.time, 'perf_counter', clock.read),
                patch.object(bench, 'request', side_effect=generate),
                patch.object(bench, 'stage', new=stage_mock),
                patch.object(bench, 'flush_cache', new=AsyncMock()),
                patch.object(aiohttp, 'ClientSession', return_value=fixture.Session()),
                patch.object(aiohttp, 'TCPConnector', return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await bench.main_async(args, ['long prompt'], {})
            report = json.loads((out / 'summary.json').read_text())
            raw = [json.loads(line) for line in
                   (out / 'warmup_requests.jsonl').read_text().splitlines()]
        return report, raw, stage_mock

    async def test_warmup_errors_abort_before_search_and_are_logged(self):
        async def generate(session, args, prompt, rid, **kwargs):
            await asyncio.sleep(0)
            if rid.endswith('-0'):
                return dict(ok=False, rid=rid, error='initialization failed')
            return {**fixture.row(), 'rid': rid}

        report, rows, measured = await self.run_main(generate)
        measured.assert_not_awaited()
        self.assertEqual(len(rows), 8)
        self.assertEqual(report['warmup_status'], 'ERROR')
        self.assertEqual(report['stop_reason'], 'invalid_warmup_error')
        self.assertEqual(report['status'], 'INCONCLUSIVE')
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['warmup_sent'], 8)
        self.assertEqual(report['warmup_completed'], 7)
        self.assertEqual(report['warmup_stages'][0]['errors'], 1)

    async def test_warmup_consumes_budget_and_late_completion_cannot_start_measurement(self):
        clock = fixture.Clock()
        released = asyncio.Event()
        started = []

        async def generate(session, args, prompt, rid, **kwargs):
            started.append(clock.read())
            if len(started) == 8:
                clock.advance(11)
                released.set()
            await released.wait()
            return {**fixture.row(), 'rid': rid, 'admitted_elapsed_s': started[-1]}

        report, rows, measured = await self.run_main(generate, clock=clock)
        measured.assert_not_awaited()
        self.assertEqual(started, [0] * 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(report['warmup_status'], 'INSUFFICIENT')
        self.assertEqual(report['stop_reason'], 'warmup_budget_exhausted')
        self.assertEqual(report['elapsed_wall_seconds'], 11)
        self.assertEqual(report['final_drain_seconds'], 1)
        self.assertEqual(report['confirmation_status'], 'NOT_RUN')
        self.assertFalse(report['warmup_stages'][0]['finished_before_deadline'])

    async def test_every_candidate_is_warmed_and_all_slots_finish_before_scoring(self):
        clock = fixture.Clock()
        warmup_finished = 0
        warmup_counts = []
        measured_counts = []

        async def generate(session, args, prompt, rid, **kwargs):
            nonlocal warmup_finished
            await asyncio.sleep(0)
            warmup_finished += 1
            return {**fixture.row(), 'rid': rid, 'ttft_s': 145}

        async def measured(session, args, prompts, c, seconds, kind, raw, run_id,
                           deadline, benchmark_start=None, request_limit=None):
            warmup_counts.append(warmup_finished)
            measured_counts.append(c)
            clock.advance(1)
            return dict(kind=kind, concurrency=c, status='PASS',
                        completed=request_limit or 20,
                        last_request_started_elapsed_s=clock.read()-.1)

        report, rows, _ = await self.run_main(
            generate, stage_impl=measured, clock=clock)
        self.assertEqual(warmup_counts, [8, 16])
        self.assertEqual(measured_counts, [8, 8])
        self.assertEqual(len(rows), 16)
        self.assertEqual(report['confirmed_concurrency'], 8)
        self.assertEqual(report['confirmation_completed'], 100)
        self.assertEqual(report['warmup_completed'], 16)
        self.assertEqual(report['warmup_status'], 'COMPLETE')
        self.assertEqual([s['for_stage'] for s in report['warmup_stages']],
                         ['probe', 'confirm'])

    async def test_real_http_preserves_cache_boundaries_and_exact_100_confirmation(self):
        received = []
        boundaries = []

        async def health(request):
            return web.Response(text='ok')

        async def flush(request):
            boundaries.append(len(received))
            return web.Response(text='Cache flushed.')

        async def generate(request):
            payload = await request.json()
            received.append(payload)
            self.assertEqual(payload['sampling_params']['max_new_tokens'], 1024)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: {"meta_info":{"completion_tokens":1}}\n\n')
            await asyncio.sleep(.002)
            await response.write(b'data: {"meta_info":{"completion_tokens":1024}}\n\n')
            await response.write(b'data: [DONE]\n\n')
            return response

        app = web.Application()
        app.router.add_get('/health', health)
        app.router.add_post('/flush_cache', flush)
        app.router.add_post('/generate', generate)
        async with TestServer(app) as server:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / 'run'
                args = fixture.options(
                    str(out), base_url=str(server.make_url('/')).rstrip('/'),
                    output_len=1024, workload_warmup=True, start_concurrency=3,
                    max_concurrency=3, probe_seconds=.03, confirm_seconds=.01,
                    budget_seconds=10)
                with contextlib.redirect_stdout(io.StringIO()):
                    await bench.main_async(args, ['real context alpha', 'real context beta'], {})
                report = json.loads((out / 'summary.json').read_text())
                measured = [json.loads(line) for line in
                            (out / 'requests.jsonl').read_text().splitlines()]
                warmed = [json.loads(line) for line in
                          (out / 'warmup_requests.jsonl').read_text().splitlines()]
                config = json.loads((out / 'config.json').read_text())
        scored = [r for r in measured if r['kind'] == 'confirm']
        self.assertEqual(len(scored), 100)
        self.assertEqual(len(warmed), 6)
        self.assertEqual(len(received), len(measured) + len(warmed))
        self.assertEqual(report['confirmation_completed'], 100)
        self.assertEqual(report['confirmed_concurrency'], 3)
        self.assertEqual(len(boundaries), 4)
        self.assertEqual(boundaries[:2], [0, 3])
        self.assertEqual(boundaries[3] - boundaries[2], 3)
        self.assertTrue(config['settings']['workload_warmup'])
        self.assertTrue(all(r['kind'] == 'warmup' for r in warmed))
        self.assertTrue(all('warmup' not in r['rid'] for r in measured))
        by_rid = {p['rid']: p for p in received}
        last_warm = next(r for r in warmed if r['for_stage'] == 'confirm'
                         and r['warmup_request_index'] == 0)
        first_score = next(r for r in scored if r['rid'].endswith('-0'))
        wp = by_rid[last_warm['rid']]['text'].split('\n', 1)
        sp = by_rid[first_score['rid']]['text'].split('\n', 1)
        self.assertEqual(wp[1], sp[1])
        self.assertNotEqual(wp[0], sp[0])
        self.assertAlmostEqual(report['stages'][-1]['mean_ttft_s'],
                               sum(r['ttft_s'] for r in scored) / 100)

    async def test_binary_search_warms_each_changed_concurrency_before_scoring(self):
        clock = fixture.Clock()
        observed = []
        outcomes = {8: 'FAIL', 4: 'PASS', 6: 'PASS', 7: 'FAIL'}

        async def measured(session, args, prompts, c, seconds, kind, raw, run_id,
                           deadline, benchmark_start=None, request_limit=None):
            observed.append((kind, c))
            clock.advance(seconds)
            return dict(kind=kind, concurrency=c, status=outcomes[c],
                        completed=request_limit or 20,
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            args = fixture.options(str(out), workload_warmup=True,
                                   budget_seconds=100, probe_seconds=10,
                                   confirm_seconds=10)
            with (
                patch.object(bench.time, 'perf_counter', clock.read),
                patch.object(bench, 'request', new=AsyncMock(
                    side_effect=lambda *a, **kw: fixture.row())),
                patch.object(bench, 'stage', side_effect=measured),
                patch.object(bench, 'flush_cache', new=AsyncMock()),
                patch.object(aiohttp, 'ClientSession', return_value=fixture.Session()),
                patch.object(aiohttp, 'TCPConnector', return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await bench.main_async(args, ['prompt'], {})
            report = json.loads((out / 'summary.json').read_text())
        self.assertEqual(observed, [('probe', 8), ('probe', 4), ('probe', 6),
                                    ('probe', 7), ('confirm', 6)])
        self.assertEqual([(w['for_stage'], w['concurrency'])
                          for w in report['warmup_stages']], observed)
        self.assertEqual([w['completed'] for w in report['warmup_stages']],
                         [8, 4, 6, 7, 6])
        self.assertEqual(report['confirmed_concurrency'], 6)

    async def test_confirmation_warmup_failure_invalidates_a_passing_probe(self):
        async def generate(session, args, prompt, rid, **kwargs):
            if '-warmup-confirm-' in rid and rid.endswith('-0'):
                return dict(ok=False, rid=rid, error='server stopped')
            return {**fixture.row(), 'rid': rid}

        async def measured(session, args, prompts, c, seconds, kind, raw, run_id,
                           deadline, benchmark_start=None, request_limit=None):
            return dict(kind=kind, concurrency=c, status='PASS', completed=20,
                        last_request_started_elapsed_s=0)

        report, _, measured_mock = await self.run_main(generate, stage_impl=measured)
        measured_mock.assert_awaited_once()
        self.assertEqual(report['highest_passing_concurrency'], 8)
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['confirmation_status'], 'NOT_RUN')
        self.assertEqual(report['status'], 'INCONCLUSIVE')
        self.assertEqual(report['warmup_status'], 'ERROR')

    async def test_measured_first_wave_outliers_are_kept_after_warmup(self):
        clock = fixture.Clock()
        calls = []

        async def generate(session, args, prompt, rid, **kwargs):
            calls.append(rid)
            await asyncio.sleep(0)
            index = int(rid.rsplit('-', 1)[1])
            ttft = 200 if '-warmup-' in rid else 145 if index < 8 else 8.45
            return {**fixture.row(), 'rid': rid, 'ttft_s': ttft}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            # Skip time probes and exercise the real fixed-count stage.
            args = fixture.options(str(out), workload_warmup=True, budget_seconds=1)
            with (
                patch.object(bench.time, 'perf_counter', clock.read),
                patch.object(bench, 'request', side_effect=generate),
                patch.object(bench, 'flush_cache', new=AsyncMock()),
                patch.object(aiohttp, 'ClientSession', return_value=fixture.Session()),
                patch.object(aiohttp, 'TCPConnector', return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await bench.main_async(args, ['long context'], {})
            report = json.loads((out / 'summary.json').read_text())
            measured = [json.loads(line) for line in
                        (out / 'requests.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 108)
        self.assertEqual(len(measured), 100)
        self.assertEqual(sum(r['ttft_s'] == 145 for r in measured), 8)
        self.assertEqual(report['warmup_stages'][0]['mean_ttft_s'], 200)
        self.assertAlmostEqual(report['stages'][0]['mean_ttft_s'],
                               (8 * 145 + 92 * 8.45) / 100)


class WarmupOptionsTests(unittest.TestCase):
    def test_defaults_and_cold_start_opt_out_for_both_entry_points(self):
        for kind in ('longswebench', 'swebench'):
            argv = ['--output-dir', 'unused', '--output-len', '1024']
            args = bench.parse_args(argv, dataset_kind=kind)
            self.assertTrue(args.workload_warmup)
            self.assertEqual(args.budget_seconds, 1200)
            self.assertEqual(args.confirm_requests, 100)
            args = bench.parse_args(argv + ['--no-workload-warmup'], dataset_kind=kind)
            self.assertFalse(args.workload_warmup)


if __name__ == '__main__':
    unittest.main()
