"""Fixed-workload confirmation, shared admission cap, and deadline regressions."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

import longswe_slo as bench
import test_longswe_slo as fixture


class ConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_exactly_100_scored_requests_with_shared_slots_and_identical_source_order(self):
        orders = []
        for concurrency in (1, 7, 64, 128):
            with self.subTest(concurrency=concurrency):
                clock = fixture.Clock()
                calls = []
                active = maximum = 0

                async def generate(session, args, prompt, rid, **kwargs):
                    nonlocal active, maximum
                    calls.append(rid)
                    active += 1
                    maximum = max(maximum, active)
                    await asyncio.sleep(0)
                    active -= 1
                    result = fixture.row()
                    result['rid'] = rid
                    result['admitted_elapsed_s'] = clock.read()
                    return result

                raw = io.StringIO()
                with (
                    patch.object(bench.time, 'perf_counter', clock.read),
                    patch.object(bench, 'request', side_effect=generate),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = await bench.stage(
                        None, fixture.options(keep_cache=True), ['a', 'b', 'c'],
                        concurrency, 100, 'confirm', raw, 'run', 100,
                        request_limit=100,
                    )
                self.assertGreaterEqual(len(calls), 100 + concurrency)
                self.assertEqual(len(set(calls)), len(calls))
                self.assertEqual(maximum, concurrency)
                self.assertEqual(result['completed'], 100)
                self.assertTrue(result['request_target_met'])
                self.assertTrue(result['steady_state_valid'])
                self.assertEqual(result['status'], 'PASS')
                rows = sorted((row for line in raw.getvalue().splitlines()
                               if (row := json.loads(line))['measurement_role'] == 'measured'),
                              key=lambda row: int(row['rid'].rsplit('-', 1)[1]))
                self.assertEqual(len(rows), 100)
                orders.append([row['sample_index'] for row in rows])
        self.assertTrue(all(order == orders[0] for order in orders))

    async def timed_stage(self, target, deadline):
        clock = fixture.Clock()
        starts = []

        async def generate(session, args, prompt, rid, **kwargs):
            self.assertLess(clock.read(), kwargs['admission_deadline'])
            starts.append(clock.read())
            clock.advance(4)
            result = fixture.row()
            result['admitted_elapsed_s'] = starts[-1]
            return result

        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'request', side_effect=generate),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = await bench.stage(
                None, fixture.options(keep_cache=True), ['prompt'], 1, deadline,
                'confirm', io.StringIO(), 'run', deadline, request_limit=target,
            )
        return result, starts

    async def test_target_stops_after_cohort_completion_without_filling_time_cap(self):
        result, starts = await self.timed_stage(target=3, deadline=100)
        self.assertEqual(starts, [0, 4, 8, 12])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['stop_reason'], 'request_target_reached')
        self.assertEqual(result['wall_seconds'], 16)
        self.assertEqual(result['admission_seconds'], 16)
        self.assertEqual(result['admission_limit_seconds'], 100)
        self.assertEqual(result['drain_seconds'], 0)

    async def test_partial_target_is_insufficient_even_with_passing_metrics(self):
        result, starts = await self.timed_stage(target=100, deadline=10)
        self.assertEqual(starts, [0, 4, 8])
        self.assertEqual(result['completed'], 2)
        self.assertLess(result['p90_ttft_s'], 10)
        self.assertGreater(result['mean_decode_tps'], 15)
        self.assertEqual(result['status'], 'INSUFFICIENT')
        self.assertFalse(result['request_target_met'])
        self.assertEqual(result['stop_reason'], 'admission_deadline')
        self.assertEqual(result['drain_seconds'], 2)

    async def test_full_target_after_deadline_is_insufficient_despite_normal_drain(self):
        result, starts = await self.timed_stage(target=3, deadline=14)
        self.assertTrue(all(start < 14 for start in starts))
        self.assertEqual(result['wall_seconds'], 16)
        self.assertTrue(result['request_target_met'])
        self.assertFalse(result['steady_state_valid'])
        self.assertEqual(result['status'], 'INSUFFICIENT')

    async def test_failed_requests_are_not_replaced_to_reach_the_target(self):
        generate = AsyncMock(side_effect=[fixture.row(), dict(ok=False, error='server error')])
        with (
            patch.object(bench, 'request', generate),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = await bench.stage(
                None, fixture.options(keep_cache=True), ['prompt'], 1, 60,
                'confirm', io.StringIO(), 'run', time.perf_counter() + 60,
                request_limit=100,
            )
        self.assertEqual(generate.await_count, 2)  # One warmup, one failed scored request.
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['errors'], 1)
        self.assertFalse(result['request_target_met'])

    async def test_complete_target_still_requires_both_strict_slo_conditions(self):
        for changed in ({'ttft_s': 10}, {'decode_tps': 15}):
            with self.subTest(changed=changed):
                row = {**fixture.row(), **changed}
                with (
                    patch.object(bench, 'request', new=AsyncMock(side_effect=lambda *a, **kw: dict(row))),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    result = await bench.stage(
                        None, fixture.options(keep_cache=True), ['prompt'], 1, 60,
                        'confirm', io.StringIO(), 'run', time.perf_counter() + 60,
                        request_limit=7,
                    )
                self.assertEqual(result['completed'], 7)
                self.assertTrue(result['request_target_met'])
                self.assertEqual(result['status'], 'FAIL')

    async def run_main(self, complete):
        clock = fixture.Clock()
        limits = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None, request_limit=None):
            limits.append(request_limit)
            clock.advance(seconds if kind == 'probe' or not complete else 1)
            return dict(kind=kind, concurrency=c,
                        status='PASS' if kind == 'probe' or complete else 'INSUFFICIENT',
                        steady_state_valid=bool(request_limit and complete),
                        completed=(request_limit if complete else 30) if request_limit else 20,
                        last_request_started_elapsed_s=clock.read() - .1)

        with tempfile.TemporaryDirectory() as tmp:
            args = fixture.options(str(Path(tmp) / 'run'), max_concurrency=8)
            with (
                patch.object(bench.time, 'perf_counter', clock.read),
                patch.object(bench, 'stage', side_effect=stage),
                patch.object(bench, 'request', new=AsyncMock(return_value=fixture.row())),
                patch.object(aiohttp, 'ClientSession', return_value=fixture.Session()),
                patch.object(aiohttp, 'TCPConnector', return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                await bench.main_async(args, ['prompt'], {})
            report = json.loads((Path(args.output_dir) / 'summary.json').read_text())
        return report, limits

    async def test_incomplete_confirmation_does_not_confirm_a_passing_probe(self):
        report, limits = await self.run_main(complete=False)
        self.assertEqual(limits, [None, 100])
        self.assertEqual(report['highest_passing_concurrency'], 8)
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['status'], 'PASSING_CANDIDATE')
        self.assertEqual(report['confirmation_status'], 'INSUFFICIENT')
        self.assertEqual(report['confirmation_completed'], 30)
        self.assertEqual(report['confirmation_target_requests'], 100)

    async def test_complete_confirmation_does_not_fill_the_remaining_time(self):
        report, limits = await self.run_main(complete=True)
        self.assertEqual(limits, [None, 100])
        self.assertEqual(report['confirmed_concurrency'], 8)
        self.assertEqual(report['confirmation_completed'], 100)
        self.assertEqual(report['confirmation_status'], 'PASS')
        self.assertLess(report['elapsed_wall_seconds'], report['send_window_seconds'])

    async def test_actual_http_streams_score_100_and_log_all_support_traffic(self):
        received = []

        async def generate(request):
            payload = await request.json()
            received.append(payload['rid'])
            self.assertEqual(payload['sampling_params']['max_new_tokens'], 8)
            response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
            await response.prepare(request)
            await response.write(b'data: {"meta_info":{"completion_tokens":1}}\n\n')
            await asyncio.sleep(.001)
            await response.write(b'data: {"meta_info":{"completion_tokens":8}}\n\n')
            await response.write(b'data: [DONE]\n\n')
            return response

        app = web.Application()
        app.router.add_post('/generate', generate)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            args = fixture.options(base_url=str(server.make_url('/')).rstrip('/'), keep_cache=True)
            with contextlib.redirect_stdout(io.StringIO()):
                result = await bench.stage(
                    session, args, ['prompt'], 7, 60, 'confirm', io.StringIO(),
                    'http', time.perf_counter() + 60, request_limit=100,
                )
        self.assertGreaterEqual(len(received), 107)
        self.assertEqual(len(set(received)), len(received))
        self.assertEqual(sum('-measured-' in rid for rid in received), 100)
        self.assertEqual(result['total_sent'], len(received))
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['completed'], 100)


class ConfirmationOptionsTests(unittest.TestCase):
    def test_defaults_are_100_requests_and_20_minutes_for_both_entry_points(self):
        for kind in ('longswebench', 'swebench'):
            args = bench.parse_args(['--output-dir', 'unused', '--output-len', '1024'],
                                    dataset_kind=kind)
            self.assertEqual(args.confirm_requests, 100)
            self.assertEqual(args.budget_seconds, 1200)

    def test_confirmation_target_cannot_be_below_the_sample_gate(self):
        for target in ('0', '-1', '19'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bench.parse_args(['--output-dir', 'unused', '--output-len', '1024',
                                  '--confirm-requests', target])


if __name__ == '__main__':
    unittest.main()

