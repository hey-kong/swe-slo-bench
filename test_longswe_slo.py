"""Timing regressions. No GPU, model, dataset, or running server required."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
import longswe_slo as bench


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class HealthResponse:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def get(self, url):
        return HealthResponse()


def options(output_dir='unused', **overrides):
    values = dict(
        output_dir=output_dir, base_url='http://unused', output_len=8,
        request_timeout=300, min_completed=1, ttft_slo=10, tps_slo=15,
        budget_seconds=10, probe_seconds=4, confirm_seconds=3,
        start_concurrency=8, max_concurrency=64, label='test', confirm_requests=100,
        keep_cache=False, seed=42, allow_prefix_reuse=False, workload_warmup=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def row():
    return dict(ok=True, ttft_s=1.0, e2e_s=4.0, decode_tps=20.0,
                e2e_tps=2.0, completion_tokens=8, first_chunk_tokens=1)


class TimingTests(unittest.IsolatedAsyncioTestCase):
    async def run_main(self, args, clock, stage_impl):
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'stage', side_effect=stage_impl),
            patch.object(bench, 'request', new=AsyncMock(return_value=row())),
            patch.object(aiohttp, 'ClientSession', return_value=Session()),
            patch.object(aiohttp, 'TCPConnector', return_value=None),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            await bench.main_async(args, ['prompt'], {})
        return json.loads((Path(args.output_dir) / 'summary.json').read_text())

    async def test_stage_sends_until_deadline_then_drains_all_requests(self):
        clock = Clock()
        args = options()
        starts = []

        async def generate(session, args, prompt, rid, **kwargs):
            if clock.read() >= kwargs['admission_deadline']:
                return None
            admitted = clock.read()
            starts.append(admitted)
            clock.advance(4)
            result = row()
            result['admitted_elapsed_s'] = admitted
            return result

        raw = io.StringIO()
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'request', side_effect=generate),
            patch.object(bench, 'flush_cache', new=AsyncMock()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = await bench.stage(
                None, args, ['prompt'], 1, 10, 'confirm', raw, 'test', 10,
                benchmark_start=0,
            )
        self.assertEqual(starts, [0, 4, 8])
        self.assertEqual(result['completed'], 3)
        self.assertEqual(result['drain_seconds'], 2)
        self.assertEqual(result['wall_seconds'], 12)
        self.assertEqual(len(raw.getvalue().splitlines()), 3)

    async def test_request_guard_never_posts_at_or_after_deadline(self):
        clock = Clock()
        clock.now = 10
        session = Mock()
        session.post.side_effect = AssertionError('generation admitted too late')
        with patch.object(bench.time, 'perf_counter', clock.read):
            result = await bench.request(
                session, options(), 'prompt', 'expired', admission_deadline=10,
            )
        self.assertIsNone(result)
        session.post.assert_not_called()

    async def test_expired_stage_does_not_even_flush(self):
        clock = Clock()
        clock.now = 10
        flush = AsyncMock()
        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'flush_cache', new=flush),
        ):
            result = await bench.stage(
                None, options(), ['prompt'], 1, 10, 'probe', io.StringIO(),
                'expired', 10,
            )
        self.assertIsNone(result)
        flush.assert_not_awaited()

    async def test_short_confirmation_uses_all_remaining_send_time(self):
        clock = Clock()
        calls = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None, request_limit=None):
            calls.append((kind, c, seconds))
            if kind == 'probe':
                clock.advance(8)  # Four seconds sending, four draining.
            else:
                self.assertEqual(seconds, 2)
                self.assertEqual(request_limit, 100)
                clock.advance(seconds + 2)  # Drain ends after the global deadline.
            return dict(kind=kind, concurrency=c, status='PASS',
                        completed=request_limit or 1,
                        last_request_started_elapsed_s=clock.read()-2.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'))
            report = await self.run_main(args, clock, stage)
        self.assertEqual([x[0] for x in calls], ['probe', 'confirm'])
        self.assertEqual(report['confirmed_concurrency'], 8)
        self.assertEqual(report['send_window_seconds'], 10)
        self.assertEqual(report['final_drain_seconds'], 2)
        self.assertEqual(report['elapsed_wall_seconds'], 12)
        self.assertLess(report['last_request_started_elapsed_s'], 10)

    async def test_insufficient_probe_continues_instead_of_ending_early(self):
        clock = Clock()
        kinds = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None, request_limit=None):
            kinds.append(kind)
            clock.advance(5 if kind == 'probe' else seconds)
            return dict(kind=kind, concurrency=c,
                        completed=request_limit or 1,
                        status='INSUFFICIENT' if kind == 'probe' else 'PASS',
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'))
            report = await self.run_main(args, clock, stage)
        self.assertEqual(kinds, ['probe', 'continue'])
        self.assertEqual(report['highest_passing_concurrency'], 8)
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['status'], 'PASSING_CANDIDATE')

    async def test_binary_search_and_confirmation_are_preserved(self):
        clock = Clock()
        probes = []
        outcomes = {8: 'FAIL', 4: 'PASS', 6: 'PASS', 7: 'FAIL'}

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None, request_limit=None):
            if kind == 'probe':
                probes.append(c)
            clock.advance(seconds)
            return dict(kind=kind, concurrency=c, status=outcomes[c],
                        completed=request_limit or 1,
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'), budget_seconds=100,
                           probe_seconds=10, confirm_seconds=10)
            report = await self.run_main(args, clock, stage)
        self.assertEqual(probes, [8, 4, 6, 7])
        self.assertEqual(report['confirmed_concurrency'], 6)
        self.assertEqual(report['smallest_observed_fail'], 7)
        self.assertEqual(report['elapsed_wall_seconds'], 100)

    async def test_failed_reconfirmation_is_not_reported_as_stable(self):
        clock = Clock()

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None, request_limit=None):
            clock.advance(seconds)
            return dict(kind=kind, concurrency=c,
                        completed=request_limit or 1,
                        status='PASS' if kind == 'probe' else 'FAIL',
                        last_request_started_elapsed_s=clock.read()-.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'), max_concurrency=8)
            report = await self.run_main(args, clock, stage)
        self.assertIsNone(report['confirmed_concurrency'])
        self.assertEqual(report['inconsistent_concurrencies'], [8])
        self.assertEqual(report['status'], 'INCONSISTENT')

    async def test_error_stops_search_instead_of_retrying_unsafe_workload(self):
        clock = Clock()
        kinds = []

        async def stage(session, args, prompts, c, seconds, kind, raw, run_id,
                        deadline, benchmark_start=None, request_limit=None):
            kinds.append(kind)
            clock.advance(1)
            return dict(kind=kind, concurrency=c, status='ERROR',
                        last_request_started_elapsed_s=.1)

        with tempfile.TemporaryDirectory() as tmp:
            args = options(str(Path(tmp) / 'run'))
            report = await self.run_main(args, clock, stage)
        self.assertEqual(kinds, ['probe'])
        self.assertEqual(report['status'], 'INCONCLUSIVE')


class FlushResponse:
    def __init__(self, status=200, body='Cache flushed.'):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return self.body


class CacheFlushTests(unittest.IsolatedAsyncioTestCase):
    async def test_post_wait_outlives_the_session_default_timeout(self):
        observed = []

        async def flush(request):
            observed.append((request.method, request.query.get('timeout')))
            # Scaled regression: the response arrives after the session deadline.
            await asyncio.sleep(.15)
            return web.Response(text='Cache flushed.')

        app = web.Application()
        app.router.add_post('/flush_cache', flush)
        async with TestServer(app) as server:
            args = options(base_url=str(server.make_url('/')).rstrip('/'))
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=.03),
            ) as session:
                await bench.flush_cache(session, args)
        self.assertEqual(observed, [('POST', '60')])

    async def test_get_fallback_can_also_wait_past_the_session_default(self):
        observed = []

        async def flush(request):
            observed.append((request.method, request.query.get('timeout')))
            if request.method == 'POST':
                return web.Response(status=405)
            await asyncio.sleep(.15)
            return web.Response(text='Cache flushed.')

        app = web.Application()
        app.router.add_route('*', '/flush_cache', flush)
        async with TestServer(app) as server:
            args = options(base_url=str(server.make_url('/')).rstrip('/'))
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=.03),
            ) as session:
                await bench.flush_cache(session, args)
        self.assertEqual(observed, [('POST', '60'), ('GET', '60')])

    async def test_both_methods_allow_more_time_than_the_server_wait(self):
        session = Mock()
        session.post.return_value = FlushResponse(status=405)
        session.get.return_value = FlushResponse()

        await bench.flush_cache(session, options())

        for method in (session.post, session.get):
            method.assert_called_once()
            call = method.call_args
            server_wait = float(call.args[0].split('timeout=')[1])
            self.assertGreater(call.kwargs['timeout'].total, server_wait)

    async def test_rejected_flush_prevents_generation(self):
        session = Mock()
        session.post.return_value = FlushResponse(status=400, body='Still busy')
        clock = Clock()
        generate = AsyncMock()

        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'request', new=generate),
        ):
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400: Still busy'):
                await bench.stage(
                    session, options(), ['prompt'], 1, 10, 'probe',
                    io.StringIO(), 'rejected', 10,
                )
        generate.assert_not_awaited()
        session.get.assert_not_called()

    async def test_flush_wait_does_not_extend_the_send_window(self):
        clock = Clock()
        generate = AsyncMock()

        async def wait_for_idle(session, args):
            clock.advance(11)

        with (
            patch.object(bench.time, 'perf_counter', clock.read),
            patch.object(bench, 'flush_cache', side_effect=wait_for_idle),
            patch.object(bench, 'request', new=generate),
        ):
            result = await bench.stage(
                None, options(), ['prompt'], 1, 10, 'probe',
                io.StringIO(), 'deadline', 10,
            )
        self.assertIsNone(result)
        generate.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()


