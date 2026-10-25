"""Adversarial scheduling checks for scored cohorts and background refill."""
import asyncio
import contextlib
import io
import json
import time
import unittest
from unittest.mock import patch

import longswe_slo as bench
import test_longswe_slo as fixture


def role_index(rid):
    _, role, index = rid.rsplit('-', 2)
    return role, int(index)


class SteadyConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def run_stage(self, generate, *, target=4, concurrency=2, clock=None, deadline=60):
        raw = io.StringIO()
        timer = (patch.object(bench.time, 'perf_counter', clock.read)
                 if clock is not None else contextlib.nullcontext())
        limit = deadline if clock is not None else time.perf_counter() + deadline
        with (
            timer,
            patch.object(bench, 'request', side_effect=generate),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = await asyncio.wait_for(bench.stage(
                None, fixture.options(keep_cache=True), ['a', 'b', 'c', 'd'],
                concurrency, deadline, 'confirm', raw, 'steady', limit,
                request_limit=target,
            ), timeout=5)
        return result, [json.loads(line) for line in raw.getvalue().splitlines()]

    async def test_early_warmup_finishers_refill_before_scoring_starts(self):
        release = asyncio.Event()
        warm_finished = set()
        warm_background = 0

        async def generate(session, args, prompt, rid, **kwargs):
            nonlocal warm_background
            role, index = role_index(rid)
            if role == 'warmup':
                if index == 1:
                    await release.wait()
                else:
                    await asyncio.sleep(0)
                warm_finished.add(index)
            elif role == 'background':
                if len(warm_finished) < 2:
                    warm_background += 1
                    if warm_background == 3:
                        release.set()
                await asyncio.sleep(0)
            else:
                self.assertEqual(len(warm_finished), 2)
                await asyncio.sleep(0)
            return fixture.row()

        result, rows = await self.run_stage(generate)
        self.assertEqual(warm_background, 3)
        self.assertEqual(result['warmup_completed'], 2)
        self.assertTrue(result['steady_state_valid'])
        self.assertEqual(sum(row['measurement_role'] == 'measured' for row in rows), 4)

    async def test_slowest_scored_request_is_kept_and_background_metrics_are_excluded(self):
        clock = fixture.Clock()
        release = asyncio.Event()
        cohort_done = asyncio.Event()
        fast_done = background_after_assignment = 0
        events = []

        async def generate(session, args, prompt, rid, **kwargs):
            nonlocal fast_done, background_after_assignment
            role, index = role_index(rid)
            events.append(('start', role, index))
            if role == 'measured' and index == 0:
                await release.wait()
                events.append(('last_scored_done', role, index))
                cohort_done.set()
            else:
                await asyncio.sleep(0)
                if role == 'measured':
                    fast_done += 1
                elif role == 'background' and fast_done == 3:
                    background_after_assignment += 1
                    if background_after_assignment == 3:
                        release.set()
                    if cohort_done.is_set():
                        clock.advance(2)  # Extra background drain is not scored.
            row = fixture.row()
            if role != 'measured':
                row.update(ttft_s=999, decode_tps=.01)
            return row

        result, rows = await self.run_stage(generate, clock=clock)
        self.assertGreaterEqual(background_after_assignment, 3)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['p90_ttft_s'], 1)
        self.assertEqual(result['mean_decode_tps'], 20)
        self.assertGreater(result['drain_seconds'], 0)
        self.assertEqual(result['total_sent'], len(rows))
        scored = [row for row in rows if row['measurement_role'] == 'measured']
        self.assertEqual({row['request_index'] for row in scored}, {0, 1, 2, 3})
        self.assertEqual(scored[-1]['request_index'], 0)
        end = next(i for i, event in enumerate(events) if event[0] == 'last_scored_done')
        self.assertFalse(any(event[0] == 'start' for event in events[end + 1:]))

    async def test_deadline_stops_background_and_late_cohort_cannot_pass(self):
        clock = fixture.Clock()
        release = asyncio.Event()
        starts = []

        async def generate(session, args, prompt, rid, **kwargs):
            role, index = role_index(rid)
            starts.append(clock.read())
            self.assertLess(clock.read(), kwargs['admission_deadline'])
            if role == 'measured':
                await release.wait()
            else:
                if role == 'background':
                    clock.advance(4)
                    if clock.read() >= 10:
                        release.set()
                await asyncio.sleep(0)
            return fixture.row()

        result, rows = await self.run_stage(generate, clock=clock, deadline=10, target=1)
        self.assertTrue(all(at < 10 for at in starts))
        self.assertEqual(result['completed'], 1)
        self.assertTrue(result['request_target_met'])
        self.assertFalse(result['steady_state_valid'])
        self.assertEqual(result['status'], 'INSUFFICIENT')
        self.assertEqual(result['stop_reason'], 'admission_deadline')
        self.assertEqual(result['total_sent'], len(rows))

    async def test_background_error_invalidates_otherwise_passing_scored_metrics(self):
        measured_started = asyncio.Event()
        failed = asyncio.Event()

        async def generate(session, args, prompt, rid, **kwargs):
            role, index = role_index(rid)
            if role == 'background':
                await measured_started.wait()
                failed.set()
                return dict(ok=False, error='background failed')
            if role == 'measured':
                measured_started.set()
                await failed.wait()
            else:
                await asyncio.sleep(0)
            return fixture.row()

        result, rows = await self.run_stage(generate)
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['errors'], 0)  # Scored errors only.
        self.assertEqual(result['load_errors'], 1)
        self.assertEqual(result['total_errors'], 1)
        self.assertEqual(result['error_examples'], ['background failed'])
        self.assertFalse(result['steady_state_valid'])
        self.assertEqual(result['total_sent'], len(rows))

    async def test_cohort_finishing_exactly_at_deadline_is_not_confirmed(self):
        clock = fixture.Clock()

        async def generate(session, args, prompt, rid, **kwargs):
            clock.advance(4)
            return fixture.row()

        result, rows = await self.run_stage(
            generate, target=3, concurrency=1, clock=clock, deadline=16)
        self.assertEqual(len(rows), 4)  # One warmup plus the whole cohort.
        self.assertTrue(result['request_target_met'])
        self.assertFalse(result['steady_state_valid'])
        self.assertEqual(result['status'], 'INSUFFICIENT')

    async def test_background_may_drain_after_deadline_when_cohort_finished_in_time(self):
        clock = fixture.Clock()
        measured_done = asyncio.Event()
        starts = []

        async def generate(session, args, prompt, rid, **kwargs):
            starts.append(clock.read())
            role, index = role_index(rid)
            if role == 'warmup':
                await asyncio.sleep(0)
            elif role == 'background':
                await measured_done.wait()
                clock.advance(20)
            else:
                measured_done.set()
            return fixture.row()

        result, rows = await self.run_stage(generate, target=1, clock=clock, deadline=10)
        self.assertTrue(all(at < 10 for at in starts))
        self.assertEqual(result['completed'], 1)
        self.assertGreater(result['wall_seconds'], 10)
        self.assertTrue(result['steady_state_valid'])
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(len(rows), result['total_sent'])

    async def test_warmup_error_cannot_start_a_scored_cohort(self):
        async def generate(session, args, prompt, rid, **kwargs):
            self.assertEqual(role_index(rid)[0], 'warmup')
            return dict(ok=False, error='warmup failed')

        result, rows = await self.run_stage(generate)
        self.assertEqual(result['status'], 'ERROR')
        self.assertEqual(result['sent'], 0)
        self.assertEqual(result['load_errors'], 1)
        self.assertEqual(len(rows), 1)


if __name__ == '__main__':
    unittest.main()
