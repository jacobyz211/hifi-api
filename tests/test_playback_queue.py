"""Playback queue: slots come back, duplicates collapse, stalled work heals.

A wedged pool used to answer 202 to every playback request forever, which is
what clients report as "no playable stream". These tests pin the behaviour
that keeps that from happening without a restart.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def test_lease_is_tracked_and_returned():
    async def scenario():
        pool = main.PlaybackCredentialPool([{"userID": 1}, {"userID": 2}])
        assert pool.available == 2
        async with pool.lease(owner="job-1") as cred:
            assert pool.available == 1
            assert pool.report()[0]["account"] == str(cred["userID"])
            assert pool.report()[0]["requestId"] == "job-1"
        assert pool.available == 2
        assert pool.report() == []

    _run(scenario())


def test_release_twice_does_not_duplicate_the_credential():
    async def scenario():
        pool = main.PlaybackCredentialPool([{"userID": 1}])
        cred = pool.try_acquire()
        pool.release(cred)
        pool.release(cred)
        assert pool.available == 1
        assert pool.try_acquire() is not None
        assert pool.try_acquire() is None

    _run(scenario())


def test_stale_lease_is_reported_and_reclaimable():
    async def scenario():
        pool = main.PlaybackCredentialPool([{"userID": 1}])
        gate = asyncio.Event()

        async def holder():
            async with pool.lease(owner="slow"):
                await gate.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0.05)
        assert pool.available == 0
        assert pool.stale(0) and not pool.stale(60)

        # The watchdog releases the credential even though the task is stuck.
        stale = pool.stale(0)
        pool.release(stale[0]["cred"])
        assert pool.available == 1
        # The stuck task releasing later must not hand it out twice.
        gate.set()
        await task
        assert pool.available == 1

    _run(scenario())


def test_dispatch_collapses_identical_pending_requests():
    async def scenario():
        main._playback_pool = main.PlaybackCredentialPool([{"userID": 1}])
        main._playback_jobs.clear()
        main._playback_jobs_by_key.clear()
        main.PLAYBACK_QUEUE_HOLD = 0
        gate = asyncio.Event()

        async def slow(_cred):
            await gate.wait()
            return {"ok": True}

        # Hold the only account so both dispatches have to queue.
        blocker = main._playback_pool.try_acquire()
        first = await main.dispatch_playback_request(slow, key=("manifest", 1, ("FLAC",)))
        second = await main.dispatch_playback_request(slow, key=("manifest", 1, ("FLAC",)))
        assert first.status_code == 202 and second.status_code == 202
        assert len(main._playback_jobs) == 1

        main._playback_pool.release(blocker)
        gate.set()
        await asyncio.sleep(0.1)
        assert main._playback_pool.available == 1

        for job in list(main._playback_jobs.values()):
            if job.task is not None:
                job.task.cancel()

    _run(scenario())


def test_acquire_waits_for_a_free_account_instead_of_failing():
    async def scenario():
        pool = main.PlaybackCredentialPool([{"userID": 1}])
        held = pool.try_acquire()

        async def hand_back():
            await asyncio.sleep(0.05)
            pool.release(held)

        asyncio.create_task(hand_back())
        cred = await pool.acquire(timeout=2)
        assert cred is not None and cred["userID"] == 1
        pool.release(cred)
        assert pool.available == 1
        # A timed-out wait must not consume the account.
        held = pool.try_acquire()
        assert await pool.acquire(timeout=0.05) is None
        pool.release(held)
        assert pool.available == 1

    _run(scenario())


def test_stalled_job_gets_a_terminal_answer():
    async def scenario():
        main._playback_pool = main.PlaybackCredentialPool([{"userID": 1}])
        main._playback_jobs.clear()
        main._playback_jobs_by_key.clear()
        main.PLAYBACK_QUEUE_HOLD = 0

        async def never(_cred):
            await asyncio.sleep(3600)
            return {}

        blocker = main._playback_pool.try_acquire()
        response = await main.dispatch_playback_request(never, key=("manifest", 2, ("FLAC",)))
        assert response.status_code == 202

        main.PLAYBACK_JOB_TIMEOUT = 0
        main._expire_stalled_jobs()
        job = next(iter(main._playback_jobs.values()))
        assert job.state == "failed"
        assert job.error_status == 503
        main._playback_pool.release(blocker)

    _run(scenario())
