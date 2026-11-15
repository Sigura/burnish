import asyncio
import time

from solution import gather_limited


def run(coro):
    return asyncio.run(coro)


def test_order_and_limit():
    running = 0
    peak = 0
    started = []

    def make(i, delay):
        async def job():
            nonlocal running, peak
            started.append(i)
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(delay)
            running -= 1
            return i * 10
        return job

    delays = [0.05, 0.01, 0.03, 0.02, 0.01, 0.04, 0.01]
    res = run(gather_limited([make(i, d) for i, d in enumerate(delays)], 3))
    assert res == [i * 10 for i in range(7)]
    assert peak == 3, peak
    assert started[:3] == [0, 1, 2]


def test_lazy_factories():
    called = []

    def make(i):
        def f():
            called.append(i)
            async def job():
                await asyncio.sleep(0.02)
                return i
            return job()
        return f

    async def main():
        t = asyncio.create_task(gather_limited([make(i) for i in range(4)], 2))
        await asyncio.sleep(0.005)
        snapshot = list(called)
        return await t, snapshot

    res, snapshot = run(main())
    assert res == [0, 1, 2, 3]
    assert snapshot == [0, 1], snapshot


def test_first_error_cancels_and_cleans_up():
    cleaned = []
    started = []

    def slow(i):
        async def job():
            started.append(i)
            try:
                await asyncio.sleep(1)
            finally:
                await asyncio.sleep(0.01)
                cleaned.append(i)
        return job

    def boom(msg, delay):
        async def job():
            await asyncio.sleep(delay)
            raise RuntimeError(msg)
        return job

    fs = [slow(0), boom("second", 0.05), boom("first", 0.02), slow(3), slow(4)]
    t0 = time.monotonic()
    try:
        run(gather_limited(fs, 4))
    except RuntimeError as e:
        assert str(e) == "first", str(e)
    else:
        raise AssertionError("expected RuntimeError")
    assert time.monotonic() - t0 < 0.5
    assert sorted(cleaned) == [0, 3], cleaned
    assert 4 not in started


def test_timeout_per_task():
    def make(d):
        async def job():
            await asyncio.sleep(d)
            return d
        return job

    assert run(gather_limited([make(0.01), make(0.02)], 1, timeout=0.1)) == [0.01, 0.02]
    try:
        run(gather_limited([make(0.01), make(0.3)], 2, timeout=0.05))
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected TimeoutError")


def test_timeout_is_per_task_not_total():
    def make():
        async def job():
            await asyncio.sleep(0.04)
            return 1
        return job

    assert run(gather_limited([make() for _ in range(5)], 1, timeout=0.1)) == [1] * 5


def test_outer_cancel():
    cleaned = []

    def make(i):
        async def job():
            try:
                await asyncio.sleep(1)
            finally:
                cleaned.append(i)
        return job

    async def main():
        t = asyncio.create_task(gather_limited([make(i) for i in range(3)], 2))
        await asyncio.sleep(0.02)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            return "cancelled"
        return "not cancelled"

    assert run(main()) == "cancelled"
    assert sorted(cleaned) == [0, 1], cleaned


def test_edge_cases():
    assert run(gather_limited([], 3)) == []
    try:
        run(gather_limited([], 0))
    except ValueError:
        pass
    else:
        raise AssertionError("limit 0 must raise ValueError")


from _runner import main
main(globals())
