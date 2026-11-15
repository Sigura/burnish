import asyncio


async def gather_limited(factories, limit, timeout=None):
    if limit < 1:
        raise ValueError("limit must be >= 1")
    results = [None] * len(factories)
    pending = {}
    next_i = 0
    error = None

    async def run_one(i):
        aw = factories[i]()
        if timeout is not None:
            return await asyncio.wait_for(aw, timeout)
        return await aw

    try:
        while next_i < len(factories) or pending:
            while next_i < len(factories) and len(pending) < limit:
                pending[asyncio.ensure_future(run_one(next_i))] = next_i
                next_i += 1
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                i = pending.pop(t)
                if t.exception() is not None:
                    error = t.exception()
                    break
                results[i] = t.result()
            if error is not None:
                break
    finally:
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    if error is not None:
        raise error
    return results
