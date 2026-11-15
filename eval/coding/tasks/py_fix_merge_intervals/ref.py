import bisect


class IntervalSet:
    def __init__(self, intervals=()):
        self._iv = []
        for s, e in intervals:
            self.add(s, e)

    def add(self, start, end):
        if start >= end:
            return
        i = bisect.bisect_left(self._iv, (start, end))
        if i > 0 and self._iv[i - 1][1] >= start:
            i -= 1
        j = i
        while j < len(self._iv) and self._iv[j][0] <= end:
            start = min(start, self._iv[j][0])
            end = max(end, self._iv[j][1])
            j += 1
        self._iv[i:j] = [(start, end)]

    def remove(self, start, end):
        if start >= end:
            return
        out = []
        for s, e in self._iv:
            if e <= start or s >= end:
                out.append((s, e))
                continue
            if s < start:
                out.append((s, start))
            if e > end:
                out.append((end, e))
        self._iv = out

    def contains(self, x):
        i = bisect.bisect_right(self._iv, (x, float("inf")))
        return i > 0 and self._iv[i - 1][0] <= x < self._iv[i - 1][1]

    def free(self, lo, hi):
        gaps = []
        cur = lo
        for s, e in self._iv:
            if e <= cur:
                continue
            if s >= hi:
                break
            if s > cur:
                gaps.append((cur, s))
            cur = max(cur, e)
        if cur < hi:
            gaps.append((cur, hi))
        return gaps

    def __iter__(self):
        return iter(self._iv)

    def __len__(self):
        return len(self._iv)
