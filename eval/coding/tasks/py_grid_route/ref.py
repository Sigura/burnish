import heapq

DIRS = [("D", 1, 0), ("L", 0, -1), ("R", 0, 1), ("U", -1, 0)]
ORDER = {m: i for i, m in enumerate("DLRTU")}


def route(grid):
    h = len(grid)
    if h == 0:
        raise ValueError("empty")
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("ragged")
    pads, s, e = {}, [], []
    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            if ch == "S":
                s.append((r, c))
            elif ch == "E":
                e.append((r, c))
            elif ch.islower() and ch.isascii():
                pads.setdefault(ch, []).append((r, c))
            elif ch not in ".#123456789":
                raise ValueError(f"bad char {ch!r}")
    if len(s) != 1 or len(e) != 1 or any(len(v) != 2 for v in pads.values()):
        raise ValueError("bad S/E/pads")
    s, e = s[0], e[0]

    def cost(p):
        ch = grid[p[0]][p[1]]
        return int(ch) if ch.isdigit() else 1

    def nbrs(p):
        r, c = p
        out = []
        for m, dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != "#":
                out.append((m, (nr, nc), cost((nr, nc))))
        ch = grid[r][c]
        if ch in pads:
            a, b = pads[ch]
            out.append(("T", b if a == p else a, 2))
        return out

    # distances to E on the reversed graph, then greedy lexicographic walk forward
    rev = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != "#":
                for m, q, k in nbrs((r, c)):
                    rev.setdefault(q, []).append(((r, c), k))
    dist = {e: 0}
    pq = [(0, e)]
    while pq:
        d, p = heapq.heappop(pq)
        if d > dist[p]:
            continue
        for q, k in rev.get(p, []):
            if d + k < dist.get(q, float("inf")):
                dist[q] = d + k
                heapq.heappush(pq, (d + k, q))
    if s not in dist:
        return None
    path, p = [], s
    while p != e:
        for m, q, k in sorted(nbrs(p), key=lambda t: ORDER[t[0]]):
            if q in dist and dist[q] + k == dist[p]:
                path.append(m)
                p = q
                break
    return dist[s], "".join(path)
