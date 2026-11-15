import random
import time

from solution import route


def test_simple():
    assert route(["S.E"]) == (2, "RR")
    assert route(["S#E"]) is None


def test_weights():
    g = ["S9E",
         "..."]
    assert route(g) == (4, "DRRU")


def test_tie_break_order():
    g = ["S.",
         ".E"]
    assert route(g) == (2, "DR")
    g = ["...",
         ".S.",
         "..E"]
    assert route(g) == (2, "DR")


def test_teleport():
    g = ["Sa#....",
         "###.#a#",
         "E.....#"]
    assert route(g) == (9, "RTDLLLLL")


def test_teleport_tie():
    g = ["SaE",
         "#.#",
         "#a#"]
    assert route(g) == (2, "RR")


def test_errors():
    for bad in [["S.", "."], ["..", ".E"], ["SS", "EE"], ["Sa", ".E"], ["Saaa", "...E"], ["S?E"]]:
        try:
            route(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


DIRS = {"D": (1, 0), "L": (0, -1), "R": (0, 1), "U": (-1, 0)}


def cell_cost(ch):
    return int(ch) if ch.isdigit() else 1


def min_cost_bruteforce(g):
    # Bellman-Ford style relaxation, independent of the solution's algorithm
    h, w = len(g), len(g[0])
    pads = {}
    for r in range(h):
        for c in range(w):
            if g[r][c].islower():
                pads.setdefault(g[r][c], []).append((r, c))
            if g[r][c] == "S":
                s = (r, c)
            if g[r][c] == "E":
                e = (r, c)
    inf = float("inf")
    dist = {s: 0}
    changed = True
    while changed:
        changed = False
        for (r, c), d in list(dist.items()):
            nbrs = []
            for dr, dc in DIRS.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and g[nr][nc] != "#":
                    nbrs.append(((nr, nc), cell_cost(g[nr][nc])))
            if g[r][c].islower():
                a, b = pads[g[r][c]]
                nbrs.append((b if a == (r, c) else a, 2))
            for p, cost in nbrs:
                if d + cost < dist.get(p, inf):
                    dist[p] = d + cost
                    changed = True
    return dist.get(e)


def replay(g, moves):
    h, w = len(g), len(g[0])
    pads = {}
    for r in range(h):
        for c in range(w):
            if g[r][c].islower():
                pads.setdefault(g[r][c], []).append((r, c))
            if g[r][c] == "S":
                pos = (r, c)
    total = 0
    for m in moves:
        if m == "T":
            a, b = pads[g[pos[0]][pos[1]]]
            pos = b if a == pos else a
            total += 2
        else:
            dr, dc = DIRS[m]
            pos = (pos[0] + dr, pos[1] + dc)
            assert g[pos[0]][pos[1]] != "#"
            total += cell_cost(g[pos[0]][pos[1]])
    assert g[pos[0]][pos[1]] == "E"
    return total


def lex_min_bruteforce(g, best):
    # BFS over move strings in lexicographic order limited by cost; only for tiny grids
    h, w = len(g), len(g[0])
    pads = {}
    for r in range(h):
        for c in range(w):
            if g[r][c].islower():
                pads.setdefault(g[r][c], []).append((r, c))
            if g[r][c] == "S":
                s = (r, c)
    results = []

    def dfs(pos, cost, path, seen):
        if cost > best:
            return
        if g[pos[0]][pos[1]] == "E":
            if cost == best:
                results.append(path)
            return
        for m in "DLRTU":
            if m == "T":
                if not g[pos[0]][pos[1]].islower():
                    continue
                a, b = pads[g[pos[0]][pos[1]]]
                np, c2 = (b if a == pos else a), 2
            else:
                dr, dc = DIRS[m]
                np = (pos[0] + dr, pos[1] + dc)
                if not (0 <= np[0] < h and 0 <= np[1] < w) or g[np[0]][np[1]] == "#":
                    continue
                c2 = cell_cost(g[np[0]][np[1]])
            if np in seen:
                continue
            dfs(np, cost + c2, path + m, seen | {np})

    dfs(s, 0, "", {s})
    return min(results) if results else None


def random_grid(rng, h, w):
    cells = [[rng.choice("....123#") for _ in range(w)] for _ in range(h)]
    spots = rng.sample([(r, c) for r in range(h) for c in range(w)], 6)
    for (r, c), ch in zip(spots, "SEabab"):
        cells[r][c] = ch
    return ["".join(row) for row in cells]


def test_random_small_against_bruteforce():
    rng = random.Random(3)
    for _ in range(150):
        g = random_grid(rng, rng.randint(3, 4), rng.randint(3, 4))
        best = min_cost_bruteforce(g)
        got = route(g)
        if best is None:
            assert got is None, (g, got)
            continue
        assert got is not None and got[0] == best, (g, got, best)
        assert replay(g, got[1]) == best, (g, got)
        assert got[1] == lex_min_bruteforce(g, best), (g, got, lex_min_bruteforce(g, best))


def test_large_fast():
    rng = random.Random(11)
    g = random_grid(rng, 60, 60)
    t = time.monotonic()
    got = route(g)
    assert time.monotonic() - t < 2.0
    best = min_cost_bruteforce(g)
    assert (got is None and best is None) or (got[0] == best and replay(g, got[1]) == best)


from _runner import main
main(globals())
