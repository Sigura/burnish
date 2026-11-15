import re

HDR = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_patch(original, diff):
    src = original.splitlines(True)
    dl = diff.splitlines(True)
    i = 0
    while i < len(dl) and not dl[i].startswith("@@"):
        i += 1
    out = []
    pos = 0
    while i < len(dl):
        m = HDR.match(dl[i])
        if not m:
            raise ValueError(f"bad header {dl[i]!r}")
        ostart, ocount = int(m[1]), int(m[2]) if m[2] is not None else 1
        ncount = int(m[4]) if m[4] is not None else 1
        i += 1
        body = []
        while i < len(dl) and not dl[i].startswith("@@"):
            line = dl[i]
            if line.startswith("\\"):
                if not body:
                    raise ValueError("stray no-newline marker")
                kind, text = body[-1]
                body[-1] = (kind, text.removesuffix("\n"))
            elif line[:1] in (" ", "-", "+"):
                body.append((line[0], line[1:]))
            elif line in ("\n", "\r\n"):
                body.append((" ", line))
            else:
                raise ValueError(f"bad body line {line!r}")
            i += 1
        olines = [t for k, t in body if k != "+"]
        nlines = [t for k, t in body if k != "-"]
        if len(olines) != ocount or len(nlines) != ncount:
            raise ValueError("hunk counts disagree with body")
        start = ostart if ocount == 0 else ostart - 1
        if start < pos or start + ocount > len(src):
            raise ValueError("hunk out of range")
        if src[start:start + ocount] != olines:
            raise ValueError("hunk does not match")
        out += src[pos:start]
        out += nlines
        pos = start + ocount
    out += src[pos:]
    return "".join(out)
