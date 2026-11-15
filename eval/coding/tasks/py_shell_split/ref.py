import re

NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WS = " \t\n"


def split_command(line, env):
    words = []
    cur = []          # list of (text, splittable)
    has_word = False  # current word has a quoted part or literal chars

    def flush():
        nonlocal cur, has_word
        fields = [""]
        started = False
        for text, splittable in cur:
            if not splittable:
                fields[-1] += text
                started = True
                continue
            for j, piece in enumerate(re.split(r"[ \t\n]+", text)):
                if j > 0:
                    if started or fields[-1]:
                        fields.append("")
                fields[-1] += piece
                if piece:
                    started = True
        if not started and fields == [""]:
            if has_word:
                words.append("")
        else:
            if fields[-1] == "" and len(fields) > 1:
                fields.pop()
            words.extend(f for f in fields if f != "" or has_word and len(fields) == 1)
        cur, has_word = [], False

    def expand(i, quoted):
        # line[i] == '$'; returns (value, new_i) or None if literal
        if i + 1 < len(line) and line[i + 1] == "{":
            j = i + 2
            m = NAME.match(line, j)
            if not m:
                raise ValueError("bad ${")
            name, j = m[0], m.end()
            if j < len(line) and line[j] == "}":
                return env.get(name, ""), j + 1
            if line.startswith(":-", j) or line.startswith("-", j):
                colon = line[j] == ":"
                j += 2 if colon else 1
                depth, k = 1, j
                while k < len(line):
                    if line[k] == "{":
                        depth += 1
                    elif line[k] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                if k >= len(line):
                    raise ValueError("unterminated ${")
                val = env.get(name)
                if val is None or (colon and val == ""):
                    val = expand_text(line[j:k])
                return val, k + 1
            raise ValueError("bad ${")
        m = NAME.match(line, i + 1)
        if m:
            return env.get(m[0], ""), m.end()
        return None

    def expand_text(text):
        out, k = [], 0
        while k < len(text):
            if text[k] == "$":
                r = split_command_expand(text, k)
                if r:
                    out.append(r[0])
                    k = r[1]
                    continue
            out.append(text[k])
            k += 1
        return "".join(out)

    def split_command_expand(text, k):
        return _expand_in(text, k, env)

    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in WS:
            if cur or has_word:
                flush()
            i += 1
        elif c == "'":
            j = line.find("'", i + 1)
            if j < 0:
                raise ValueError("unterminated '")
            cur.append((line[i + 1:j], False))
            has_word = True
            i = j + 1
        elif c == '"':
            has_word = True
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise ValueError('unterminated "')
                c = line[i]
                if c == '"':
                    i += 1
                    break
                if c == "\\" and i + 1 < n and line[i + 1] in '$"\\\n':
                    if line[i + 1] != "\n":
                        buf.append(line[i + 1])
                    i += 2
                elif c == "$":
                    r = expand(i, True)
                    if r:
                        buf.append(r[0])
                        i = r[1]
                    else:
                        buf.append("$")
                        i += 1
                else:
                    buf.append(c)
                    i += 1
            cur.append(("".join(buf), False))
        elif c == "\\":
            if i + 1 >= n:
                raise ValueError("trailing backslash")
            if line[i + 1] != "\n":
                cur.append((line[i + 1], False))
                has_word = True
            i += 2
        elif c == "$":
            r = expand(i, False)
            if r:
                cur.append((r[0], True))
                i = r[1]
            else:
                cur.append(("$", False))
                has_word = True
                i += 1
        else:
            cur.append((c, False))
            has_word = True
            i += 1
    if cur or has_word:
        flush()
    return words


def _expand_in(text, k, env):
    m = NAME.match(text, k + 1)
    if m:
        return env.get(m[0], ""), m.end()
    if text.startswith("{", k + 1):
        end = text.find("}", k)
        if end < 0:
            raise ValueError("unterminated ${")
        return env.get(text[k + 2:end], ""), end + 1
    return None
