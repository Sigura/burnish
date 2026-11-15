import re

TOKEN = re.compile(r"\s*(?:(\d+\.\d*|\.\d+|\d+)|([A-Za-z_]\w*)|(==|!=|<=|>=|&&|\|\||[-+*/%^!<>?:(),]))")


def tokenize(s):
    pos, out = 0, []
    s = s.rstrip()
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:
            raise SyntaxError(f"bad input at {pos}")
        pos = m.end()
        if m[1]:
            out.append(("num", float(m[1]) if "." in m[1] else int(m[1])))
        elif m[2]:
            out.append(("id", m[2]))
        else:
            out.append(("op", m[3]))
    out.append(("end", None))
    return out


class P:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i]

    def take(self, op=None):
        tok = self.t[self.i]
        if op is not None and tok != ("op", op):
            raise SyntaxError(f"expected {op}")
        self.i += 1
        return tok

    def at(self, *ops):
        k, v = self.peek()
        return k == "op" and v in ops

    # every parse_* returns a thunk so branches can be skipped
    def ternary(self):
        c = self.lor()
        if self.at("?"):
            self.take()
            a = self.ternary()
            self.take(":")
            b = self.ternary()
            return lambda env: a(env) if c(env) != 0 else b(env)
        return c

    def lor(self):
        l = self.land()
        while self.at("||"):
            self.take()
            r = self.land()
            l = (lambda l, r: lambda env: 1 if l(env) != 0 or r(env) != 0 else 0)(l, r)
        return l

    def land(self):
        l = self.cmp()
        while self.at("&&"):
            self.take()
            r = self.cmp()
            l = (lambda l, r: lambda env: 1 if l(env) != 0 and r(env) != 0 else 0)(l, r)
        return l

    def cmp(self):
        l = self.add()
        ops = ("==", "!=", "<", "<=", ">", ">=")
        if self.at(*ops):
            op = self.take()[1]
            r = self.add()
            if self.at(*ops):
                raise SyntaxError("chained comparison")
            f = {"==": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
                 "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}[op]
            return lambda env: int(f(l(env), r(env)))
        return l

    def add(self):
        l = self.mul()
        while self.at("+", "-"):
            op = self.take()[1]
            r = self.mul()
            l = (lambda l, r, op: lambda env: l(env) + r(env) if op == "+" else l(env) - r(env))(l, r, op)
        return l

    def mul(self):
        l = self.unary()
        while self.at("*", "/", "%"):
            op = self.take()[1]
            r = self.unary()
            l = (lambda l, r, op: lambda env: binop(op, l(env), r(env)))(l, r, op)
        return l

    def unary(self):
        if self.at("-", "+", "!"):
            op = self.take()[1]
            x = self.unary()
            if op == "-":
                return lambda env: -x(env)
            if op == "+":
                return x
            return lambda env: 1 if x(env) == 0 else 0
        return self.power()

    def power(self):
        base = self.atom()
        if self.at("^"):
            self.take()
            exp = self.unary()
            return lambda env: base(env) ** exp(env)
        return base

    def atom(self):
        k, v = self.take()
        if k == "num":
            return lambda env: v
        if k == "id":
            if self.at("("):
                self.take()
                args = []
                if not self.at(")"):
                    args.append(self.ternary())
                    while self.at(","):
                        self.take()
                        args.append(self.ternary())
                self.take(")")
                if v in ("min", "max"):
                    if not args:
                        raise SyntaxError(f"{v} needs arguments")
                    fn = min if v == "min" else max
                    return lambda env: fn(a(env) for a in args)
                if v == "abs":
                    if len(args) != 1:
                        raise SyntaxError("abs takes one argument")
                    return lambda env: abs(args[0](env))
                return lambda env: (_ for _ in ()).throw(NameError(v))

            def lookup(env):
                if v not in env:
                    raise NameError(v)
                return env[v]
            return lookup
        if (k, v) == ("op", "("):
            e = self.ternary()
            self.take(")")
            return e
        raise SyntaxError(f"unexpected {v!r}")


def binop(op, a, b):
    if op == "*":
        return a * b
    if b == 0:
        raise ZeroDivisionError(op)
    if op == "%":
        return a % b
    if isinstance(a, int) and isinstance(b, int) and a % b == 0:
        return a // b
    return a / b


def evaluate(expr, env=None):
    p = P(tokenize(expr))
    f = p.ternary()
    if p.peek()[0] != "end":
        raise SyntaxError("trailing tokens")
    return f(env or {})
