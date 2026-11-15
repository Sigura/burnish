fun main() {
    check("example") {
        val t = tokenize("status == \"open\" && (priority >= 2 || tag != 'x\\'y')")
        eq(listOf(
            Token.Ident("status", 0), Token.Op("==", 7), Token.Str("open", 10), Token.Op("&&", 17),
            Token.Paren(true, 20), Token.Ident("priority", 21), Token.Op(">=", 30), Token.Num(2.0, 33),
            Token.Op("||", 35), Token.Ident("tag", 38), Token.Op("!=", 42), Token.Str("x'y", 45), Token.Paren(false, 51)
        ), t)
    }
    check("negative numbers") {
        eq(listOf(Token.Num(-1.5, 0)), tokenize("-1.5"))
        eq(listOf(Token.Ident("a", 0), Token.Op("<", 2), Token.Num(-3.0, 4)), tokenize("a < -3"))
        eq(listOf(Token.Paren(true, 0), Token.Num(-2.0, 1), Token.Paren(false, 3)), tokenize("(-2)"))
    }
    check("minus not allowed after value") {
        val e = runCatching { tokenize("a -1") }.exceptionOrNull()
        eq(true, e is TokenizeException); eq(2, (e as TokenizeException).pos)
    }
    check("ops longest match") {
        eq(listOf("!", "!=", "<", "<=", ">", ">="), tokenize("! != < <= > >=").map { (it as Token.Op).symbol })
    }
    check("escapes and newline") {
        eq(listOf(Token.Str("a\nb\\c\"", 0)), tokenize("\"a\\nb\\\\c\\\"\""))
    }
    check("dotted ident and whitespace") {
        eq(listOf(Token.Ident("user.name_2", 1), Token.Op("==", 14), Token.Ident("x", 18)), tokenize("\tuser.name_2 \n== \tx"))
    }
    check("empty") { eq(emptyList<Token>(), tokenize("  ")) }
    check("errors") {
        for ((src, pos) in listOf("a = b" to 2, "a & b" to 2, "'open" to 0, "x == .5" to 5, "x == 5." to 5, "a#" to 1)) {
            val e = runCatching { tokenize(src) }.exceptionOrNull()
            if (e !is TokenizeException) throw AssertionError("$src: expected TokenizeException, got $e")
            if (e.pos != pos) throw AssertionError("$src: expected pos $pos, got ${e.pos}")
        }
    }
    finish()
}
