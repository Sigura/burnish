sealed class Token {
    abstract val pos: Int
    data class Ident(val name: String, override val pos: Int) : Token()
    data class Str(val value: String, override val pos: Int) : Token()
    data class Num(val value: Double, override val pos: Int) : Token()
    data class Op(val symbol: String, override val pos: Int) : Token()
    data class Paren(val open: Boolean, override val pos: Int) : Token()
}

class TokenizeException(message: String, val pos: Int) : Exception(message)

private val OPS = listOf("==", "!=", "<=", ">=", "&&", "||", "<", ">", "!")

fun tokenize(input: String): List<Token> {
    val out = mutableListOf<Token>()
    var i = 0
    while (i < input.length) {
        val c = input[i]
        when {
            c == ' ' || c == '\t' || c == '\n' -> i++
            c.isLetter() && c.code < 128 || c == '_' -> {
                val start = i
                i++
                while (i < input.length && (input[i].isLetterOrDigit() && input[i].code < 128 || input[i] == '_' || input[i] == '.')) i++
                out.add(Token.Ident(input.substring(start, i), start))
            }
            c.isDigit() || c == '-' -> {
                val start = i
                if (c == '-') {
                    val prev = out.lastOrNull()
                    val ok = prev == null || prev is Token.Op || (prev is Token.Paren && prev.open)
                    if (!ok || i + 1 >= input.length || !input[i + 1].isDigit()) throw TokenizeException("unexpected -", i)
                    i++
                }
                while (i < input.length && input[i].isDigit()) i++
                if (i < input.length && input[i] == '.') {
                    i++
                    if (i >= input.length || !input[i].isDigit()) throw TokenizeException("bad number", start)
                    while (i < input.length && input[i].isDigit()) i++
                }
                out.add(Token.Num(input.substring(start, i).toDouble(), start))
            }
            c == '"' || c == '\'' -> {
                val start = i
                val sb = StringBuilder()
                i++
                while (true) {
                    if (i >= input.length) throw TokenizeException("unterminated string", start)
                    val d = input[i]
                    if (d == c) { i++; break }
                    if (d == '\\') {
                        if (i + 1 >= input.length) throw TokenizeException("unterminated string", start)
                        val e = input[i + 1]
                        sb.append(if (e == 'n') '\n' else e)
                        i += 2
                    } else {
                        sb.append(d); i++
                    }
                }
                out.add(Token.Str(sb.toString(), start))
            }
            c == '(' || c == ')' -> { out.add(Token.Paren(c == '(', i)); i++ }
            else -> {
                val op = OPS.firstOrNull { input.startsWith(it, i) } ?: throw TokenizeException("unexpected '$c'", i)
                out.add(Token.Op(op, i))
                i += op.length
            }
        }
    }
    return out
}
