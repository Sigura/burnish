fun toJson(value: Any?, indent: Int = 0): String {
    val sb = StringBuilder()
    write(sb, value, indent, 0)
    return sb.toString()
}

private fun escape(s: String): String {
    val sb = StringBuilder("\"")
    for (c in s) {
        when {
            c == '"' -> sb.append("\\\"")
            c == '\\' -> sb.append("\\\\")
            c == '\b' -> sb.append("\\b")
            c == '\u000c' -> sb.append("\\f")
            c == '\n' -> sb.append("\\n")
            c == '\r' -> sb.append("\\r")
            c == '\t' -> sb.append("\\t")
            c < ' ' -> sb.append("\\u%04x".format(c.code))
            else -> sb.append(c)
        }
    }
    return sb.append('"').toString()
}

private fun write(sb: StringBuilder, v: Any?, indent: Int, level: Int) {
    when (v) {
        null -> sb.append("null")
        is Boolean, is Int, is Long -> sb.append(v.toString())
        is Double -> {
            require(v.isFinite()) { "non-finite double" }
            sb.append(v.toString())
        }
        is Float -> {
            require(v.isFinite()) { "non-finite float" }
            sb.append(v.toString())
        }
        is String -> sb.append(escape(v))
        is Enum<*> -> sb.append(escape(v.name))
        is Map<*, *> -> {
            val entries = v.entries.map { (k, x) ->
                require(k is String) { "map keys must be strings" }
                k to x
            }
            container(sb, '{', '}', entries, indent, level) { (k, x) ->
                sb.append(escape(k)).append(if (indent > 0) ": " else ":")
                write(sb, x, indent, level + 1)
            }
        }
        is List<*> -> container(sb, '[', ']', v, indent, level) { write(sb, it, indent, level + 1) }
        is Array<*> -> container(sb, '[', ']', v.toList(), indent, level) { write(sb, it, indent, level + 1) }
        else -> throw IllegalArgumentException("unsupported type ${v::class}")
    }
}

private fun <T> container(sb: StringBuilder, open: Char, close: Char, items: List<T>, indent: Int, level: Int,
                          item: (T) -> Unit) {
    sb.append(open)
    if (items.isEmpty()) {
        sb.append(close)
        return
    }
    items.forEachIndexed { i, x ->
        if (i > 0) sb.append(',')
        if (indent > 0) sb.append('\n').append(" ".repeat(indent * (level + 1)))
        item(x)
    }
    if (indent > 0) sb.append('\n').append(" ".repeat(indent * level))
    sb.append(close)
}
