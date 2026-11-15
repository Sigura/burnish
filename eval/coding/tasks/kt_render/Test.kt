enum class Color { RED, GREEN }

fun main() {
    check("scalars") {
        eq("null", toJson(null)); eq("true", toJson(true)); eq("42", toJson(42)); eq("9000000000", toJson(9000000000L))
        eq("2.0", toJson(2.0)); eq("1.5", toJson(1.5f)); eq("\"RED\"", toJson(Color.RED))
    }
    check("escapes") {
        eq("\"a\\\"b\\\\c\\nd\\te\\u0001é\"", toJson("a\"b\\c\nd\te\u0001é"))
        eq("\"\\b\\f\\r\\u001f\"", toJson("\b\u000c\r\u001f"))
    }
    check("compact nested") {
        val v = linkedMapOf("a" to 1, "b" to listOf(1, null, "x"), "c" to mapOf<String, Any>(), "d" to emptyList<Int>())
        eq("{\"a\":1,\"b\":[1,null,\"x\"],\"c\":{},\"d\":[]}", toJson(v))
    }
    check("array") { eq("[1,2]", toJson(arrayOf(1, 2))) }
    check("pretty") {
        val v = linkedMapOf("name" to "x", "tags" to listOf("a", "b"), "empty" to emptyMap<String, Int>(),
                            "nested" to mapOf("k" to listOf(mapOf("z" to 1))))
        val expected = """
{
  "name": "x",
  "tags": [
    "a",
    "b"
  ],
  "empty": {},
  "nested": {
    "k": [
      {
        "z": 1
      }
    ]
  }
}""".trimStart('\n')
        eq(expected, toJson(v, 2))
    }
    check("bad key") { throws<IllegalArgumentException> { toJson(mapOf(1 to 2)) } }
    check("nan") { throws<IllegalArgumentException> { toJson(Double.NaN) } }
    check("unsupported") { throws<IllegalArgumentException> { toJson(Any()) } }
    finish()
}
