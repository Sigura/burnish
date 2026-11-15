fun main() {
    check("basic") {
        val c = LruCache<String, Int>(2)
        c["a"] = 1; c["b"] = 2
        eq(1, c["a"]); eq(2, c.size); eq(listOf("b", "a"), c.keys())
    }
    check("eviction order and callback") {
        val evicted = mutableListOf<Pair<String, Int>>()
        val c = LruCache<String, Int>(2) { k, v -> evicted.add(k to v) }
        c["a"] = 1; c["b"] = 2; c["a"]; c["c"] = 3
        eq(listOf("b" to 2), evicted); eq(listOf("a", "c"), c.keys())
        eq(null, c["b"])
    }
    check("update marks recent, no eviction") {
        val evicted = mutableListOf<String>()
        val c = LruCache<String, Int>(2) { k, _ -> evicted.add(k) }
        c["a"] = 1; c["b"] = 2; c["a"] = 10
        eq(emptyList<String>(), evicted); eq(listOf("b", "a"), c.keys()); eq(10, c["a"])
        c["c"] = 3
        eq(listOf("b"), evicted)
    }
    check("contains does not touch recency") {
        val c = LruCache<String, Int>(2)
        c["a"] = 1; c["b"] = 2
        eq(true, "a" in c); eq(listOf("a", "b"), c.keys())
        c["c"] = 3
        eq(false, "a" in c)
    }
    check("remove without callback") {
        var calls = 0
        val c = LruCache<Int, String>(3) { _, _ -> calls++ }
        c[1] = "x"; c[2] = "y"
        eq("x", c.remove(1)); eq(null, c.remove(1)); eq(1, c.size); eq(0, calls)
    }
    check("nullable values") {
        val c = LruCache<String, String?>(2)
        c["n"] = null
        eq(true, "n" in c); eq(1, c.size)
    }
    check("capacity one") {
        val c = LruCache<Int, Int>(1)
        c[1] = 1; c[2] = 2
        eq(listOf(2), c.keys())
    }
    check("bad capacity") { throws<IllegalArgumentException> { LruCache<Int, Int>(0) } }
    finish()
}
