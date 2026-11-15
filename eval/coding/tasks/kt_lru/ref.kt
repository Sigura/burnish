class LruCache<K, V>(val capacity: Int, private val onEvict: (K, V) -> Unit = { _, _ -> }) {
    private val map = LinkedHashMap<K, V>(16, 0.75f, false)

    init {
        require(capacity >= 1) { "capacity must be >= 1" }
    }

    val size: Int get() = map.size

    operator fun get(key: K): V? {
        if (!map.containsKey(key)) return null
        @Suppress("UNCHECKED_CAST")
        val v = map.remove(key) as V
        map[key] = v
        return v
    }

    operator fun set(key: K, value: V) {
        if (map.containsKey(key)) {
            map.remove(key)
            map[key] = value
            return
        }
        map[key] = value
        if (map.size > capacity) {
            val k = map.keys.first()
            @Suppress("UNCHECKED_CAST")
            val v = map.remove(k) as V
            onEvict(k, v)
        }
    }

    fun remove(key: K): V? = map.remove(key)

    fun keys(): List<K> = map.keys.toList()

    operator fun contains(key: K): Boolean = map.containsKey(key)
}
