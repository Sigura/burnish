var passed = 0
var failed = 0

fun check(name: String, block: () -> Unit) {
    try {
        block()
        passed++
    } catch (e: Throwable) {
        failed++
        println("FAIL $name: $e")
    }
}

fun eq(expected: Any?, actual: Any?) {
    if (expected != actual) throw AssertionError("expected <$expected> but was <$actual>")
}

inline fun <reified T : Throwable> throws(block: () -> Unit) {
    try {
        block()
    } catch (e: Throwable) {
        if (e is T) return
        throw AssertionError("expected ${T::class.simpleName} but got $e")
    }
    throw AssertionError("expected ${T::class.simpleName}")
}

fun finish() {
    println("$passed/${passed + failed} passed")
    kotlin.system.exitProcess(if (failed == 0) 0 else 1)
}

