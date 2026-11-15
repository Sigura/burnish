import java.math.BigDecimal

fun bd(s: String) = BigDecimal(s)

fun main() {
    check("no drift") {
        val l = Ledger()
        l.openAccount("a", bd("1.00")); l.openAccount("b")
        repeat(10) { l.transfer("a", "b", bd("0.10")) }
        eq(bd("0.00"), l.balance("a")); eq(bd("1.00"), l.balance("b"))
    }
    check("scale normalized") {
        val l = Ledger()
        l.openAccount("a", bd("5")); eq(bd("5.00"), l.balance("a"))
        eq("5.00", l.balance("a").toPlainString())
    }
    check("overdraft atomic") {
        val l = Ledger()
        l.openAccount("a", bd("10.00")); l.openAccount("b")
        throws<IllegalStateException> { l.transfer("a", "b", bd("10.01")) }
        eq(bd("10.00"), l.balance("a")); eq(bd("0.00"), l.balance("b")); eq(emptyList<BigDecimal>(), l.history("a"))
    }
    check("exact balance allowed") {
        val l = Ledger()
        l.openAccount("a", bd("10.00")); l.openAccount("b")
        l.transfer("a", "b", bd("10.00"))
        eq(bd("0.00"), l.balance("a"))
    }
    check("history") {
        val l = Ledger()
        l.openAccount("a", bd("3.00")); l.openAccount("b")
        l.transfer("a", "b", bd("1.25")); l.transfer("b", "a", bd("0.05"))
        eq(listOf(bd("-1.25"), bd("0.05")), l.history("a"))
    }
    check("validation") {
        val l = Ledger()
        l.openAccount("a", bd("3.00")); l.openAccount("b")
        throws<IllegalArgumentException> { l.transfer("a", "b", bd("0.001")) }
        throws<IllegalArgumentException> { l.transfer("a", "b", bd("0")) }
        throws<IllegalArgumentException> { l.transfer("a", "b", bd("-1")) }
        throws<IllegalArgumentException> { l.openAccount("a") }
        throws<NoSuchElementException> { l.balance("zzz") }
        throws<NoSuchElementException> { l.transfer("a", "zzz", bd("1")) }
        eq(bd("3.00"), l.balance("a"))
    }
    check("split") {
        eq(listOf(bd("3.34"), bd("3.33"), bd("3.33")), Ledger.split(bd("10.00"), 3))
        eq(listOf(bd("0.01"), bd("0.01"), bd("0.00")), Ledger.split(bd("0.02"), 3))
        eq(listOf(bd("2.50"), bd("2.50")), Ledger.split(bd("5"), 2))
        throws<IllegalArgumentException> { Ledger.split(bd("1.00"), 0) }
    }
    finish()
}
