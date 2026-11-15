import java.math.BigDecimal
import java.math.RoundingMode

class Ledger {
    private val balances = HashMap<String, BigDecimal>()
    private val log = HashMap<String, MutableList<BigDecimal>>()

    fun openAccount(id: String, initial: BigDecimal = BigDecimal.ZERO) {
        require(id !in balances) { "account $id exists" }
        val v = money(initial)
        require(v.signum() >= 0) { "negative initial balance" }
        balances[id] = v
        log[id] = mutableListOf()
    }

    fun balance(id: String): BigDecimal = balances[id] ?: throw NoSuchElementException(id)

    fun transfer(from: String, to: String, amount: BigDecimal) {
        val a = money(amount)
        require(a.signum() > 0) { "amount must be positive" }
        val fb = balance(from)
        val tb = balance(to)
        if (fb < a) throw IllegalStateException("insufficient funds")
        balances[from] = fb - a
        balances[to] = (if (from == to) fb - a else tb) + a
        log.getValue(from).add(a.negate())
        log.getValue(to).add(a)
    }

    fun history(id: String): List<BigDecimal> = (log[id] ?: throw NoSuchElementException(id)).toList()

    companion object {
        private fun money(x: BigDecimal): BigDecimal {
            require(x.stripTrailingZeros().scale() <= 2) { "more than 2 decimals: $x" }
            return x.setScale(2, RoundingMode.UNNECESSARY)
        }

        fun split(total: BigDecimal, n: Int): List<BigDecimal> {
            require(n >= 1) { "n must be >= 1" }
            val cents = money(total).movePointRight(2).longValueExact()
            val base = cents / n
            val extra = (cents % n).toInt()
            return List(n) { i -> BigDecimal.valueOf(base + if (i < extra) 1 else 0, 2) }
        }
    }
}
