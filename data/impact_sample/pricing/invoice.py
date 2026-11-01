"""Invoice totals."""
from pricing.rates import apply_discount, apply_tax


def invoice_total(items, percent=0):
    """Total of all items after discount and tax."""
    subtotal = sum(items)
    return apply_tax(apply_discount(subtotal, percent))


def tax_only(amount):
    """Tax owed on an amount."""
    return apply_tax(amount) - amount