"""Formatting helpers."""


def format_money(amount):
    """Format a number as dollars."""
    return f"${amount:,.2f}"


def apply_tax(amount, rate=0.08):
    """Add sales tax to an amount."""
    return round(amount * (1 + rate), 2)


def apply_callback(callback, value):
    """Call a function passed in by the caller."""
    return callback(value)