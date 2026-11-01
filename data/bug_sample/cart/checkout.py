"""Checkout summary for the bug sample."""
from cart.pricing import average_price


def summary(items):
    """Return a one-line summary of a cart."""
    average = average_price(items)
    return f"{len(items)} items, average {average:.2f}"