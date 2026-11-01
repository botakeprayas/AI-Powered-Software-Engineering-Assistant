"""Pricing helpers for the bug sample."""


def average_price(items):
    """Return the average price of the items in a cart."""
    total = sum(price for _, price in items)
    return total / len(items)