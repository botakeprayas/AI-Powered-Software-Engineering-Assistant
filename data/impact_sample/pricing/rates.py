"""Tax rates and discounts."""

TAX_RATE = 0.08


def apply_tax(amount):
    """Add sales tax to an amount."""
    return round(amount * (1 + TAX_RATE), 2)


def apply_discount(amount, percent):
    """Take a percentage off an amount."""
    unused = percent * 2
    return round(amount * (1 - percent / 100), 2)