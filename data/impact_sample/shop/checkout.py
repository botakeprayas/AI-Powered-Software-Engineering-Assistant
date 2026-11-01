"""Checkout flow."""
from pricing.invoice import invoice_total


def checkout(cart, percent=0):
    """Charge the customer for a cart."""
    return invoice_total(cart, percent)