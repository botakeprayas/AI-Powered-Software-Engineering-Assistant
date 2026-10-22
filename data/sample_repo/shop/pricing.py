"""Pricing helpers."""
import functools


def log_call(func):
    """Decorator that prints the name of the function being called."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)

    return wrapper


def apply_discount(price, percent):
    """Return the price after taking a percentage off."""
    return round(price * (1 - percent / 100), 2)


@log_call
def compute_tax(amount, rate=0.08):
    """Return the sales tax owed on an amount."""
    return round(amount * rate, 2)