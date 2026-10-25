"""Order service."""
from utils.helpers import apply_tax
from . import payments


class OrderService:
    """Places orders and charges a payment gateway."""

    def __init__(self, gateway):
        self.gateway = gateway

    def subtotal(self, items):
        """Price of all items."""
        return sum(len(item) for item in items)

    def checkout(self, items):
        """Charge the gateway for the order total."""
        total = apply_tax(self.subtotal(items))
        self.gateway.charge(total)
        self.log_total(total)
        return total

    def log_total(self, total):
        """Print the total."""
        print(total)


def place_order(items, gateway=None):
    """Place an order using a default gateway when none is given."""
    service = OrderService(gateway or payments.MockGateway())
    return service.checkout(items)