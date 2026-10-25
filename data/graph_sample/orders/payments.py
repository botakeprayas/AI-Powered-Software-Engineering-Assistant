"""Payment gateways."""


class StripeGateway:
    """Pretend to charge a card."""

    def charge(self, amount):
        """Charge an amount and return a receipt id."""
        return f"stripe-{amount}"


class MockGateway:
    """A gateway for tests."""

    def charge(self, amount):
        """Pretend to charge an amount."""
        return "mock"