"""Shopping cart logic for the sample shop."""


class ShoppingCart:
    """Holds the products a shopper wants to buy."""

    def __init__(self):
        self.items = []

    def add_item(self, name, price, quantity=1):
        """Add a product to the cart."""
        self.items.append((name, price, quantity))

    def remove_item(self, name):
        """Remove every line item with the given product name."""
        self.items = [item for item in self.items if item[0] != name]

    def total(self):
        """Return the sum of price times quantity for all items."""
        return sum(price * quantity for _, price, quantity in self.items)

    async def checkout(self, payment_gateway):
        """Charge the customer and empty the cart."""
        amount = self.total()
        await payment_gateway.charge(amount)
        self.items = []
        return amount