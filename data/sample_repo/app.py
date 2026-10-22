"""Entry point of the sample application."""
from shop.cart import ShoppingCart


def main():
    """Create a cart, add two products and print the total."""
    cart = ShoppingCart()
    cart.add_item("book", 12.5, 2)
    cart.add_item("pen", 1.2)
    print(cart.total())


async def fetch_data(url):
    """Download data from a URL (placeholder, no real network call)."""
    return {"url": url}


if __name__ == "__main__":
    main()