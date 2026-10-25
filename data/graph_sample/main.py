"""Entry point of the graph sample."""
from orders.service import OrderService, place_order
from orders import payments
from utils.helpers import format_money as money


def run():
    """Run a demo order."""
    service = OrderService(payments.StripeGateway())
    total = service.checkout(["book", "pen"])
    print(money(total))
    place_order(["lamp"])


if __name__ == "__main__":
    run()