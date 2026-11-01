from cart.pricing import average_price


def test_average_of_two_items():
    assert average_price([("a", 2.0), ("b", 4.0)]) == 3.0