from pricing.rates import apply_discount, apply_tax


def test_apply_tax():
    assert apply_tax(100) == 108.0


def test_apply_discount():
    assert apply_discount(100, 10) == 90.0