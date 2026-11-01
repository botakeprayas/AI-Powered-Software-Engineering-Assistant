from shop.checkout import checkout


def test_checkout():
    assert checkout([100]) == 108.0