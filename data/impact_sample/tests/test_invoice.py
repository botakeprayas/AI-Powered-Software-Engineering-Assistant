from pricing.invoice import invoice_total, tax_only


def test_invoice_total():
    assert invoice_total([100]) == 108.0


def test_tax_only():
    assert tax_only(100) == 8.0