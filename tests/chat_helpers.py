"""Builders shared by the Phase 3 tests."""

from __future__ import annotations

from core.chat.models import Evidence
from core.semantic.models import SearchResult, Symbol

CART_CODE = (
    "    async def checkout(self, payment_gateway):\n"
    '        """Charge the customer and empty the cart."""\n'
    "        amount = self.total()\n"
    "        await payment_gateway.charge(amount)\n"
    "        self.items = []\n"
    "        return amount"
)
TOTAL_CODE = "    def total(self):\n        return sum(p * q for _, p, q in self.items)"


def make_symbol(
    file="shop/cart.py",
    qualname="ShoppingCart.checkout",
    kind="async_method",
    start=22,
    end=27,
    code=CART_CODE,
    doc="Charge the customer and empty the cart.",
):
    return Symbol(
        key=f"{file}::{qualname}", file=file, kind=kind, name=qualname.split(".")[-1],
        qualname=qualname, start_line=start, end_line=end, docstring=doc, code=code,
    )


def make_total_symbol():
    return make_symbol(qualname="ShoppingCart.total", kind="method", start=18, end=20,
                       code=TOTAL_CODE, doc="Return the total.")


def make_result(symbol, semantic=0.55, kw_rank=1, kw_score=12.0):
    return SearchResult(
        symbol=symbol, score=0.03,
        semantic_rank=1 if semantic is not None else None, semantic_score=semantic,
        keyword_rank=kw_rank, keyword_score=kw_score if kw_rank is not None else None,
    )


def make_evidence(count=2):
    symbols = [make_symbol(), make_total_symbol()][:count]
    return [
        Evidence(id=f"E{i + 1}", symbol=s, code=s.code, code_truncated=False,
                 semantic_score=0.5, keyword_rank=i + 1, keyword_score=9.0)
        for i, s in enumerate(symbols)
    ]


class FakeRetriever:
    """Stands in for CodeSearchEngine: returns canned results and records queries."""

    def __init__(self, results):
        self.results = list(results)
        self.queries: list[tuple] = []

    def search(self, query, mode="hybrid", top_k=5):
        self.queries.append((query, mode, top_k))
        return self.results[:top_k]