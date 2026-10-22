from core.repo.tree import render_tree


def test_tree_lists_folders_before_files_and_shows_notes():
    text = render_tree(
        ["app.py", "shop/cart.py", "shop/pricing.py", "utils/validation.py"],
        {"shop/cart.py": "[6 symbols]"},
        root_name="repo",
    )
    lines = text.splitlines()
    assert lines[0] == "repo/"
    assert lines.index(next(line for line in lines if line.endswith("shop/"))) < lines.index(
        next(line for line in lines if line.endswith("app.py"))
    )
    assert any(line.endswith("cart.py  [6 symbols]") for line in lines)


def test_empty_input_gives_just_the_root():
    assert render_tree([], root_name="x") == "x/"