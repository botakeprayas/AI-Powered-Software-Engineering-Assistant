import pytest
from impact_helpers import CART, make_diff, make_new_diff, replace_line

from core.impact.diff_parser import ADDED, DELETED, MODIFIED, RENAMED, Block, parse_diff

MODIFY = '''diff --git a/shop/cart.py b/shop/cart.py
index 1111111..2222222 100644
--- a/shop/cart.py
+++ b/shop/cart.py
@@ -10,6 +10,7 @@ class ShoppingCart:
     def remove_item(self, name):
         """Remove every line item."""
-        self.items = [i for i in self.items if i[0] != name]
+        kept = [i for i in self.items if i[0] != name]
+        self.items = kept

     def total(self):
         """Return the sum."""
@@ -40,0 +41,2 @@ def helper():
+    extra = 1
+    return extra
'''
NEW_FILE = """diff --git a/pkg/new.py b/pkg/new.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/pkg/new.py
@@ -0,0 +1,3 @@
+def added():
+    return 1
+
"""
DELETED_FILE = """diff --git a/old/gone.py b/old/gone.py
deleted file mode 100644
index 4444444..0000000
--- a/old/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def gone():
-    return 1
"""
RENAME_ONLY = """diff --git a/pkg/old_name.py b/pkg/new_name.py
similarity index 100%
rename from pkg/old_name.py
rename to pkg/new_name.py
"""
RENAME_EDIT = """diff --git a/pkg/old_name.py b/pkg/new_name.py
similarity index 90%
rename from pkg/old_name.py
rename to pkg/new_name.py
index 1111111..2222222 100644
--- a/pkg/old_name.py
+++ b/pkg/new_name.py
@@ -1 +1 @@
-a = 1
+a = 2
"""
PLAIN = (
    "--- a/utils.py\t2024-01-01 00:00:00\n"
    "+++ b/utils.py\t2024-01-02 00:00:00\n"
    "@@ -1,2 +1,2 @@\n"
    "-x = 1\n"
    "+x = 2\n"
    " y = 3\n"
    "--- a/other.py\t2024-01-01 00:00:00\n"
    "+++ b/other.py\t2024-01-02 00:00:00\n"
    "@@ -1 +1 @@\n"
    "-p = 1\n"
    "+p = 2\n"
)
BINARY = (
    "diff --git a/img/logo.png b/img/logo.png\n"
    "index 1111111..2222222 100644\n"
    "Binary files a/img/logo.png and b/img/logo.png differ\n"
)
NO_NEWLINE = (
    "diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
    "@@ -1 +1 @@\n-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
)
HEADER_LOOKALIKE = (
    "diff --git a/q.py b/q.py\n--- a/q.py\n+++ b/q.py\n"
    "@@ -1,2 +1,1 @@\n--- comment\n keep\n"
)
QUOTED = (
    'diff --git "a/x y.py" "b/x y.py"\n--- "a/x y.py"\n+++ "b/x y.py"\n'
    "@@ -1 +1 @@\n-a\n+b\n"
)
SPACES = (
    "diff --git a/my file.py b/my file.py\n--- a/my file.py\n+++ b/my file.py\n"
    "@@ -1 +1 @@\n-a\n+b\n"
)


def test_a_modified_file_gives_hunks_and_change_blocks():
    parsed = parse_diff(MODIFY)
    assert parsed.warnings == []
    (file_diff,) = parsed.files
    assert (file_diff.old_path, file_diff.new_path) == ("shop/cart.py", "shop/cart.py")
    assert file_diff.status == MODIFIED
    first, second = file_diff.hunks
    assert first.heading == "class ShoppingCart:"
    assert first.blocks == [
        Block(
            12,
            ("        self.items = [i for i in self.items if i[0] != name]",),
            12,
            (
                "        kept = [i for i in self.items if i[0] != name]",
                "        self.items = kept",
            ),
        )
    ]
    assert (13, "") in first.old_text  # the blank context line lost its leading space
    assert (16, '        """Return the sum."""') in first.new_text
    assert second.blocks == [Block(41, (), 41, ("    extra = 1", "    return extra"))]
    assert (file_diff.lines_added, file_diff.lines_removed) == (4, 1)


def test_a_new_file_a_deleted_file_and_renames():
    (added,) = parse_diff(NEW_FILE).files
    assert (added.status, added.old_path, added.new_path) == (ADDED, None, "pkg/new.py")
    assert added.hunks[0].blocks == [Block(1, (), 1, ("def added():", "    return 1", ""))]
    (deleted,) = parse_diff(DELETED_FILE).files
    assert (deleted.status, deleted.old_path, deleted.new_path) == (DELETED, "old/gone.py", None)
    assert deleted.lines_removed == 2
    (renamed,) = parse_diff(RENAME_ONLY).files
    assert renamed.status == RENAMED and renamed.similarity == 100
    assert (renamed.old_path, renamed.new_path) == ("pkg/old_name.py", "pkg/new_name.py")
    assert renamed.hunks == []
    (edited,) = parse_diff(RENAME_EDIT).files
    assert edited.status == RENAMED and len(edited.hunks) == 1


def test_a_plain_diff_without_git_headers_gives_one_file_per_header_pair():
    parsed = parse_diff(PLAIN)
    assert [f.new_path for f in parsed.files] == ["utils.py", "other.py"]
    assert [len(f.hunks) for f in parsed.files] == [1, 1]


def test_binary_files_are_flagged():
    (file_diff,) = parse_diff(BINARY).files
    assert file_diff.binary and file_diff.new_path == "img/logo.png"


def test_no_newline_markers_do_not_break_the_counts():
    (file_diff,) = parse_diff(NO_NEWLINE).files
    assert file_diff.hunks[0].blocks == [Block(1, ("old",), 1, ("new",))]
    assert not file_diff.hunks[0].truncated


def test_a_removed_line_that_looks_like_a_file_header_stays_inside_its_hunk():
    parsed = parse_diff(HEADER_LOOKALIKE)
    (file_diff,) = parsed.files
    assert file_diff.hunks[0].blocks == [Block(1, ("-- comment",), 1, ())]
    assert parsed.warnings == []


@pytest.mark.parametrize(("text", "path"), [(QUOTED, "x y.py"), (SPACES, "my file.py")])
def test_paths_with_spaces(text, path):
    (file_diff,) = parse_diff(text).files
    assert (file_diff.old_path, file_diff.new_path) == (path, path)


def test_crlf_line_endings_are_accepted():
    parsed = parse_diff(MODIFY.replace("\n", "\r\n"))
    assert len(parsed.files[0].hunks) == 2 and parsed.warnings == []


def test_a_cut_off_hunk_is_flagged_but_still_read():
    text = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,5 +1,5 @@\n-a\n+b\n"
    parsed = parse_diff(text)
    assert parsed.files[0].hunks[0].truncated
    assert any("incomplete" in warning for warning in parsed.warnings)


def test_a_mode_change_has_no_hunks():
    text = "diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n"
    (file_diff,) = parse_diff(text).files
    assert file_diff.hunks == [] and file_diff.status == MODIFIED


@pytest.mark.parametrize("text", ["", "   \n", "hello world", None, 5, b"bytes"])
def test_input_that_is_not_a_diff_gives_a_warning_not_an_exception(text):
    parsed = parse_diff(text)
    assert parsed.files == [] and parsed.warnings


def test_a_hunk_before_any_file_header_is_ignored_with_a_warning():
    parsed = parse_diff("@@ -1 +1 @@\n-a\n+b\n")
    assert parsed.files == []
    assert any("before any file header" in warning for warning in parsed.warnings)


def test_a_generated_diff_round_trips_through_the_parser():
    new = replace_line(CART, 10, "        self.items.insert(0, item)")
    (file_diff,) = parse_diff(make_diff("shop/cart.py", CART, new)).files
    (hunk,) = file_diff.hunks
    assert not hunk.truncated
    assert (file_diff.lines_added, file_diff.lines_removed) == (1, 1)
    assert hunk.blocks[0].old_start == hunk.blocks[0].new_start == 10
    (created,) = parse_diff(make_new_diff("pkg/x.py", "a = 1\nb = 2\n")).files
    assert created.status == ADDED and created.lines_added == 2