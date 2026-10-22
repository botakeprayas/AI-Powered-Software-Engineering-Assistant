import pytest
from repo_helpers import SAMPLE_REPO, build_hazard_repo

from core.repo.scanner import scan_repository, sensitive_reason


@pytest.fixture
def hazard_scan(tmp_path):
    return scan_repository(build_hazard_repo(tmp_path / "repo"))


def skipped_map(scan):
    return {item.path: item for item in scan.skipped}


def kept_paths(scan):
    return {f.path for f in scan.files}


def test_sample_repo_files_are_kept():
    paths = kept_paths(scan_repository(SAMPLE_REPO))
    expected = {
        "app.py", "shop/cart.py", "shop/pricing.py", "utils/validation.py",
        "broken_syntax.py", "empty_module.py", "notes.txt",
    }
    assert expected <= paths


@pytest.mark.parametrize(
    "path",
    [".env", ".env.local", "keys/server.pem", "id_rsa", "config/credentials.json", "secrets.yaml"],
)
def test_sensitive_files_are_skipped(hazard_scan, path):
    assert path in skipped_map(hazard_scan)
    assert "sensitive" in skipped_map(hazard_scan)[path].reason
    assert path not in kept_paths(hazard_scan)


def test_env_example_template_is_kept(hazard_scan):
    assert ".env.example" in kept_paths(hazard_scan)


@pytest.mark.parametrize(
    "folder", [".venv", "my_env", "node_modules", "build", "dist", "__pycache__", ".git"]
)
def test_environment_build_and_cache_dirs_are_skipped(hazard_scan, folder):
    item = skipped_map(hazard_scan)[folder]
    assert item.kind == "dir"
    assert not any(path.startswith(folder + "/") for path in kept_paths(hazard_scan))


def test_venv_with_custom_name_is_found_by_pyvenv_cfg(hazard_scan):
    assert "pyvenv.cfg" in skipped_map(hazard_scan)["my_env"].reason


@pytest.mark.parametrize("path", ["schema_pb2.py", "auto_gen.py"])
def test_generated_files_are_skipped(hazard_scan, path):
    assert "generated" in skipped_map(hazard_scan)[path].reason


def test_binary_files_are_skipped(hazard_scan):
    assert "binary" in skipped_map(hazard_scan)["logo.png"].reason


def test_scanner_keeps_unreadable_python_for_the_extractor(hazard_scan):
    paths = kept_paths(hazard_scan)
    assert {"bad_encoding.py", "binary_like.py", "broken_syntax.py", "empty_module.py"} <= paths


def test_oversized_file_is_skipped(tmp_path):
    scan = scan_repository(build_hazard_repo(tmp_path / "repo"), max_file_bytes=50)
    assert "too large" in skipped_map(scan)["app.py"].reason


def test_file_limit_stops_the_scan(tmp_path):
    scan = scan_repository(build_hazard_repo(tmp_path / "repo"), max_files=3)
    assert scan.limit_reached
    assert len(scan.files) == 3


def test_missing_folder_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_repository(tmp_path / "does_not_exist")


def test_symlinked_file_is_not_followed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "real.py").write_text("def f():\n    pass\n", encoding="utf-8")
    try:
        (root / "link.py").symlink_to(root / "real.py")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not allowed on this machine (this test did NOT run)")
    scan = scan_repository(root)
    assert "link.py" in skipped_map(scan)
    assert "real.py" in kept_paths(scan)


@pytest.mark.parametrize(
    ("name", "sensitive"),
    [
        (".env", True), (".env.example", False), ("id_rsa", True), ("id_rsa.pub", False),
        ("client_secret.json", True), ("Credentials.JSON", True), ("main.py", False),
        ("README.md", False),
    ],
)
def test_sensitive_name_rules(name, sensitive):
    assert (sensitive_reason(name) is not None) == sensitive