from pathlib import Path


def test_matdo_new_package_imports() -> None:
    from matdo_new import PACKAGE_ROOT

    assert PACKAGE_ROOT.name == "matdo_new"


def test_matdo_new_readme_exists() -> None:
    package_root = Path(__file__).resolve().parents[1]
    assert (package_root / "README.md").exists()
