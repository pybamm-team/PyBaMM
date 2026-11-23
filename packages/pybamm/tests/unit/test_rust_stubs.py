"""``pybamm/rust/_core.pyi`` stays in sync with the built extension."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pybamm.rust


class TestRustStubs:
    def test_stub_matches_the_built_extension(self, tmp_path):
        stub = Path(pybamm.rust.__file__).parent / "_core.pyi"
        assert stub.is_file(), f"missing stub file {stub}"

        # Empty ancestor stubs keep mypy from analysing the whole pybamm package.
        stub_root = tmp_path / "stubs"
        package = stub_root / "pybamm" / "rust"
        package.mkdir(parents=True)
        (stub_root / "pybamm" / "__init__.pyi").touch()
        (package / "__init__.pyi").touch()
        shutil.copyfile(stub, package / "_core.pyi")
        # scipy ships no stubs; stubtest checks names and arities, not annotations.
        config = tmp_path / "mypy.ini"
        config.write_text("[mypy]\n[mypy-scipy.*]\nignore_missing_imports = True\n")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy.stubtest",
                "pybamm.rust._core",
                "--mypy-config-file",
                str(config),
            ],
            env={**os.environ, "MYPYPATH": str(stub_root)},
            # tmp_path, so mypy's cache directory never lands in the repo.
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, (
            f"stubtest found mismatches between pybamm/rust/_core.pyi and the "
            f"built extension:\n{result.stdout}\n{result.stderr}"
        )
