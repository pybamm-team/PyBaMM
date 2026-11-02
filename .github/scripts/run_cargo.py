"""Run `cargo` with the given arguments, or skip where no Rust toolchain is installed.

The cargo pre-commit hooks go through this so that `pre-commit run --all-files` works
without Rust; CI installs Rust, so the `style` job still runs them.
"""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> int:
    cargo = shutil.which("cargo")
    if cargo is None:
        print(
            "cargo is not on PATH, so this Rust hook is skipped; install Rust "
            "(https://rustup.rs) to run it."
        )
        return 0
    return subprocess.call([cargo, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
