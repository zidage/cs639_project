#!/usr/bin/env python3
"""Create a Python virtual environment for this repository.

Features:
- Optional recreation of an existing venv.
- Installs requirements.txt.
- Writes setup logs to experiments/setup_env_logs/.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple


def run_command(command: Sequence[str], cwd: Path, log_file: Path, step_name: str) -> Tuple[int, float]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    with log_file.open("a", encoding="utf-8") as fh:
        fh.write("=" * 80 + "\n")
        fh.write(f"[{datetime.now().isoformat(timespec='seconds')}] Step: {step_name}\n")
        fh.write(f"CWD: {cwd}\n")
        fh.write(f"Command: {' '.join(command)}\n\n")
        fh.flush()

        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            fh.write(line)
            print(f"[{step_name}] {line.rstrip()}", flush=True)

        rc = process.wait()
        fh.write(f"\nReturn code: {rc}\n")

    return rc, time.time() - start


def resolve_venv_python(venv_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create venv and install dependencies")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[2]),
        help="Repository root path",
    )
    parser.add_argument(
        "--venv-dir",
        type=str,
        default=".venv-antidote",
        help="Virtual environment directory (relative to project root by default)",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable used to create the venv",
    )
    parser.add_argument(
        "--requirements",
        type=str,
        default="requirements.txt",
        help="Requirements file path, relative to project root unless absolute",
    )
    parser.add_argument(
        "--extra-packages",
        type=str,
        default="",
        help="Optional comma-separated extra packages to install after requirements",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        default=False,
        help="Delete and recreate the venv directory if it already exists",
    )
    parser.add_argument(
        "--skip-requirements",
        action="store_true",
        default=False,
        help="Create venv only, skip requirements installation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    root = Path(args.project_root).resolve()
    if not root.exists():
        print(f"[error] project root does not exist: {root}")
        return 2

    venv_dir = Path(args.venv_dir)
    if not venv_dir.is_absolute():
        venv_dir = (root / venv_dir).resolve()

    requirements_path = Path(args.requirements)
    if not requirements_path.is_absolute():
        requirements_path = (root / requirements_path).resolve()

    log_dir = root / "experiments" / "setup_env_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"setup_venv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print("=" * 88)
    print("Python environment setup")
    print(f"Project root : {root}")
    print(f"Venv dir     : {venv_dir}")
    print(f"Python bin   : {args.python_bin}")
    print(f"Log file     : {log_file}")
    print("=" * 88)

    if venv_dir.exists() and args.recreate:
        print(f"[setup] Removing existing venv: {venv_dir}")
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print("[setup] Creating virtual environment...")
        rc, elapsed = run_command(
            [args.python_bin, "-m", "venv", str(venv_dir)],
            cwd=root,
            log_file=log_file,
            step_name="create-venv",
        )
        if rc != 0:
            print(f"[error] Failed to create venv (rc={rc})")
            return rc
        print(f"[setup] venv created in {elapsed:.1f}s")
    else:
        print(f"[setup] Reusing existing venv: {venv_dir}")

    venv_python = resolve_venv_python(venv_dir)
    if not venv_python.exists():
        print(f"[error] venv python not found: {venv_python}")
        return 3

    print("[setup] Upgrading pip/setuptools/wheel...")
    rc, _ = run_command(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=root,
        log_file=log_file,
        step_name="upgrade-pip",
    )
    if rc != 0:
        print(f"[error] Failed during pip upgrade (rc={rc})")
        return rc

    if not args.skip_requirements:
        if not requirements_path.exists():
            print(f"[error] requirements.txt not found: {requirements_path}")
            return 4
        print("[setup] Installing requirements...")
        rc, _ = run_command(
            [str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)],
            cwd=root,
            log_file=log_file,
            step_name="install-requirements",
        )
        if rc != 0:
            print(f"[error] Failed to install requirements (rc={rc})")
            return rc

    extra = [p.strip() for p in args.extra_packages.split(",") if p.strip()]
    if extra:
        print(f"[setup] Installing extra packages: {extra}")
        cmd: List[str] = [str(venv_python), "-m", "pip", "install"] + extra
        rc, _ = run_command(
            cmd,
            cwd=root,
            log_file=log_file,
            step_name="install-extra-packages",
        )
        if rc != 0:
            print(f"[error] Failed to install extra packages (rc={rc})")
            return rc

    print("[setup] Running quick import sanity check...")
    sanity_cmd = [
        str(venv_python),
        "-c",
        (
            "import torch, transformers, datasets, peft; "
            "print('sanity-ok', torch.__version__, transformers.__version__)"
        ),
    ]
    rc, _ = run_command(
        sanity_cmd,
        cwd=root,
        log_file=log_file,
        step_name="sanity-check",
    )
    if rc != 0:
        print(f"[error] Sanity check failed (rc={rc})")
        return rc

    if platform.system().lower().startswith("win"):
        activation_hint = str(venv_dir / "Scripts" / "activate")
    else:
        activation_hint = f"source {venv_dir / 'bin' / 'activate'}"

    print("=" * 88)
    print("Environment setup completed successfully.")
    print(f"Activate with: {activation_hint}")
    print(f"Then run experiments with: {venv_python} script/automation/run_lisa_grid.py")
    print(f"Setup log file: {log_file}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
