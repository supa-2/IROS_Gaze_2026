#!/usr/bin/env python3
"""
Setup script to install sam2 as a Python package
Run this from the project root directory: python setup_sam2.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    sam2_dir = project_root / "sam2"

    if not sam2_dir.exists():
        print(f"[!] Error: sam2 directory not found at {sam2_dir}")
        return 1

    # Create __init__.py in sam2 if it doesn't exist
    init_file = sam2_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('"SAM2 module wrapper"\n')
        print(f"[+] Created {init_file}")

    # Install sam2 as editable package
    print(f"\n[*] Installing sam2 as editable package...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("[+] sam2 installed successfully!")
            print("\n[*] Verifying installation...")
            # Try to import sam2
            result = subprocess.run(
                [sys.executable, "-c", "import sam2; print(sam2.__file__)"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"[+] sam2 is installed at: {result.stdout.strip()}")
                return 0
            else:
                print(f"[!] Verification failed: {result.stderr}")
                return 1
        else:
            print(f"[!] Installation failed: {result.stderr}")
            return 1
    except Exception as e:
        print(f"[!] Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
