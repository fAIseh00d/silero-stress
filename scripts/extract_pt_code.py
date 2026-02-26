#!/usr/bin/env python3
"""
Extract Python source code from TorchScript .pt files.

TorchScript archives contain:
- code/ - Python source files
- constants.pkl - Pickled constants
- data.pkl - Model data
- version - Format version

Usage:
    python scripts/extract_pt_code.py
"""

import zipfile
import os
from pathlib import Path

PT_FILES = [
    ('src/silero_stress/data/accentor.pt', 'scripts/extracted_pt/ru'),
    ('src/silero_stress/data/accentor-ukr.pt', 'scripts/extracted_pt/ukr'),
    ('src/silero_stress/data/accentor-bel.pt', 'scripts/extracted_pt/bel'),
]


def extract_code_from_pt(pt_path: str, output_dir: str):
    """Extract Python files from a TorchScript .pt file."""
    pt_path = Path(pt_path)
    output_dir = Path(output_dir)

    if not pt_path.exists():
        print(f"  SKIP: {pt_path} not found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # TorchScript files are ZIP archives
    with zipfile.ZipFile(pt_path, 'r') as zf:
        # Extract all .py files (both source and ts_code)
        all_files = zf.namelist()
        py_files = [f for f in all_files if f.endswith('.py')]

        print(f"  Found {len(py_files)} Python files:")

        for name in sorted(py_files):
            print(f"    - {name}")

            # Read content
            content = zf.read(name).decode('utf-8')

            # Simplify path - extract from archive prefix
            # e.g., "silero_stress-v1.4-public/models/model.py" -> "models/model.py"
            parts = name.split('/')
            if len(parts) > 1:
                # Find first "models" or "custom_tokenizers" or keep ts_code structure
                simplified = '/'.join(parts[1:])  # Skip first component (archive name)
            else:
                simplified = name

            out_path = output_dir / simplified
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Write
            with open(out_path, 'w') as f:
                f.write(content)

        print(f"  Extracted to: {output_dir}")


def main():
    print("Extracting Python code from TorchScript .pt files\n")

    for pt_path, output_dir in PT_FILES:
        print(f"Processing: {pt_path}")
        extract_code_from_pt(pt_path, output_dir)
        print()

    print("Done! Extracted code is in scripts/extracted_pt/")


if __name__ == '__main__':
    main()
