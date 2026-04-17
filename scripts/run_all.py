from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "scripts/01_eda.py",
    "scripts/02_data_preprocessing.py",
    "scripts/03_baseline_train.py",
    "scripts/04_improvement_train.py",
    "scripts/05_final_evaluation.py",
    "scripts/06_save_best_model.py",
]

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    for script in SCRIPTS:
        print(f"\n>>> Ishga tushmoqda: {script}")
        subprocess.run(
            [sys.executable, str(project_root / script)],
            check=True,
            cwd=project_root,
        )

    print("\nHammasi muvaffaqiyatli yakunlandi.")

if __name__ == "__main__":
    main()
