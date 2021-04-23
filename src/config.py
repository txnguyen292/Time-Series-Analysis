from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent / "data"
    reports = file_dir.parent / "reports"
    notebooks = file_dir.parent / "notebooks"

if __name__ == "__main__":
    pass