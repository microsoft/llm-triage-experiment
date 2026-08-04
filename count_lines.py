from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_EXTENSIONS = {".py", ".r", ".js", ".ts", ".ps1", ".sh"}
LIMIT = 5_000


def main() -> int:
    files = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS
    )
    total = 0
    for path in files:
        count = len(path.read_text(encoding="utf-8").splitlines())
        total += count
        print(f"{count:5d}  {path.relative_to(ROOT)}")
    print(f"\nTotal source lines: {total:,} / {LIMIT:,}")
    return 0 if total < LIMIT else 1


if __name__ == "__main__":
    raise SystemExit(main())
