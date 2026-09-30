"""Check excluded local assets by path and size, without importing ML packages."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inventory', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = args.inventory or root / 'docs/local-assets.json'
    rows = json.loads(inventory.read_text())['files']
    failures = []
    for row in rows:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Inventory path must stay within the workspace')
        path = root / relative
        if row['kind'] == 'symlink':
            if not path.is_symlink() or os.readlink(path) != row['target']:
                failures.append(dict(path=str(relative), reason='symlink_missing_or_changed'))
            elif not path.exists():
                failures.append(dict(path=str(relative), reason='symlink_target_missing'))
        elif not path.is_file():
            failures.append(dict(path=str(relative), reason='missing'))
        elif path.stat().st_size != row['bytes']:
            failures.append(dict(path=str(relative), reason='size_differs'))
    print(json.dumps(dict(checked=len(rows), failures=len(failures), first_failures=failures[:20],
        note='Size checks do not replace experiment SHA-256 validation.'), indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
