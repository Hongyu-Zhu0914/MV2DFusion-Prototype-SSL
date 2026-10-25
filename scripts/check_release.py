"""Dependency-free source release checks. Never import/train the detector."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {'.pth', '.pt', '.pkl', '.pickle', '.npy', '.npz', '.ckpt', '.pem', '.key'}
SKIP = {'.git', '__pycache__', '.pytest_cache'}


def main():
    errors = []
    checked = 0
    patterns = [re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
                re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b')]
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if any(part in SKIP for part in relative.parts) or not path.is_file():
            continue
        if path.is_symlink():
            errors.append(str(relative) + ': source release must not contain symlinks')
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(str(relative) + ': forbidden artifact or credential extension')
        if path.suffix == '.py':
            checked += 1
            try:
                ast.parse(path.read_bytes(), filename=str(relative))
            except SyntaxError as exc:
                errors.append(str(relative) + ': ' + str(exc))
        if path.stat().st_size < 4 * 1024 * 1024:
            text = path.read_text(encoding='utf-8', errors='replace')
            if any(pattern.search(text) for pattern in patterns):
                errors.append(str(relative) + ': potential credential pattern; value not printed')
    manifest = json.loads((ROOT/'docs/SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path = ROOT / row['published_path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append(row['published_path'] + ': source checksum mismatch')
    for requirement in ('README.md', '.gitignore', 'UPSTREAM_AND_LICENSE.md', 'docs/CODE_MAP.md'):
        if not (ROOT/requirement).is_file():
            errors.append(requirement + ': missing release documentation')
    print(json.dumps({'python_files': checked, 'source_files_verified': len(manifest['files']),
                      'status': 'FAIL' if errors else 'PASS', 'errors': errors}, ensure_ascii=False, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
