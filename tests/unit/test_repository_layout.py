"""Dependency-free regression checks for the categorized publication layout."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MAPPING = json.loads((ROOT/'docs/RELOCATION_MAP.json').read_text(encoding='utf-8'))['files']
OLD_MODULES = {Path(name).stem for name in MAPPING}


class LayoutTests(unittest.TestCase):
    def test_all_root_scripts_moved(self):
        self.assertEqual(len(MAPPING), 48)
        self.assertEqual(list(ROOT.glob('*.py')), [])
        for old, new in MAPPING.items():
            self.assertFalse((ROOT/old).exists())
            self.assertTrue((ROOT/new).is_file(), new)

    def test_no_stale_imports_or_script_literals(self):
        for folder in ('scripts', 'tests', 'ssl_aug_ab_2026-10-14'):
            for path in (ROOT/folder).rglob('*.py'):
                if path == Path(__file__).resolve(): continue
                for node in ast.walk(ast.parse(path.read_bytes())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertNotIn(node.module, OLD_MODULES, str(path))
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self.assertNotIn(alias.name, OLD_MODULES, str(path))
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertNotIn(node.value, MAPPING, str(path))
                        if node.value.startswith(('scripts/', 'tests/')) and node.value.endswith('.py'):
                            self.assertTrue((ROOT/node.value).is_file(), (path,node.value))

    def test_recommended_wrappers(self):
        count = 0
        for path in (ROOT/'scripts').rglob('*.py'):
            if 'from _dispatch import dispatch' not in path.read_text(encoding='utf-8'): continue
            result = subprocess.run([sys.executable,str(path),'--show-command'],cwd=ROOT,
                                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(str(ROOT),result.stdout)
            count += 1
        self.assertEqual(count,13)

    def test_stage1_subprocess_and_stage2_audit(self):
        stage1 = (ROOT/'scripts/prototype/train_prototype_stage1_0920.py').read_text(encoding='utf-8')
        self.assertNotIn('with_name(',stage1)
        self.assertIn('scripts/evaluation/validate_prototype_subset_0924.py',stage1)
        path = ROOT/'scripts/prototype/train_prototype_stage2_20261025.py'
        tree = ast.parse(path.read_bytes())
        source_files = next(ast.literal_eval(node.value) for node in tree.body
                            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in node.targets))
        for name in source_files: self.assertTrue((ROOT/name).is_file(),name)
        self.assertIn('scripts/prototype/train_prototype_stage2_20261025.py',source_files)


if __name__ == '__main__':
    unittest.main()
