"""Preserve the root entrypoint's __file__, imports, and working directory."""
from pathlib import Path
import os
import runpy
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]


def dispatch(relative_script):
    target = (ROOT / relative_script).resolve()
    if ROOT not in target.parents or not target.is_file():
        raise SystemExit('Invalid or missing entrypoint: ' + relative_script)
    if sys.argv[1:] == ['--show-command']:
        print('cd ' + shlex.quote(str(ROOT)))
        print(shlex.quote(sys.executable) + ' ' + shlex.quote(str(target)))
        return
    os.chdir(str(ROOT))
    sys.path.insert(0, str(ROOT))
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name='__main__')
