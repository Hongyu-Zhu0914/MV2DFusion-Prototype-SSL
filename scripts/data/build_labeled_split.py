"""Categorized compatibility entry; executes the original root script."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dispatch import dispatch

if __name__ == '__main__':
    dispatch('build_labeled_6000_0919.py')
