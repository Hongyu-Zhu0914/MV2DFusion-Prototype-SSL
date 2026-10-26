"""Categorized entry; executes the implementation at its relocated path."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _dispatch import dispatch

if __name__ == '__main__':
    dispatch('scripts/evaluation/export_proto_val_iter6000_0929.py')
