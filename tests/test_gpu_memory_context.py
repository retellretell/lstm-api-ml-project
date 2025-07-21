import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pytest.importorskip('tensorflow')
from gpu_memory_optimization import GPUMemoryManager


def test_gpu_memory_context_calls_clear(monkeypatch):
    manager = GPUMemoryManager()
    calls = []

    def fake_clear():
        calls.append("called")

    monkeypatch.setattr(manager, "_clear_memory", fake_clear)

    with manager.gpu_memory_context():
        pass

    # _clear_memory should be called before and after context
    assert calls == ["called", "called"]
