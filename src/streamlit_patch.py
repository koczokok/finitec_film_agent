import types
import torch

# Fake __path__ to prevent Streamlit from introspecting torch.classes
class DummyPath:
    _path = []

torch.classes.__path__ = DummyPath()
