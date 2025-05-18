import torch

class DummyPath:
    _path = []

torch.classes.__path__ = DummyPath()
