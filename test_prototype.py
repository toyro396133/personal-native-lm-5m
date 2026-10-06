import torch
from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState

def test_shapes():
    cfg = ModelConfig()
    model = PersonalNativeLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    s = torch.stack([
        PersonalState("a").flatten(),
        PersonalState("b").flatten(),
    ])
    logits, loss = model(x, s, x)
    assert logits.shape == (2, 24, cfg.vocab_size)
    assert loss.ndim == 0

def test_state_changes_computation():
    cfg = ModelConfig()
    model = PersonalNativeLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))

    a = PersonalState("a")
    b = PersonalState("b")
    b.core.vector[0] = 1.0
    b.core.confidence = 1.0

    with torch.no_grad():
        la, _ = model(x, a.flatten().unsqueeze(0))
        lb, _ = model(x, b.flatten().unsqueeze(0))

    assert not torch.allclose(la, lb)
