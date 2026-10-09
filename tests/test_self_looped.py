import torch
from config import ModelConfig
from experiments.looped_5m.self_looped_model import SelfLoopedLM


def cfg():
    return ModelConfig(vocab_size=48, d_model=32, n_heads=4, n_layers=2,
                       d_ff=64, max_seq_len=16, controller_dim=16)


def test_anchor_capacity_matched():
    a = SelfLoopedLM(cfg(), loops=2, self_mode="anchor")
    b = SelfLoopedLM(cfg(), loops=2, self_mode="capacity")
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_interventions_change_and_gradients_work():
    torch.manual_seed(42)
    m = SelfLoopedLM(cfg(), loops=2, self_mode="anchor")
    x = torch.randint(0, 48, (2, 8))
    y = torch.randint(0, 48, (2, 8))
    personal = torch.zeros(2, 228)
    original, loss = m(x, personal, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.self_vector.grad is not None
    assert torch.isfinite(m.self_vector.grad).all()
    with torch.no_grad():
        m.anchor_intervention = "zero"
        zero, _ = m(x, personal)
        m.anchor_intervention = "shuffle"
        shuffled, _ = m(x, personal)
    assert not torch.allclose(original, zero)
    assert not torch.allclose(original, shuffled)
