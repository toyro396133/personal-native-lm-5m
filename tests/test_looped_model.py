import pytest
import torch

from config import ModelConfig
from model import PersonalNativeLM
from looped_model import LoopedPersonalNativeLM


def small_cfg():
    return ModelConfig(vocab_size=48, d_model=32, n_heads=4,
                       n_layers=2, d_ff=64, max_seq_len=16,
                       controller_dim=16)


def test_one_loop_matches_original_including_personal_conditioning():
    torch.manual_seed(42)
    cfg = small_cfg()
    base = PersonalNativeLM(cfg).eval()
    looped = LoopedPersonalNativeLM(cfg, loops=1).eval()
    looped.load_state_dict(base.state_dict(), strict=True)
    x = torch.randint(0, cfg.vocab_size, (2, 9))
    y = torch.randint(0, cfg.vocab_size, (2, 9))
    personal = torch.randn(2, cfg.state_dim)
    for condition in (None, x[:, :4]):
        a, la = base(x, personal, y, condition_ids=condition)
        b, lb = looped(x, personal, y, condition_ids=condition)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(la, lb, atol=0, rtol=0)


@pytest.mark.parametrize("loops", [2, 4, 8])
@pytest.mark.parametrize("inject", [0.0, 0.1])
def test_shared_weights_and_finite_gradients(loops, inject):
    torch.manual_seed(9)
    model = LoopedPersonalNativeLM(small_cfg(), loops=loops, input_injection=inject)
    x = torch.randint(0, 48, (2, 8))
    y = torch.randint(0, 48, (2, 8))
    personal = torch.zeros(2, 228)
    _, loss = model(x, personal, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_params_same_across_loops():
    cfg = small_cfg()
    counts = [sum(p.numel() for p in LoopedPersonalNativeLM(cfg, loops=n).parameters())
              for n in (1, 2, 4, 8)]
    assert len(set(counts)) == 1


def test_invalid_options():
    with pytest.raises(ValueError):
        LoopedPersonalNativeLM(small_cfg(), loops=0)
    with pytest.raises(ValueError):
        LoopedPersonalNativeLM(small_cfg(), loops=2, input_injection=1.1)
