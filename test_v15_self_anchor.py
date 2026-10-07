import torch

from config import ModelConfig
from model import PersonalNativeLM
from model_self_anchor import SelfAnchoredPersonalNativeLM


def tiny_cfg():
    return ModelConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=32,
        state_dim=228,
        controller_dim=16,
    )


def test_self_anchor_is_exact_noop_at_initialization():
    torch.manual_seed(123)
    base = PersonalNativeLM(tiny_cfg())
    ids = torch.randint(0, 64, (2, 12))
    state = torch.zeros(2, 228)

    with torch.no_grad():
        expected, _ = base(ids, state)

    anchored = SelfAnchoredPersonalNativeLM(base, rank=4)
    with torch.no_grad():
        actual, _ = anchored(ids, state)

    assert torch.equal(expected, actual)
    assert all(layer["up_norm"] == 0.0 for layer in anchored.self_diagnostics()["layers"])


def test_self_anchor_uses_one_shared_identity_point():
    torch.manual_seed(321)
    anchored = SelfAnchoredPersonalNativeLM(PersonalNativeLM(tiny_cfg()), rank=4)
    assert tuple(anchored.self_anchor.shape) == (32,)
    assert len(anchored.self_adapters) == 2
    # There is exactly one anchor parameter; layer adapters reference it at runtime.
    anchor_names = [name for name, _ in anchored.named_parameters() if name == "self_anchor"]
    assert anchor_names == ["self_anchor"]
