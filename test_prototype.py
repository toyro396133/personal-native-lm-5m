import torch
from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState
from tokenizer import ByteTokenizer

TOK = ByteTokenizer()


def _condition(text):
    return torch.tensor([TOK.encode(text, bos=False, eos=False)], dtype=torch.long)


def test_shapes():
    cfg = ModelConfig.byte_prototype()
    model = PersonalNativeLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    q = torch.randint(0, cfg.vocab_size, (2, 12))
    s = torch.stack([PersonalState("a").flatten(), PersonalState("b").flatten()])
    logits, loss = model(x, s, x, condition_ids=q)
    assert logits.shape == (2, 24, cfg.vocab_size)
    assert loss.ndim == 0


def test_personal_state_changes_runtime_computation():
    cfg = ModelConfig.byte_prototype()
    model = PersonalNativeLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    q = _condition("בחר אפשרות למשימה")
    a = PersonalState("a")
    b = PersonalState("b")
    b.core.vector[0] = 1.0
    b.core.confidence = 1.0
    with torch.no_grad():
        la, _ = model(x, a.flatten().unsqueeze(0), condition_ids=q)
        lb, _ = model(x, b.flatten().unsqueeze(0), condition_ids=q)
    assert not torch.allclose(la, lb)


def test_query_changes_personal_conditioning():
    cfg = ModelConfig.byte_prototype()
    model = PersonalNativeLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    s = PersonalState("a")
    s.core.vector[0] = 1.0
    s.policies.vector[0] = -1.0
    state = s.flatten().unsqueeze(0)
    with torch.no_grad():
        la, _ = model(x, state, condition_ids=_condition("בחר אפשרות למשימה"))
        lb, _ = model(x, state, condition_ids=_condition("בחר פורמט לתשובה"))
    assert not torch.allclose(la, lb)


def test_inference_does_not_mutate_lm_weights():
    cfg = ModelConfig.byte_prototype()
    model = PersonalNativeLM(cfg).eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    s = PersonalState("a").flatten().unsqueeze(0)
    with torch.no_grad():
        model(x, s, condition_ids=_condition("בקשה נקודתית"))
    after = model.state_dict()
    for key in before:
        assert torch.equal(before[key], after[key]), key
