import copy
import torch
from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState
from personal_model import PersonalLearningSystem, PersonalMicroModel, PERSONAL_LATENT_DIM
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


def test_personal_micro_model_is_tiny_and_slot_routable():
    micro = PersonalMicroModel()
    assert sum(p.numel() for p in micro.parameters()) == 116
    assert PERSONAL_LATENT_DIM == 116
    core = PersonalMicroModel.slot_gradient_mask("core")
    policy = PersonalMicroModel.slot_gradient_mask("policy")
    assert int(core.sum().item()) == 33  # 32 core latent + core confidence
    assert int(policy.sum().item()) == 33
    assert torch.count_nonzero(core * policy).item() == 0


def test_competitive_router_includes_none_route():
    cfg = ModelConfig.byte_prototype()
    model = PersonalNativeLM(cfg).eval()
    state = PersonalState("a").flatten().unsqueeze(0)
    q = _condition("בקשה כלשהי")
    with torch.no_grad():
        _, gates, relevance = model.controller(
            state, model.token_embedding(q), return_routing=True
        )
    none = 1.0 - relevance
    probs = torch.cat([gates, none], dim=-1)
    assert probs.shape == (1, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6)


def test_frozen_lm_can_backprop_only_into_personal_micro_model():
    cfg = ModelConfig.byte_prototype()
    lm = PersonalNativeLM(cfg).eval()
    personal = PersonalLearningSystem(cfg).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    for p in personal.parameters():
        p.requires_grad_(False)

    micro = PersonalMicroModel()
    before_lm = {k: v.detach().clone() for k, v in lm.state_dict().items()}
    before_micro = micro.latent.detach().clone()

    x = torch.randint(0, cfg.vocab_size, (1, 12))
    q = _condition("בקשה אישית")
    state = micro(personal.state_decoder)
    logits, _ = lm(x, state, condition_ids=q)
    loss = logits[..., 0].mean()
    opt = torch.optim.SGD([micro.latent], lr=0.1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    assert not torch.equal(before_micro, micro.latent.detach())
    for key, value in before_lm.items():
        assert torch.equal(value, lm.state_dict()[key]), key


def test_hybrid_tokenizer_roundtrip_without_unknowns(tmp_path):
    from hybrid_tokenizer import HybridHebrewTokenizer
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("המערכת שומרת נתונים.\nבדיקה קצרה עובדת.\n", encoding="utf-8")
    tok = HybridHebrewTokenizer.train([corpus], vocab_size=512, min_frequency=1)
    text = "המערכת שומרת מידע חדש, וגם סימן 123."
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    assert tok.vocab_size == 512


def test_longitudinal_profiles_include_change_noise_and_world_state():
    from longitudinal_data import make_profiles, target_state
    profiles = make_profiles()
    assert len(profiles) == 8
    assert all(len(p.history) >= 14 for p in profiles)
    # Odd-index profiles include an old opposite core choice followed by repeated
    # recent evidence for the current one.
    p = profiles[1]
    assert len(p.history) > 14
    state = target_state(p)
    assert state.shape == (228,)
    assert abs(float(state[128])) > 0.5  # world slot is explicitly represented
