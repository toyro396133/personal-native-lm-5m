"""V9.0: owner-conditioned residual INSIDE a frozen LM decoder block.

This initial proof of concept trains the shared interface once, while a tiny
per-user preference state is swapped without any per-user gradient updates.
It is NOT yet a full independently trained neural model for each user.
"""
import torch
from torch import nn
from torch.nn import functional as F

class PersonalNeuralGate(nn.Module):
    """Inject preference-conditioned residual before the frozen backbone final norm."""
    def __init__(self,hidden_size=960,rank=16):
        super().__init__()
        self.down=nn.Linear(hidden_size,rank,bias=False)
        self.owner=nn.Linear(1,rank,bias=False)
        self.up=nn.Linear(rank,hidden_size,bias=False)
        nn.init.normal_(self.down.weight,std=0.02)
        nn.init.normal_(self.owner.weight,std=0.12)
        nn.init.zeros_(self.up.weight)
        self.rank=rank
    def delta(self,hidden,owner_value):
        owner=hidden.new_tensor([[float(owner_value)]])
        h=F.layer_norm(hidden,(hidden.shape[-1],))
        return self.up(torch.tanh(self.down(h)+self.owner(owner)))
    def forward(self,hidden,owner_value):
        return hidden+self.delta(hidden,owner_value)

def hook_last_block(model,gate,owner_value):
    """REAL live model hook; not text prompting, not a terminal reranker.

    Only the last token is modified at a block output BEFORE final RMSNorm.
    This is a limited one-layer residual conditioning experiment, not yet
    repeated cross-layer personalized inference.
    """
    layers=model.model.layers
    def hook(_module,_inputs,out):
        h=out[0] if isinstance(out,tuple) else out
        end=gate(h[:,-1,:],owner_value).unsqueeze(1)
        if h.shape[1]==1:updated=end
        else: updated=torch.cat((h[:,:-1,:],end),dim=1)
        return (updated,*out[1:]) if isinstance(out,tuple) else updated
    return layers[-1].register_forward_hook(hook)

def personal_state(mode):
    """Explicitly approved owner setting, never a gold label."""
    if mode not in ("rapid","thorough"):
        raise ValueError("Only approved rapid/thorough states allowed in this pilot")
    return -1.0 if mode=="rapid" else +1.0

def params(gate):return sum(x.numel() for x in gate.parameters())
