"""V9.1 personal neural residual BEFORE downstream attention layers."""
import torch
from torch import nn
from torch.nn import functional as F

class PersonalConditioner(nn.Module):
    """Trainable shared interface, 4-channel per-user state, zero-output init."""
    def __init__(self,hidden=960,rank=16,state_dim=4):
        super().__init__()
        self.down=nn.Linear(hidden,rank,bias=False)
        self.state=nn.Linear(state_dim,rank,bias=False)
        self.up=nn.Linear(rank,hidden,bias=False)
        nn.init.normal_(self.down.weight,std=0.02)
        nn.init.normal_(self.state.weight,std=0.02)
        nn.init.zeros_(self.up.weight)
        self.rank=rank
        self.state_dim=state_dim
    def forward(self,x,personal):
        """Input: [batch,hidden], preference channels are NUMERIC, not text."""
        if personal.ndim==1:personal=personal.unsqueeze(0)
        if personal.shape[-1]!=self.state_dim:
            raise ValueError("Incompatible personal state")
        h=F.layer_norm(x,(x.shape[-1],))
        residual=self.up(torch.tanh(self.down(h)+self.state(personal.to(x.dtype))))
        return x+residual

def state_for(mode,device=None):
    """Pilot's 4-slot state: two active channels, two reserved slots.

    No false claim that these independent channels already represent learned
    user goals, needs, risk tolerance, etc. Their future semantics need data.
    """
    if mode=="rapid":v=[1.,0.,0.,0.]
    elif mode=="thorough":v=[0.,1.,0.,0.]
    else:raise ValueError("Only explicitly approved rapid/thorough")
    return torch.tensor(v,dtype=torch.float32,device=device)

def attach(model,conditioner,personal,layer_from_end=3):
    """Live forward hook EARLIER than final block, before downstream attention."""
    layers=model.model.layers
    idx=len(layers)-int(layer_from_end)
    if not 0<=idx<len(layers):
        raise ValueError("Invalid decoder block index")
    def hook(_module,_inputs,out):
        hidden=out[0] if isinstance(out,tuple) else out
        # The intervention changes last token at this block output; downstream
        # blocks receive the conditioned token and execute attention normally.
        changed=conditioner(hidden[:,-1,:],personal)
        altered=torch.cat([hidden[:,:-1,:],changed.unsqueeze(1)],dim=1)
        return (altered,*out[1:]) if isinstance(out,tuple) else altered
    return layers[idx].register_forward_hook(hook)

def count(module):
    return sum(p.numel() for p in module.parameters())
