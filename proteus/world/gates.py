"""gates.py — trainable residual gates for continual learning (proteus port).

From an earlier gated-adapter prototype: gates are nn.Parameter logits, gate = sigmoid(logit),
applied as residual scalars on a sub-module. This port wraps the H-JEPA macro
dynamics predictor so a domain shift is learned WITHOUT forgetting the old:

    z_{t+k} = z + (1-g)*old_delta(z,A) + g*new_delta(z,A)
    g = sigmoid(gate_logit), trainable.
    old_delta frozen after domain A; new_delta + gate_logit train on domain B.

Usage:
    from proteus.world.gates import GatedDynamics, build_gated
    model = build_gated()                      # wraps pred_macro with gates
    model.pred_macro.freeze_old()              # after domain A
    # train domain B: optimizer over new.parameters() + gate_logit
"""
import torch
from torch import nn

from proteus.world.hjepa_torch import MLP


class GatedDynamics(nn.Module):
    """Residual gate on a dynamics predictor delta (earlier prototype style).

    forward: out = (1-g)*old(inp) + g*new(inp), g = sigmoid(gate_logit).
    At gate_logit = -inf the new path is off (pure old); at +inf pure new.
    """

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.old = MLP([in_dim, hidden, out_dim])   # domain A (freeze after A)
        self.new = MLP([in_dim, hidden, out_dim])   # domain B (trains)
        self.gate_logit = nn.Parameter(torch.zeros(1))
        with torch.no_grad():
            for pn, po in zip(self.new.parameters(), self.old.parameters()):
                pn.copy_(po)  # new = copy of old -> g=0 is exactly old behavior

    def forward(self, inp):
        g = torch.sigmoid(self.gate_logit)
        return (1 - g) * self.old(inp) + g * self.new(inp)

    def freeze_old(self):
        """Freeze the old-domain path so continual learning never overwrites it."""
        for p in self.old.parameters():
            p.requires_grad = False

    def gate_value(self):
        return float(torch.sigmoid(self.gate_logit).item())


def build_gated(model):
    """Replace a HJEPA model's pred_macro with a GatedDynamics wrapper.
    PRESERVES the trained pred_macro weights as the 'old' path (identity at
    g=0). Returns the same model object (mutated).

    IMPORTANT: the existing pred_macro must be a plain MLP (has .net). Its
    weights become old; new is a copy. gate_logit init 0 -> sigmoid 0.5, but
    since old==new at build, output is identical to before (0.5*old+0.5*new=old).
    """
    in_dim = model.z_macro + model.k * model.act_dim
    old_net = model.pred_macro  # the trained MLP
    gated = GatedDynamics(in_dim, 128, model.z_macro)
    # copy the TRAINED weights into old (not random), new = copy of old
    with torch.no_grad():
        gated.old.load_state_dict(old_net.state_dict())
        gated.new.load_state_dict(old_net.state_dict())
    model.pred_macro = gated
    return model


def predict_macro_gated(model, z, A):
    """predict_macro that routes through the gated predictor.
    z: (B, z_macro), A: (B, k) ints."""
    A_oh = torch.nn.functional.one_hot(A.long(), model.act_dim).float().reshape(A.shape[0], -1)
    inp = torch.cat([z, A_oh], dim=-1)
    return z + model.pred_macro(inp)