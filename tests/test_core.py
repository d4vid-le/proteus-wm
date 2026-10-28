#!/usr/bin/env python3
"""test_core.py — pytest smoke tests. CI-able: run `python3 -m pytest test_core.py`.

Covers the core building blocks so regressions are caught fast:
  - HJEPA forward + predict + update_targets (architecture smoke)
  - GatedDynamics / build_gated preserves behavior (the critical bug we fixed)
  - GrowableJEPA growth is invariant + learns new type
  - RealEnv obs/step/verify on an actual temp dir
  - Proteus Gates vs naive (no-forgetting property on a representable shift)
"""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import pathlib
import sys

import numpy as np
import torch
import pytest

from proteus.world.hjepa_torch import HJEPA_K4
from proteus.world.gates import GatedDynamics, build_gated, predict_macro_gated


def make_model(**kw):
    defaults = dict(obs_dim=12, act_dim=4, hidden=128, z_micro=32, z_macro=64, k=4, pos_dim=12)
    defaults.update(kw)
    return HJEPA_K4(**defaults)


def test_hjepa_forward_and_targets():
    torch.manual_seed(0)
    m = make_model()
    obs = torch.randn(8, 12)
    A = torch.randint(0, 4, (8, 4))
    obs1 = torch.randn(8, 12)
    obs4 = torch.randn(8, 12)
    losses = m(obs, A, obs1, obs4)
    assert "total" in losses and "macro" in losses and "pos" in losses
    m.update_targets()
    # predict works
    z = m.enc_macro(obs)
    zp = m.predict_macro(z, A)
    assert zp.shape == (8, 64)


def test_build_gated_preserves_behavior():
    """The critical bug: build_gated must NOT change predictions on old states."""
    torch.manual_seed(0)
    m = make_model()
    z = torch.randn(16, 64)
    A = torch.randint(0, 4, (16, 4))
    with torch.no_grad():
        before = m.predict_macro(z, A).clone()
    build_gated(m)
    m.eval()
    with torch.no_grad():
        after = m.predict_macro(z, A)
    assert float((before - after).abs().max()) < 1e-5, "build_gated changed behavior"


def test_gated_dynamics_learns_representable_shift_no_forget():
    """Gate learns a new target while old WEIGHTS frozen -> old err bounded."""
    torch.manual_seed(0)
    m = make_model()
    z = torch.randn(64, 64)
    A = torch.randint(0, 4, (64, 4))
    with torch.no_grad():
        target_old = m.predict_macro(z, A).clone()
    build_gated(m)
    m.eval()
    shift = torch.zeros_like(target_old)
    shift[:, :3] = 0.8
    target_new = (target_old + shift).detach()
    m.pred_macro.freeze_old()
    opt = torch.optim.Adam(list(m.pred_macro.new.parameters()) + [m.pred_macro.gate_logit], lr=3e-3)
    for _ in range(40):
        opt.zero_grad()
        zp = predict_macro_gated(m, z, A)
        loss = torch.mean((zp - target_new) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        err_new = torch.mean((predict_macro_gated(m, z, A) - target_new) ** 2).item()
        err_old = torch.mean((predict_macro_gated(m, z, A) - target_old) ** 2).item()
    assert err_new < 0.01, f"gate did not learn new target ({err_new})"
    assert err_old < 0.5, f"old drifted too much ({err_old})"


def test_growable_growth_invariant():
    from proteus.world.growable import GrowableJEPA, BASE_DIM, ACT_DIM
    torch.manual_seed(0)
    m = GrowableJEPA(BASE_DIM, ACT_DIM, hidden=128, z_macro=64, k=4)
    z = torch.randn(8, 64)
    A = torch.randint(0, 4, (8, 4))
    with torch.no_grad():
        zp0 = m.predict(z, A, torch.zeros(8, 0)).clone()
    m.add_type()
    m.freeze_base()
    with torch.no_grad():
        zp1 = m.predict(z, A, torch.zeros(8, 1))  # type absent -> masked
    assert float((zp0 - zp1).abs().max()) < 1e-5, "growth not invariant"


def test_realenv_step_verify():
    import proteus.envs.real_env as real_env
    root = pathlib.Path("/tmp/proteus-test-realenv")
    real_env.seed_real_world(root, keep_pdf=True)
    env = real_env.RealEnv(root)
    text, vec = env.obs()
    assert vec.shape == (12,)
    # move a txt
    before = vec.copy()
    t2, v2, r, d, info = env.step(0)  # move_txt
    ok, detail = env.verify_after("move_txt", before, v2, True, v2[8])
    assert ok, detail
    # KEEP not lost
    assert not info.get("keep_lost")


def test_le_gates_vs_naive_no_forgetting():
    """Proteus Gates freezes old -> no forgetting; naive drifts. Compact synthetic."""
    from proteus.world.gates import build_gated, predict_macro_gated
    torch.manual_seed(0)
    z = torch.randn(64, 64)
    A = torch.randint(0, 4, (64, 4))

    def train_naive():
        m = make_model()
        with torch.no_grad():
            target_old = m.predict_macro(z, A).clone()
        shift = torch.randn_like(target_old) * 0.8
        target_new = (target_old + shift).detach()
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for _ in range(40):
            opt.zero_grad()
            loss = torch.mean((m.predict_macro(z, A) - target_new) ** 2)
            loss.backward(); opt.step()
        with torch.no_grad():
            eo = torch.mean((m.predict_macro(z, A) - target_old) ** 2).item()
            en = torch.mean((m.predict_macro(z, A) - target_new) ** 2).item()
        return eo, en

    def train_gated():
        m = make_model()
        with torch.no_grad():
            target_old = m.predict_macro(z, A).clone()
        shift = torch.randn_like(target_old) * 0.8
        target_new = (target_old + shift).detach()
        build_gated(m); m.eval()
        m.pred_macro.freeze_old()
        opt = torch.optim.Adam(list(m.pred_macro.new.parameters()) + [m.pred_macro.gate_logit], lr=3e-3)
        for _ in range(40):
            opt.zero_grad()
            loss = torch.mean((predict_macro_gated(m, z, A) - target_new) ** 2)
            loss.backward(); opt.step()
        with torch.no_grad():
            eo = torch.mean((predict_macro_gated(m, z, A) - target_old) ** 2).item()
            en = torch.mean((predict_macro_gated(m, z, A) - target_new) ** 2).item()
        return eo, en

    naive_old, naive_new = train_naive()
    gated_old, gated_new = train_gated()
    # BOTH should learn the new target, but gated must forget the old less
    assert gated_new < 0.1 and naive_new < 0.1, "neither learned the new target"
    assert gated_old < naive_old, f"gates should forget less ({gated_old} vs {naive_old})"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

def test_lifelong_routing_diagonal():
    """WO1: each shift type routes to its own slot (confusion diagonal)."""
    from proteus.world.lifelong import LifelongJEPA, LifelongWorld, BASE_DIM, ACT_DIM
    torch.manual_seed(0)
    m = LifelongJEPA(BASE_DIM, ACT_DIM, hidden=128, z_macro=64, k=4)
    m.add_slot()
    m.add_slot()
    # two types present, route should pick distinct slots
    obs = np.zeros(BASE_DIM + 2, dtype=np.float32)
    obs[BASE_DIM] = 2.0  # type 1 present
    r1 = m.route(obs)
    obs2 = obs.copy()
    obs2[BASE_DIM] = 0.0
    obs2[BASE_DIM + 1] = 3.0  # type 2 present
    r2 = m.route(obs2)
    assert int(np.argmax(r1)) == 0, f"type1 should route to slot 0 ({r1})"
    assert int(np.argmax(r2)) == 1, f"type2 should route to slot 1 ({r2})"


def test_chaos_c1_collision():
    """WO3 C1: a normal-named file carrying the KEEP guard is never touched."""
    from proteus.eval.chaos import make_agent, c1_collision
    a = make_agent("/tmp/proteus-pytest-c1")
    ok, detail = c1_collision(a)
    assert ok, detail


def test_barrier_certificate_holds():
    """P3-3: with a trained barrier head, no gate combo crosses the KEEP boundary."""
    from proteus.safety.barrier import BarrierCertifiedModel, certify
    from proteus.world.hjepa_torch import HJEPA_K4
    import proteus.envs.cleanup_env as cleanup_env
    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=12, act_dim=4, hidden=128, z_micro=32, z_macro=64, k=4, pos_dim=12)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    states = []
    for s in range(10):
        env = cleanup_env.CleanupDesktop()
        _, vec = env.reset(seed=s * 3 + 1)
        if vec[7] == 1:
            states.append(vec)
    viol, _, _, margin = certify(model, states, perturb_scale=0.2)
    assert viol == 0, f"barrier violated ({viol})"
    assert margin.get("barrier_margin", 0) > 1.0


def test_torchscript_export_matches():
    """P3-4: TorchScript export is bit-identical to eager."""
    from proteus.eval.latency import InferencePipeline, export
    from proteus.world.hjepa_torch import HJEPA_K4
    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=12, act_dim=4, hidden=128, z_micro=32, z_macro=64, k=4, pos_dim=12)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    _, ts = export(model)
    obs = torch.randn(1, 12)
    eager = InferencePipeline(model)(obs)
    d = float((eager - ts(obs)).abs().max())
    assert d < 1e-6, f"TorchScript mismatch ({d})"


def test_vision_real_perception():
    """Vision: corrupting the KEEP red block must flip the KEEP decision (real
    perception, not a lookup)."""
    import proteus.perception.vision as vis
    from proteus.world.hjepa_torch import HJEPA_K4
    import proteus.envs.real_env as real_env
    torch.manual_seed(0)
    model = HJEPA_K4(obs_dim=12, act_dim=4, hidden=128, z_micro=32, z_macro=64, k=4, pos_dim=12)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    agent = vis.VisionAgent(model)
    pairs = vis.collect_pairs(n=120, seed=0)
    vis.train_vision(model, agent, pairs, epochs=80)
    flips = 0
    for t in range(10):
        root = pathlib.Path(f"/tmp/proteus-pytest-vis-{t}")
        real_env.seed_real_world(root, keep_pdf=True, seed=t)
        env = real_env.RealEnv(root)
        _, vec = env.obs()
        img = vis.render_scene(vec)
        img_c = img.copy()
        img_c[vis.IMG_SIZE - 16:vis.IMG_SIZE - 6, vis.IMG_SIZE - 16:vis.IMG_SIZE - 6] = np.array([10, 10, 20]) / 255.0
        _, kc = agent.route_from_vision(img, vec)
        _, kd = agent.route_from_vision(img_c, vec)
        if kc and not kd:
            flips += 1
    assert flips >= 7, f"vision not reading KEEP from pixels ({flips}/10)"


def test_real_vision_capture():
    """Real perception: render a real directory to a real image; verify the image
    is a valid non-trivial PNG (the capture mechanism works end-to-end)."""
    import proteus.perception.real_vision as real_vision, proteus.envs.real_env as real_env
    import tempfile, pathlib as _p
    real_env.seed_real_world("/tmp/proteus-pytest-rv", keep_pdf=True)
    img = _p.Path(tempfile.gettempdir()) / "proteus_pytest_rv.png"
    ok = real_vision.capture_render(pathlib.Path("/tmp/proteus-pytest-rv"), img)
    assert ok, "capture failed"
    assert img.exists() and img.stat().st_size > 1000, "image not a valid render"
    from PIL import Image
    im = Image.open(img)
    assert im.size[0] >= 64, "rendered image too small"
    im.verify()  # valid image


def test_vision_jepa_frozen_vit():
    """The JEPA way: frozen ViT image->latent is fast AND aligns to the macro latent."""
    import proteus.perception.vision_jepa as vj
    from proteus.world.hjepa_torch import HJEPA_K4
    from PIL import Image
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    vit, transforms = vj.build_frozen_vit(dev)
    model = HJEPA_K4(obs_dim=12, act_dim=4, hidden=128, z_micro=32, z_macro=64, k=4, pos_dim=12)
    ckpt = pathlib.Path(ROOT / "checkpoints" / "cleanup_hjepa.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval(); model.to(dev)
    fvj = vj.FrozenVisionJEPA(vit, z_macro=64).to(dev)
    pairs = vj.collect_pairs(n=60, seed=0)
    opt = torch.optim.Adam(fvj.proj.parameters(), lr=3e-3)
    import random
    for ep in range(25):
        for img, vec in random.sample(pairs, min(24, len(pairs))):
            x = transforms(Image.fromarray(img)).unsqueeze(0).to(dev)
            zt = model.enc_macro(torch.as_tensor(vec, dtype=torch.float32, device=dev).unsqueeze(0))
            opt.zero_grad(); fvj.align_loss(x, zt).backward(); opt.step()
    # alignment must be decent
    x = transforms(Image.fromarray(pairs[0][0])).unsqueeze(0).to(dev)
    with torch.no_grad():
        z = fvj.encode(x); z = z / z.norm()
        zt = model.enc_macro(torch.as_tensor(pairs[0][1], dtype=torch.float32, device=dev).unsqueeze(0))
        zt = zt / zt.norm()
        cos = float((z * zt).sum().item())
    assert cos > 0.5, f"vision-jepa not aligned ({cos})"


def test_count_head_reads_counts():
    """Vision count head: reads exact counts from pixels (mean abs err < 0.3)."""
    import proteus.perception.count_head as ch
    torch.manual_seed(0)
    enc, tr = ch.build_resnet18()
    cv = ch.CountVision(enc, tr).to(ch.DEVICE)
    pairs = ch.collect(n=80, seed=0)
    ch.train_count(cv, pairs, epochs=50)
    errs = []
    for img, vec in ch.collect(n=40, seed=99):
        with torch.no_grad():
            c = cv.count(img).cpu().numpy()[0]
        errs.append(np.mean([abs(c[i] - vec[i]) for i in range(3)]))
    assert np.mean(errs) < 0.3, f"count head mean err {np.mean(errs):.3f}"


def test_visual_confirm_detects_change():
    """Vision superpower F4: full-res pixel diff detects a real layout change
    (file move), while a no-change comparison stays below threshold."""
    from proteus.perception.finder_vision import visual_confirm
    from PIL import Image
    import tempfile, pathlib as _p, numpy as np
    b = _p.Path(tempfile.gettempdir()) / "vc_b.png"
    a = _p.Path(tempfile.gettempdir()) / "vc_a.png"
    base = np.full((100, 100, 3), 50, dtype=np.uint8)
    Image.fromarray(base).save(b)
    changed = base.copy()
    changed[40:45, 40:45] = [250, 40, 40]  # a file icon appears
    Image.fromarray(changed).save(a)
    ok, d = visual_confirm(None, b, a)
    assert ok, f"change not detected (diff {d:.5f})"
    # no-change should be below threshold
    ok2, d2 = visual_confirm(None, b, b)
    assert not ok2, f"false positive on identical (diff {d2:.5f})"
