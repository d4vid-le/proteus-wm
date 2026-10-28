"""Agent talk loop - DSML + grounded language, both verified by world_result."""
import pathlib as _pathlib
import sys as _sys

ROOT = _pathlib.Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(ROOT))

for _sub in ("world_model", "continual", "vision", "agents", "evals"):
    _sys.path.insert(0, str(ROOT / "experiments" / _sub))
import numpy as np
import re
try:
    from wm import WorldModel
except ImportError:
    try:
        from wm import WorldModelV2 as WorldModel
    except ImportError:
        from wm import WorldModelV3 as WorldModel
from env import GridWorld, ACTIONS

TOOL_RESULT = '<tool_result>{content}</tool_result>'

def encode_action_dsml(act_id):
    param = f'<｜DSML｜parameter name="dir" string="true">{ACTIONS[act_id]}</｜DSML｜parameter>'
    return f'<｜DSML｜invoke name="move">\n{param}\n</｜DSML｜invoke>'

def decode_dsml_action(text):
    m = re.search(r'name="move".*?<｜DSML｜parameter name="dir".*?>(.*?)</', text, re.DOTALL)
    if not m: return None
    return ACTIONS.index(m.group(1).strip()) if m.group(1).strip() in ACTIONS else None

def parse_language_pos(text):
    """Extract [x,y] from language: 'at [1.0, 2.0]' """
    m = re.search(r'at\s*\[([0-9\.\-\s,]+)\]', text)
    if not m: return None
    try:
        vals = [float(x.strip()) for x in m.group(1).split(',')]
        return np.array(vals)
    except: return None

def verify_language(lang, true_pos, tol=1.0):
    """Language is verified if claimed pos within tol of true world_result pos."""
    pred = parse_language_pos(lang)
    if pred is None: return False, "no pos in lang"
    err = np.linalg.norm(pred - np.array(true_pos, dtype=float))
    return err < tol, f"err {err:.2f}"

class Agent:
    def __init__(self, wm):
        self.wm = wm

    def sense(self, obs):
        return self.wm.encode(obs)

    def act_mpc(self, s, horizon=1):
        n_act=getattr(self.wm,'act_dim',4)
        if hasattr(self.wm, 'decode_pos'):
            best=None; bestc=1e9
            is_file=getattr(self.wm,'obs_dim',75)==9
            for a in range(n_act):
                s_pred,_=self.wm.predict(s,a)
                pos=self.wm.decode_pos(s_pred)
                if is_file:
                    cost=float(abs(pos[0])+abs(pos[1]))  # file task: want 0,0 messy
                else:
                    cost=abs(float(pos[0])-4)+abs(float(pos[1])-4)
                if cost<bestc:
                    bestc=cost; best=a
            return best
        return np.random.randint(0,n_act)

    def talk(self, obs, mode="action"):
        s=self.sense(obs)
        if mode=="action":
            a=self.act_mpc(s)
            dsml=encode_action_dsml(a)
            block=f"<｜DSML｜tool_calls>\n{dsml}\n</｜DSML｜tool_calls>"
            return block, a
        elif mode=="language":
            # Grounded language: claim current pos + intended move
            pos=self.wm.decode_pos(s) if hasattr(self.wm,'decode_pos') else s[:2]
            pos_r=np.round(pos,1).tolist()
            # also include predicted next pos for verifiability
            a=self.act_mpc(s)
            s_pred,_=self.wm.predict(s,a)
            pos_next=self.wm.decode_pos(s_pred) if hasattr(self.wm,'decode_pos') else pos
            pos_next_r=np.round(pos_next,1).tolist()
            lang=f"Agent at {pos_r} goal at [4, 4] will move {ACTIONS[a]} to {pos_next_r}"
            return lang, a

def run_episode(wm, max_steps=20, render=True, verify_lang=True):
    env=GridWorld(size=5)
    agent=Agent(wm)
    obs=env.reset()
    lang_verified=0; lang_total=0; dsml_verified=0
    for t in range(max_steps):
        # --- Language talk (verifiable) ---
        lang, a_lang = agent.talk(obs, mode="language")
        # Language claims current pos; verify against true env.agent BEFORE move
        ok, info = verify_language(lang, env.agent.tolist(), tol=1.0)
        lang_total+=1
        if ok: lang_verified+=1
        if render:
            print(f"\nStep {t} Talk(lang): {lang} | verify {ok} {info}")
        # --- DSML talk ---
        block, act = agent.talk(obs, mode="action")
        # DSML verification: predict vs world_result
        s=agent.sense(obs)
        s_pred,_=wm.predict(s, act)
        if render:
            print(f"Step {t} Talk(DSML): {ACTIONS[act]} {block[:55]}...")
            print(env.render_ascii())
        obs_next, reward, done,_=env.step(act)
        s_true=wm.encode_target(obs_next) if hasattr(wm,'encode_target') else wm.encode(obs_next)
        mse=float(np.mean((s_pred-s_true)**2))
        dsml_ok=mse<0.05
        if dsml_ok: dsml_verified+=1
        # Language next-pos verification
        pos_next_pred=wm.decode_pos(s_pred) if hasattr(wm,'decode_pos') else None
        world_pos=np.array(env.agent.tolist(),dtype=float)
        lang_next_ok=False
        if pos_next_pred is not None:
            lang_next_ok=np.linalg.norm(pos_next_pred-world_pos)<1.0
        result=TOOL_RESULT.format(content=f"reward={reward:.2f} done={done} agent={env.agent.tolist()} dsml_mse={mse:.4f} dsml_ok={dsml_ok} lang_next_ok={lang_next_ok}")
        if render:
            print(f"World->Agent: {result}")
            # verify language next pos claim
            m=re.search(r'to\s*\[([0-9\.\-\s,]+)\]', lang)
            if m:
                claimed_next=np.array([float(x) for x in m.group(1).split(',')])
                err_next=np.linalg.norm(claimed_next-world_pos)
                print(f"  Lang claim next {claimed_next.tolist()} vs world {world_pos.tolist()} err {err_next:.2f} {'OK' if err_next<1.0 else 'FAIL'}")
        obs=obs_next
        if done:
            if render: print(f"Goal reached in {t+1} steps | lang {lang_verified}/{lang_total} dsml {dsml_verified}/{t+1}")
            break
    return done, lang_verified, lang_total, dsml_verified
