import sys, types, torch, numpy as np
torch.Tensor.cuda = lambda self, *a, **k: self
_orig_tensor = torch.tensor
sys.path.insert(0, "src")
import run_features as rf
torch.manual_seed(0)
d, Fn, k = 16, 40, 5
sae = {"W_enc": torch.randn(Fn, d), "b_enc": torch.zeros(Fn), "W_dec": torch.nn.functional.normalize(torch.randn(d, Fn), dim=0),
       "b_dec": torch.randn(d) * 0.1, "thr": torch.full((Fn,), 0.5), "sub_bias": True}
dirv = torch.nn.functional.normalize(torch.randn(d), dim=0)
fe = rf.FeatureEdit(sae, list(range(k)), scale=0.0, dir_bella=dirv)
B = 3
# prefill
fe(None, None, (torch.randn(B, 7, d),))
# 4 decode steps; row lengths 4, 2, 1
xs = [torch.randn(B, 1, d) for _ in range(4)]
outs = [fe(None, None, (x,)) for x in xs]
fe.finish_batch([4, 2, 1])
st = fe.stats()
assert st["generated_positions_real"] == 7 and st["decode_positions_incl_padding"] == 12, st
# exact check of projection: recompute delta on the counted positions
tot = 0.0; act = 0.0
for t, x in enumerate(xs):
    pre = (x[:, 0] - sae["b_dec"]) @ sae["W_enc"][:k].T
    z = pre * (pre > 0.5)
    delta = -(z @ sae["W_dec"][:, :k].T)
    for r, n in enumerate([4, 2, 1]):
        if t < n:
            tot += float(delta[r] @ dirv); act += float((z[r] > 0).sum())
assert abs(st["mean_shift_along_bella_dir_per_generated_token"] - tot / 7) < 1e-5
assert abs(st["mean_active_per_generated_token"] - act / 7) < 1e-6
# direction add stats
da = rf.DirectionAdd(dirv * -3.0, dir_bella=dirv)
assert abs(da.stats()["mean_shift_along_bella_dir_per_generated_token"] + 3.0) < 0.02
# sub_bias false path
sae2 = dict(sae, sub_bias=False)
z1 = rf.sae_encode(sae, torch.randn(2, d)); z2 = rf.sae_encode(sae2, torch.randn(2, d))
assert z1.shape == z2.shape
print("ok", st["mean_shift_along_bella_dir_per_generated_token"], st["mean_active_per_generated_token"], rf.SUB_BIAS)
