import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLoraLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        u: int,
        projection_seed: int = 0,
        init_v_bound: float = 1e-3,
        train_dtype: torch.dtype = torch.float32,
        shared_v: nn.Parameter | None = None,
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        assert r > 0 and u > 0

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.u = u

        # 冻结原始参数
        self.weight = nn.Parameter(
            base_linear.weight.detach().clone(),
            requires_grad=False,
        )
        if base_linear.bias is not None:
            self.bias = nn.Parameter(
                base_linear.bias.detach().clone(),
                requires_grad=False,
            )
        else:
            self.bias = None


        with torch.no_grad():
            W = self.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)

            rr = min(r, U.shape[1], Vh.shape[0])
            U = U[:, :rr]       # [d, r]
            S = S[:rr]          # [r]
            V = Vh[:rr, :].T    # [k, r]

        self.rr = rr

        self.register_buffer("U", U.contiguous())
        self.register_buffer("S", S.contiguous())
        self.register_buffer("V", V.contiguous())

        g = torch.Generator(device="cpu")
        g.manual_seed(projection_seed)
        P = torch.randn(u, rr, rr, generator=g, dtype=torch.float32) / math.sqrt(rr)
        self.register_buffer("P", P.contiguous())

        if shared_v is None:
            v = torch.empty(u, dtype=train_dtype)
            nn.init.uniform_(v, -init_v_bound, init_v_bound)
            self.v = nn.Parameter(v)
        else:
            assert shared_v.numel() == u
            self.v = shared_v

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, r={self.rr}, u={self.u}"

    def tiny_matrix(self):
        # M = sum_i v_i P_i, shape [r, r]
        v = self.v.float()
        M = torch.einsum("u,ujk->jk", v, self.P)
        return M

    def delta_weight(self):
        # ΔW = U diag(S) M V^T
        # U: [d, r], S: [r], M: [r, r], V: [k, r]
        US = self.U * self.S.unsqueeze(0)   # [d, r]
        M = self.tiny_matrix()              # [r, r]
        delta = US @ M @ self.V.T           # [d, k]
        return delta.to(self.weight.dtype)

    def forward(self, x):
        W_eff = self.weight + self.delta_weight()
        return F.linear(x, W_eff, self.bias)


def get_parent_module(model: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def should_replace(name: str, target_modules):
    return any(name.endswith(t) for t in target_modules)


def apply_tinylora(model, cfg):
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_replace(name, cfg.target_modules):
            replacements.append((name, module))

    print(f"[TinyLoRA] Found {len(replacements)} target linear modules.")

    weight_tying = float(getattr(cfg, "tinylora_weight_tying", 0.0))

    if weight_tying == 1.0:
        v = torch.empty(cfg.tinylora_u, dtype=torch.float32)
        nn.init.uniform_(v, -cfg.init_v_bound, cfg.init_v_bound)
        shared_v = nn.Parameter(v)
        print(f"[TinyLoRA] Full tying: all modules share one v, shape={tuple(shared_v.shape)}")
        shared_v_map = {name: shared_v for name, _ in replacements}

    elif weight_tying == 0.0:
        print(f"[TinyLoRA] No tying: each module gets its own v (ntie=1), total={len(replacements)} v params")

    else:
        group_size = max(1, int(weight_tying * len(replacements)))
        print(f"[TinyLoRA] Partial tying: group_size={group_size}, "
              f"num_groups={math.ceil(len(replacements) / group_size)}")
        shared_v_map = {}
        current_v = None
        for i, (name, _) in enumerate(replacements):
            if i % group_size == 0:
                v = torch.empty(cfg.tinylora_u, dtype=torch.float32)
                nn.init.uniform_(v, -cfg.init_v_bound, cfg.init_v_bound)
                current_v = nn.Parameter(v)
            shared_v_map[name] = current_v

    for name, module in replacements:
        parent, child_name = get_parent_module(model, name)
        new_module = TinyLoraLinear(
            base_linear=module,
            r=cfg.tinylora_r,
            u=cfg.tinylora_u,
            projection_seed=cfg.seed,
            init_v_bound=cfg.init_v_bound,
            train_dtype=torch.float32,
            shared_v=shared_v_map.get(name, None),  
        )
        setattr(parent, child_name, new_module)

    for n, p in model.named_parameters():
        if ".v" not in n:
            p.requires_grad = False

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print("[TinyLoRA] Trainable parameter names:")
    for n in trainable_names:
        print("  ", n)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[TinyLoRA] Trainable parameters: {trainable} / {total} ({trainable / total:.6%})")

    return model



def untie_tinylora_shared_v(model):
    """
    Break shared TinyLoRA v Parameters before save_pretrained().
    After this, each TinyLoraLinear owns an independent nn.Parameter.
    """
    count = 0
    seen = {}

    for name, module in model.named_modules():
        if isinstance(module, TinyLoraLinear):
            if not isinstance(module.v, nn.Parameter):
                continue

            ptr = module.v.data_ptr()
            if ptr in seen:
                module.v = nn.Parameter(module.v.detach().clone(), requires_grad=True)
                count += 1
            else:
                seen[ptr] = name

    print(f"[TinyLoRA] Untied shared v for {count} modules before saving.")