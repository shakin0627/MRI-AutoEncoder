from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

## 根据 input：（x_t, t, cond) 预测 output: v(x_t)Optional
## Flow matching 接收 shared_encoder 的传入 ：[z_patient A_T1, z_patient B_T1, ..... z_patient Z_T1....] 
""" cond_feats = [
    [c_patientA_T1, c_patientA_T2, c_patientA_FLAIR],
    [c_patientB_T1, c_patientB_T2, c_patientB_FLAIR],
    ...
]   # shape = (B, N, C_cond)
"""
class SinusoidalTimeEmb(nn.Module):
    """ D: embedding_dim """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """ t的推荐形状：（B，） """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=device, dtype=torch.float32) 
            * (torch.log(torch.tensor(10000.0, device=device)) / max(half-1,1))
        )
        args = t[:, None] * freqs[None, :] * 2 * torch.pi  ## t-> (B, 1), freqs-> (1, H), (B, H)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))  # (B, D)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, time_emb_dim: Optional[int]):
        """ 目的是对齐通道数做残差加法 """
        """ 
            Input： x(B, in_channels, H, W)  t(B, time_emb_dim)
            Output: (B, out_channels, H, W) 
        """
        super().__init__()
        num_groups_in = min(8, in_channels) if in_channels >= 8 and in_channels % 8 == 0 else 1
        num_groups_out = min(8, out_channels) if out_channels >= 8 and out_channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(num_groups_in, num_channels=in_channels) # num_channels必须被num_groups整除    （B，C，H，W） 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1) # 输出只有通道数改变 （B，out_channels, H, W)
        self.norm2 = nn.GroupNorm(num_groups_out, num_channels=out_channels) # out_channels必须被8整除
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1) # (B，out_channels, H, W)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim is not None else None  # 将(B, time_emb_dim)投影到(B, out_channels)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, stride=1, padding=0, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ t_emb 是一个时间步的向量，来自 SinusoidalTimeEmb """
        """ h [B, out_ch, H, W] """
        h = self.norm1(x) 
        h = F.silu(h) 
        h = self.conv1(h) 
        if self.time_mlp is not None and t_emb is not None: 
            h = h + self.time_mlp(t_emb)[:, :, None, None] # 将(B, out_channels)广播到(B, out_channels, 1, 1)
        h = self.norm2(h) 
        h = F.silu(h) 
        h = self.conv2(h) 

        return h + self.nin_shortcut(x)
    
class CrossAttentionInjector(nn.Module):
    def __init__(self, 
        dim:int, 
        cond_dim: Optional[int] = None, 
        kv_dim: Optional[int] = None, 
        n_heads: int = 4, 
        enabled: bool = True, 
        layer_idx: int = 0,
        total_layers: int = 1,
        selector_config: Optional[Dict[str, Any]] = None,
        rel_dim: int = 64,
        ):
        """
        Cross-attention injector (token-only).
        cond_feats: (B, N, cond_dim)
        cond_mask:  (B, N) with 1 valid, 0 missing
        
        """
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim or dim
        self.n_heads = n_heads if self.kv_dim % n_heads == 0 else 1 

        self.enabled = enabled
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        # Query Injection
        self.q_proj = nn.Linear(dim, self.kv_dim)

        # K/V projection
        if cond_dim is not None: 
            self.k_proj = nn.Linear(cond_dim, self.kv_dim) 
            self.v_proj = nn.Linear(cond_dim, self.kv_dim) 
            self.use_token_proj = True # also create rel_k_proj for relevance computations 
            self.rel_k_proj = nn.Linear(cond_dim, rel_dim) 
        else: 
            self.k_conv = nn.Conv2d(dim, self.kv_dim, kernel_size=1) 
            self.v_conv = nn.Conv2d(dim, self.kv_dim, kernel_size=1) 
            self.use_token_proj = False 
            self.rel_k_proj = None

        self.rel_q_proj = nn.Linear(dim, rel_dim)
        # selector config defaults
        default_selector = {
            "mode": "auto",
            "k": None,  # int or float ratio; if None use default ratios
            "ratio_shallow": 0.6, # 浅层保留60%的token (更多通用特征)
            "ratio_deep": 0.25, # 深层只保留25%的token (最相关特征)
            "temp": 1.0,
        }
        self.selector_config = {**default_selector, **(selector_config or {})}
        self.rel_dim = rel_dim
        self.out_proj = nn.Linear(self.kv_dim, dim)
        ## 通常kv_dim = dim，压缩除外    

    def set_enabled(self, flag: bool):
        self.enabled = flag

    def set_selector_config(self, cfg: Dict[str, Any]):
        self.selector_config.update(cfg)

    # ========== relevance computation ==========

    def _compute_q_and_cond_proj(self, h: torch.Tensor, cond_feats: torch.Tensor):
        # h: (B, C_h, H, W) -> Pooling
        B = h.size(0)
        qg = F.adaptive_avg_pool2d(h, 1).view(B, -1)  # (B, C_h, 1, 1) 
        qg_proj = F.normalize(self.rel_q_proj(qg), dim=-1)  
        cond_proj = F.normalize(self.rel_k_proj(cond_feats), dim=-1)  # (B, N, rel_dim)
        return qg_proj, cond_proj
    
    def _compute_qsim(self, qg_proj: torch.Tensor, cond_proj: torch.Tensor) -> torch.Tensor:
        # cosine sim between qg and each cond token
        return (qg_proj[:, None, :] * cond_proj).sum(dim=-1)  # (B, N)
    
    def _compute_centrality(self, cond_proj: torch.Tensor) -> torch.Tensor:
        # Grammian Matrix (token-token similarity)
        S = torch.matmul(cond_proj, cond_proj.transpose(1, 2))  # (B, N, N)
        diag_mask = torch.eye(S.size(-1), dtype=torch.bool, device=S.device)[None, :, :]
        return S.masked_fill(diag_mask, 0.0).sum(dim=-1)  # (B, N)

    def forward(self, h: torch.Tensor, cond_feats: Optional[torch.Tensor], cond_mask: Optional[torch.Tensor] = None, layer_idx: Optional[int] = None) -> torch.Tensor:
        if (not self.enabled) or (cond_feats is None):
            return h
        if h.dim() != 4:
            raise ValueError(f"h must be (B,C,H,W), got {h.shape}")
        if cond_feats.dim() != 3:
            raise ValueError(f"cond_feats must be (B,N,C), got {cond_feats.shape}")
        
        B, C_h, H, W = h.shape
        S = H * W
        N_cond = cond_feats.size(1)
        cond_feats = cond_feats.to(h.device)
        if cond_mask is not None:
            cond_mask = cond_mask.to(h.device).bool()
            if cond_mask.shape != (B, N_cond):
                raise ValueError(f"cond_mask must be (B,N), got {cond_mask.shape}")
            
        q = h.view(B, C_h, S).permute(0, 2, 1)  # (B, H*W, dim)
        q = self.q_proj(q)                      # (B, H*W, kv_dim)

        K = self.k_proj(cond_feats)             # (B, N, kv_dim)
        V = self.v_proj(cond_feats)             # (B, N, kv_dim)

        ## Multi-head Attention
        d_head = self.kv_dim // self.n_heads
        qh = q.view(B, S, self.n_heads, d_head).permute(0, 2, 1, 3)
        kh = K.view(B, K.size(1), self.n_heads, d_head).permute(0, 2, 1, 3)
        vh = V.view(B, V.size(1), self.n_heads, d_head).permute(0, 2, 1, 3)

        ## qh: (B, n_head, H*W, d_head)
        ## kh: (B, n_heads, N, d_head)
        ## vh: (B, n_heads, N, d_head)

        attn_scores = torch.einsum("bhqd,bhkd->bhqk", qh, kh) / (d_head ** 0.5)

        sel_mode = self.selector_config.get("mode","auto")
        k_cfg = self.selector_config.get("k", None)
        ratio_shallow = float(self.selector_config.get("ratio_shallow",0.6))
        ratio_deep = float(self.selector_config.get("ratio_deep",0.25))
        temp = float(self.selector_config.get("temp",1.0))

        if sel_mode != "all":
            qg_proj, cond_proj = self._compute_q_and_cond_proj(h, cond_feats)
            qsim = self._compute_qsim(qg_proj, cond_proj)        # (B,N) 每个 token 和 query 的相似度
            centrality = self._compute_centrality(cond_proj)     # (B,N) 每个 token 和其他 token 的相似度总和

            if sel_mode=="auto":
                li = layer_idx if layer_idx is not None else self.layer_idx
                use_central = (li < self.total_layers//2)
                scores = centrality if use_central else qsim
                k = max(1, int(N_cond*(ratio_shallow if use_central else ratio_deep))) if k_cfg is None else int(k_cfg)
                op="mask"
            elif sel_mode=="topk_q":
                scores, k, op = qsim, int(k_cfg or max(1,N_cond*0.25)), "mask"
            elif sel_mode=="topk_central":
                scores, k, op = centrality, int(k_cfg or max(1,N_cond*0.6)), "mask"
            elif sel_mode=="weighted":
                weights = torch.softmax(qsim/max(temp,1e-6), dim=-1)
                if cond_mask is not None:
                    weights = weights*cond_mask.float()
                    weights = weights/weights.sum(dim=-1,keepdim=True).clamp_min(1e-6)
                attn_scores = attn_scores + torch.log(weights[:,None,None,:]+1e-12)
                op,k="weighted",None
            else:
                scores, k, op = qsim, N_cond, "mask"

            if op=="mask" and k is not None:
                if k>=N_cond:
                    selected_mask = torch.ones(B,N_cond,dtype=torch.bool,device=h.device)
                else:
                    topk_idx = torch.topk(scores,k,dim=-1).indices
                    selected_mask = torch.zeros(B,N_cond,dtype=torch.bool,device=h.device)
                    selected_mask.scatter_(1, topk_idx, True)
                allowed = selected_mask if cond_mask is None else (selected_mask & cond_mask)

                # fallback
                rows_all_false = (~allowed).all(dim=1)
                if rows_all_false.any():
                    idx0 = torch.topk(scores[rows_all_false],1,dim=-1).indices
                    for bi, ii in enumerate(torch.nonzero(rows_all_false,as_tuple=False).view(-1)):
                        allowed[ii, idx0[bi,0]] = True

                attn_scores = attn_scores.masked_fill(~allowed[:,None,None,:], float("-1e9"))

        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, vh)
        out = out.permute(0,2,1,3).contiguous().view(B,S,self.kv_dim)
        out = self.out_proj(out).permute(0,2,1).view(B,C_h,H,W)
    
        return out

class UNetFlow(nn.Module):
    def __init__(self, 
        in_channels:int, 
        base_channels:int, 
        time_emb_dim:int, 
        depth:int, 
        use_cross_attn: bool = True,
        cond_dim: Optional[int] = None,
        kv_dim: Optional[int] = None,
        selector_config: Optional[Dict[str, Any]] = None,
    ):
        """ chs: 每一层通道数列表 """
        """
        - in_channels: should match Shared_Encoder output (e.g. 256)
        - selector_config: passed to CrossAttentionInjector (dict)
        """
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.time_mlp = nn.Sequential( 
            SinusoidalTimeEmb(time_emb_dim), 
            nn.Linear(time_emb_dim, time_emb_dim), 
            nn.SiLU(), 
            nn.Linear(time_emb_dim, time_emb_dim) 
            )
        
        self.down = nn.ModuleList() 
        self.up = nn.ModuleList()
        self.attn_injectors = nn.ModuleList()
        
        """  ResBlock: 卷积 + 残差，提取特征，同时加上时间嵌入 """
        """  CrossAttentionInjector: 把条件特征（cond_feats，比如另一模态图像 embedding）融合进来 """
        """  保存 skip 特征图，给后面的上采样用 """
        """  F.avg_pool2d: 下采样，空间分辨率减半 """

        chs = [in_channels] + [base_channels * (2**i) for i in range(depth)]
        # downsampling
        for i in range(depth):
            resblock = ResBlock(chs[i], chs[i+1], time_emb_dim=time_emb_dim)
            self.down.append(resblock)
            inj = CrossAttentionInjector(
                dim=chs[i + 1],
                cond_dim=cond_dim,
                kv_dim=kv_dim,
                n_heads=4,
                enabled=use_cross_attn,
                layer_idx=i,
                total_layers=depth,
                selector_config=selector_config,
            )
            self.attn_injectors.append(inj)

        self.bottleneck = ResBlock(chs[-1], chs[-1]*2, time_emb_dim=time_emb_dim)
        self.bottle_attn = CrossAttentionInjector(
            dim=chs[-1] * 2,
            cond_dim=cond_dim,
            kv_dim=kv_dim,
            n_heads=4,
            enabled=use_cross_attn,
            layer_idx=depth,
            total_layers=depth,
            selector_config=selector_config,
        )

        # upsampling
        for i in reversed(range(depth)):
            if i == depth - 1:  # 第一个上采样层
                in_ch = chs[i+1] * 3  # bottleneck(chs[i+1]*2) + skip(chs[i+1])
            else:
                in_ch = chs[i+1] * 2  # up_prev(chs[i+1]) + skip(chs[i+1])
            out_ch = chs[i]
            resblock = ResBlock(in_ch, out_ch, time_emb_dim=time_emb_dim)
            self.up.append(resblock)
            inj = CrossAttentionInjector(
                dim=chs[i],
                cond_dim=cond_dim,
                kv_dim=kv_dim,
                n_heads=4,
                enabled=use_cross_attn,
                layer_idx=depth + (depth - 1 - i) + 1,  # unique increasing idx for up layers
                total_layers=depth,
                selector_config=selector_config,
            )
            self.attn_injectors.append(inj)

        self.final = nn.Sequential( 
            nn.GroupNorm(8 if chs[0] % 8 == 0 else 1, chs[0]), 
            nn.SiLU(), 
            nn.Conv2d(chs[0], in_channels, kernel_size=3, padding=1, stride=1)
         )
        
    def set_cross_attn_enabled(self, flag: bool):
        self.use_cross_attn = flag
        for inj in self.attn_injectors:
            inj.set_enabled(flag)
        self.bottle_attn.set_enabled(flag)

    def set_selector_config(self, cfg: Dict[str, Any]):
        # update all injectors' selector config
        for inj in self.attn_injectors:
            inj.set_selector_config(cfg)
        self.bottle_attn.set_selector_config(cfg)
    

    def forward(self, x: torch.Tensor, cond_feats: torch.Tensor, t: torch.Tensor, cond_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        skips = []
        h = x

        for i, resblock in enumerate(self.down):
            h = resblock(h, t_emb) ## 维度不变，增加Time embedding
            h = self.attn_injectors[i](h, cond_feats, cond_mask)
            skips.append(h)
            h = F.avg_pool2d(h, 2)

        h = self.bottleneck(h, t_emb)
        h = self.bottle_attn(h, cond_feats, cond_mask)

        for i, resblock in enumerate(self.up):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            # 获取对应的跳跃连接
            skip_connection = skips[-(i + 1)]
            # 确保空间维度匹配
            if h.shape[2:] != skip_connection.shape[2:]:
                skip_connection = F.interpolate(skip_connection, size=h.shape[2:], mode="nearest")
            h = torch.cat([h, skips[-(i + 1)]], dim=1)
            h = resblock(h, t_emb)
            inj = self.attn_injectors[len(self.down) + i]
            h = inj(h, cond_feats, cond_mask)

        return self.final(h)
    
## cond_feats shape：(B, N, C_cond)
## B = batch size
## N = 条件数量（模态数、token 数，比如 T1/T2/FLAIR 就是 3）
## C_cond = 条件 embedding 维度（和 cond_dim 对应，一般是 Fusion Transformer 输出的 hidden dim）


## cond_mask shape：(B, N)
""" 模型forward方法的输入需要满足："""
""" x：形状为 (B, in_channels, H, W)（输入特征图，通道数与in_channels一致）"""
""" t：形状为 (B,)（时间步，每个样本一个标量时间值）"""
"""cond_feats：形状为 (B, N, C_cond)（条件特征，N为条件数量，C_cond需与cond_dim一致)"""
"""cond_mask（可选）：形状为 (B, N)（掩码，1 表示有效条件，0 表示缺失）"""