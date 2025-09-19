import torch
import torch.nn as nn

## 把每个模态的 feature map 压缩成一个 token 向量（形状 (𝐵,𝑁,𝐶_tok))
## 输入 (B,N,C_tok) → 输出 (B,N,C_cond)

class FusionTransformer(nn.Module):
    def __init__(self, C_tok=256, C_cond=128, n_heads=4, n_layers=2, mlp_ratio=2, dropout=0.1, max_N=5):
        """
        Simple pooling-based fusion transformer.
        Input:
            tokens: (B, N, C_tok)   # e.g. C_tok=256 from encoder GAP
            cond_mask: (B, N)       # 1=available, 0=missing
        Output:
            cond_feats: (B, N, C_cond)
        """
        super().__init__()
        self.C_tok = C_tok
        self.C_cond = C_cond
        self.max_N = max_N

        # ensure model_dim is divisible by n_heads
        model_dim = C_tok
        if model_dim % n_heads != 0:
            model_dim = (model_dim // n_heads) * n_heads or n_heads
        self.model_dim = model_dim

        self.in_proj = nn.Linear(C_tok, self.model_dim)
        self.pos_emb = nn.Embedding(max_N, self.model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=n_heads,
            dim_feedforward=self.model_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # we use (S,B,E) Transformer 在 PyTorch 早期只支持 (S, B, E)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, C_cond),
        )
        
        self._init_weights()
        self.enabled = True
    
    def _init_weights(self):
        """ Weight Initialization """
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        for module in [self.in_proj, self.out_proj]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)

    def forward(self, tokens: torch.Tensor, cond_mask: torch.Tensor):
        """
        tokens: (B, N, C_tok), float32
        cond_mask: (B, N), 1=available / 0=missing
        returns: cond_feats (B, N, C_cond) or None if disabled
        """
        if not self.enabled:
            return None

        B, N, _ = tokens.shape
        assert N <= self.max_N, f"N ({N}) > max_N ({self.max_N})"
        device = tokens.device
        dtype = tokens.dtype

        # ---- 1. project tokens ----
        x = self.in_proj(tokens)   # (B, N, model_dim)

        # ---- 2. add positional embedding (require long dtype) ----
        pos_ids = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, N)
        x = x + self.pos_emb(pos_ids)

        # ---- 3. make padding mask (bool) ----
        cond_mask = cond_mask.to(device=device, dtype=torch.bool)    # (B, N)
        src_key_padding_mask = ~cond_mask             # True where to mask

        # ---- 4. transformer (expects S,B,E) ----
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        cond_feats = self.out_proj(out)   # (B, N, C_cond)
        cond_feats = cond_feats.to(dtype=dtype)

        return cond_feats