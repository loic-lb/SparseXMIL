import torch
import torch.nn as nn
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x, return_weights=False):
        if return_weights:
            x1, weights = self.attn(self.norm(x), return_attn=True)
            x = x + x1
            return x, weights
        else:
            x = x + self.attn(self.norm(x))
            return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes, transmil_size=512):
        super(TransMIL, self).__init__()
        self.name = "TransMIL"
        self.transmil_size = transmil_size
        self.pos_layer = PPEG(dim=transmil_size)
        self._fc1 = nn.Sequential(nn.Linear(1024, self.transmil_size), nn.ReLU()) #512
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.transmil_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=self.transmil_size)
        self.layer2 = TransLayer(dim=self.transmil_size)
        self.norm = nn.LayerNorm(self.transmil_size)
        self._fc2 = nn.Linear(self.transmil_size, self.n_classes)

    def forward(self, data, return_attention=False):
        res = []
        if return_attention:
            attention_weights = []
        for x in data:

            x = x.cuda().unsqueeze(0)  # [B, n, 1024]

            h = self._fc1(x)  # [B, n, 512]

            # ---->pad
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

            # ---->cls_token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            h = torch.cat((cls_tokens, h), dim=1)

            # ---->Translayer x1
            h = self.layer1(h)  # [B, N, 512]

            # ---->PPEG
            h = self.pos_layer(h, _H, _W)  # [B, N, 512]

            # ---->Translayer x2
            if return_attention:
                h, weights = self.layer2(h, return_weights=True)  # [B, N, 512]
                attention_weights.append(weights)
            else:
                h = self.layer2(h)

            # ---->cls_token
            h = self.norm(h)[:, 0]

            # ---->predict
            res.append(self._fc2(h))  # [B, n_classes]

        if return_attention:
            return torch.cat(res), attention_weights
        else:
            return torch.cat(res)