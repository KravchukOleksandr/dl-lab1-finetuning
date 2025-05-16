from torch import nn
import torch
import math


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels, patch_size, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.P = patch_size
        self.D = embedding_dim

        self.lin_projection = nn.Conv2d(
            in_channels=in_channels, out_channels=self.D,
            kernel_size=self.P, stride=self.P
        )

    def forward(self, x):
        # x.shape = [bs, 3, H, W]
        embd = self.lin_projection(x)
        # embd.shape = [bs, D, H // P, W // P]
        embd = embd.flatten(2, -1).transpose(1, 2)
        # embd.shape = [bs, H // P * W // P, D]
        return embd
    

class ClassEmbedding(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.D = embedding_dim
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, self.D))
    
    def forward(self, x):
        batch_cls_tokens = self.cls_tokens.expand(x.size(0), 1, self.D)
        x = torch.cat((batch_cls_tokens, x), 1)
        return x


class PositionalEmbeddings(nn.Module):
    def __init__(self, embedding_size, max_patch_num=14**2):
        super().__init__()
        self.D = embedding_size
        PE = torch.zeros(max_patch_num, embedding_size)

        for p in range(max_patch_num):
            for i in range(embedding_size):
                if i % 2 == 0:
                    PE[p, i] = math.sin(p / 10000**(i/self.D))
                else:
                    PE[p, i] = math.cos(p / 10000**((i-1)/self.D))
        
        self.register_buffer('PE', PE.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.PE[:, :x.size(1), :]
        return x


class BasicTransformerBlock(nn.Module):

    def __init__(self, embedding_size, n_heads, mlp_size):
        super().__init__()
        self.D = embedding_size

        self.lnorm1 = nn.LayerNorm(self.D)
        self.mha = nn.MultiheadAttention(self.D, n_heads, batch_first=True)
        self.lnorm2 = nn.LayerNorm(self.D)
        self.mlp = nn.Sequential(
            nn.Linear(self.D, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, self.D)
        )
    
    def forward(self, x):
        x1 = self.lnorm1(x)
        x1, _ = self.mha(x1, x1, x1)
        x2 = x + x1

        x3 = self.lnorm2(x2)
        x3 = self.mlp(x3)
        out = x3 + x2

        return out


class ViTBackbone(nn.Module):

    def __init__(
            self, in_channels, embedding_dim,
            patch_size, max_patch_num,
            L, n_heads, mlp_size
    ):
        super().__init__()
        self.D = embedding_dim

        self.patch_embd = PatchEmbedding(
            in_channels=in_channels,
            patch_size=patch_size, embedding_dim=self.D
        )
        self.cls_embd = ClassEmbedding(embedding_dim=self.D)
        self.pos_embd = PositionalEmbeddings(
            embedding_size=self.D, max_patch_num=max_patch_num
        )

        self.transformer = nn.Sequential(
            *[
                BasicTransformerBlock(
                    embedding_size=self.D,
                    n_heads=n_heads, mlp_size=mlp_size
                ) for _ in range(L)
            ]
        )
    
    def forward(self, x):
        embd = self.patch_embd(x)
        embd = self.cls_embd(embd)
        embd = self.pos_embd(embd)

        tr_out = self.transformer(embd)[:,0]
        
        return tr_out


class ClassificationHead(nn.Module):

    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.D = embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.D, n_classes),
            # nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        out = self.classifier(x)
        return out

