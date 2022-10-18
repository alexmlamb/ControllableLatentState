
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vqema import VectorQuantizerEMA

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes_1, num_classes_2, num_classes_3, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., args):
        super().__init__()

        self.args = args

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_1 = nn.Sequential(
            nn.Linear(dim, num_classes_1)
        )

        self.mlp_head_2 = nn.Sequential(
            nn.Linear(dim, num_classes_2)
        )

        self.mlp_head_3 = nn.Sequential(
            nn.Linear(dim, num_classes_3)
        )

        self.vq = VectorQuantizerEMA(dim, 528)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x, loss, indices = self.vq(x)

        x = self.to_latent(x)

        if self.args.task == 'pred_gt':
            pred_gt = self.mlp_head_1(x)
            pred_action = self.mlp_head_2(x.detach())
            pred_time_ind = self.mlp_head_3(x.detach())

        elif self.args.task == 'pred_exo':
            pred_gt = self.mlp_head_1(x.detach())
            pred_action = self.mlp_head_2(x.detach())
            pred_time_ind = self.mlp_head_3(x)

        return pred_gt, pred_action, pred_time_ind, loss


'''
ViT GENIK WITH GAUSSIAN AND VQ BOTTLENECK
'''


class ViTGenik(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., args, use_gb, vq):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.vq = vq
        self.use_gb = use_gb
        self.args = args

        print("ViTGenik init  ================== BEGIN ==================== ")
        print("gaussian bottleneck: " + str(self.use_gb))
        print("VQ: " + str(self.vq))
        print("args: " + str(args))
        print("image_size: " + str(image_size))
        print("patch_size: " + str(patch_size))
        print("dim: " + str(dim))
        print("depth: " + str(depth))
        print("heads: " + str(heads))
        print("mlp_dim: " + str(mlp_dim))
        print("pool: " + str(pool))
        print("channels: " + str(channels))
        print("dim_head: " + str(dim_head))
        print("dropout: " + str(dropout))
        print("emb_dropout: " + str(emb_dropout))
        print("ViTGenik init  ================== END ==================== ")

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.k_embedding = nn.Embedding(200, dim // 2 if self.use_gb else dim)


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.decoder_linear = nn.Linear(dim//2, 512*3*5)

        self.decoder_conv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
            nn.Sigmoid())


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.1)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.linear_projection_1 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.linear_projection_2 = nn.Linear(dim // 2 if self.use_gb else dim, dim // 2 if self.use_gb else dim)

        self.post_enc = nn.Linear(dim // 2, dim//2)

        self.mlp_head_1 = nn.Sequential(nn.Linear(dim // 2 if self.use_gb else dim, dim), nn.GELU(),
            nn.Linear(dim, 1)
        )

        self.mlp_head_2 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 1)
        )

        self.mlp_head_3 = nn.Sequential(nn.Linear(dim // 2 if self.use_gb else dim, dim), nn.GELU(),
            nn.Linear(dim, 1)
        )

        self.ng = 1
        print('using ngroups', self.ng)
        self.vq = VectorQuantizerEMA(dim//(2*self.ng), args.ncodes, 1)

        self.action_mlp = nn.Sequential(nn.Linear(3 * (dim //2) if self.use_gb else 3 * dim, 2 * dim), nn.BatchNorm1d(2*dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.action_mlp_1 = nn.Linear(dim, 1)
        self.action_mlp_2 = nn.Linear(dim, 1)
        self.action_mlp_3 = nn.Linear(dim, 1)

        self.ce_loss = nn.CrossEntropyLoss()


    # def forward(self, img, img_k, k, action_y1, action_y2, vq = False, initialize = False):
    def forward(self, img, img_k, k, vq = False, use_gb=False, initialize = False):
        
        raw_img = img*1.0

        img = torch.cat((img, img_k), dim = 0)

        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)


        x = self.transformer(x)


        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        loss = 0
        x = self.linear_projection_1(x)

        klb_loss = 0

        if self.use_gb and use_gb:
            d = x.shape[-1]
            mu = x[:, :d // 2]

            if self.training:
                std = torch.nn.functional.softplus(x[:,d//2:])
                klb_loss = (mu**2 + std**2 - 2*torch.log(std)).sum(dim=1).mean() * self.args.kl_penalty
                x = mu + torch.randn_like(std) * std
            else:    
                klb_loss = (mu.sum())*0.0
                x = mu

            x = self.post_enc(x)

        if self.use_gb and not use_gb:
            d = x.shape[-1]
            mu = x[:, :d // 2]
            x = mu
            x = self.post_enc(x)
            klb_loss = (mu.sum())*0.0

        if self.vq and vq:
            b, d = x.shape
            x = x.reshape(b, self.ng, d // self.ng)
            x = x.reshape(b * self.ng, 1, d // self.ng)
            #print('x going to vq', x.shape)
            #x = x.unsqueeze(1)
            x, loss, indices = self.vq(x)
            x = x.squeeze(1)
            x = x.reshape(b, self.ng, d // self.ng)
            x = x.reshape(b, d)
            #x = self.linear_projection_2(x)
        else:
            indices = None

        x, x_k = torch.chunk(x, 2, dim = 0)

        if indices is not None:
            indices, _ = torch.chunk(indices, 2, dim=0)

        h_rep = x*1.0

        rec_decoder = True
        rec_ts_detach = (not self.args.train_ae)

        if rec_decoder:
            if rec_ts_detach:
                d = self.decoder_linear(h_rep.detach())
            else:
                d = self.decoder_linear(h_rep)
            d = d.reshape((h_rep.shape[0],512,3,5))
            d = self.decoder_conv(d)

            loss += 0.01 * ((d - raw_img)**2).sum(dim=(1,2,3)).mean()
        else:
            d = 0.0

        k = self.k_embedding(k)

        x_cat = torch.cat((x, x_k, k), dim = 1)
        x_cat = self.action_mlp(x_cat)
        action_pred_1 = self.action_mlp_1(x_cat)
        action_pred_2 = self.action_mlp_2(x_cat)
        action_pred_3 = self.action_mlp_3(x_cat)

        x = self.to_latent(x)

        total_loss = loss + klb_loss

        pred_action = action_pred_1

        return action_pred_1, action_pred_2, action_pred_3, h_rep, d, indices, total_loss

    def encode(self, img):
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return x

