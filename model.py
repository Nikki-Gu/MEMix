
from functools import partial
import timm
import torch
import torch.nn as nn
from timm.models.layers import PatchEmbed, Mlp, DropPath
import torchvision.models as models
# atten
class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0., return_attn=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attn = return_attn

    def forward(self, x):
        # print('return attn:',self.return_attn)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_softmax = attn.detach().clone()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.return_attn:
            return x, attn_softmax
        return x


# block
class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            merge=False,
            return_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              return_attn=return_attn)
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.return_attn = return_attn
        self.merge=merge

    def forward(self, x,y):
        if self.merge:
            x[:, 1:, :] = x[:, 1:, :] + y
        if self.return_attn:
            res = x
            x, attn = self.attn(self.norm1(x))
            x = res + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


from timm.models.layers import to_2tuple


# vit
class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=5,  # 1000,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            class_token=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            weight_init='',
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            merge=False,
            return_attn=False
    ):
        super().__init__()
        self.return_attn = return_attn
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        # use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            # bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.dist_token = None  # nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + self.num_tokens, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                merge= merge and i==0,
                return_attn=return_attn
                            and i == depth - 1
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)  # if not use_fc_norm else nn.Identity()

        self.pre_logits = nn.Identity()
        # Classifier Head
        # self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        self.depth=depth
        self.merge=merge

    def forward_features(self, x,y=None):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for i in range(self.depth):
            x = self.blocks[i](x,y)
        if self.return_attn:
            x, attn = x[0], x[1]
            attn = torch.mean(attn[:, :, 0, 1:], dim=1)  # attn from cls_token to images
            x = self.norm(x)
            if self.dist_token is None:
                return self.pre_logits(x[:, 0]), attn
            else:
                return x[:, 0], x[:, 1], attn
        else:
            x = self.norm(x)
            if self.dist_token is None:
                return self.pre_logits(x[:, 0])
            else:
                return x[:, 0], x[:, 1]

    def forward(self, x,y=None):
        #print('11111')
        x = self.forward_features(x,y)
        if self.return_attn:
            if self.head_dist is not None:
                x, x_dist, attn = self.head(x[0]), self.head_dist(x[1]), x[2]  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    # during inference, return the average of both classifier predictions
                    return x, x_dist, attn
                else:
                    return (x + x_dist) / 2, attn
            else:
                x, attn = x[0], x[1]
                x = self.head(x)
            return x, attn
        else:
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    # during inference, return the average of both classifier predictions
                    return x, x_dist
                else:
                    return (x + x_dist) / 2
            else:
                x = self.head(x)
            return x


class MixMe_Net(nn.Module):
    def __init__(self,img_size=224,
            patch_size=16,
            in_chans=2,
            num_classes=5,  # 1000,
            global_pool='token',
            embed_dim= 768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            return_attn=False,
            merge=False,
            pretrained=True):
        super(MixMe_Net, self).__init__()
        self.return_attn=return_attn
        model = VisionTransformer(return_attn=return_attn,merge=merge,
                                  num_classes=0,in_chans=in_chans,img_size=img_size,patch_size=patch_size,
                                  embed_dim=embed_dim,depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio)  # ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
        resmodel = models.resnet18(pretrained=pretrained)


        self.resmodel=nn.Sequential(*list(resmodel.children())[:-3])
        if pretrained: 
            pre_model = timm.create_model('vit_base_patch'+str(patch_size)+'_224', num_classes=num_classes, pretrained=True)
            pre_dict = pre_model.state_dict()
            model_dict = model.state_dict()
            conv1_w = torch.mean(pre_model.patch_embed.proj.weight, dim=1, keepdim=True).repeat(1, in_chans, 1, 1)
            pre_dict['patch_embed.proj.weight'] = conv1_w
            pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
            model_dict.update(pre_dict)
            model.load_state_dict(model_dict)
        self.merge=merge
        self.model=model
        self.normal=nn.BatchNorm1d(embed_dim)
        self.head=nn.Linear(embed_dim,num_classes)
        self.res_embed=nn.Conv2d(256,768,kernel_size=1,stride=1)


    def forward(self, x,y=None):
        if self.merge==True:
            y = self.resmodel(y)  # [B,256,14,14]
            y=self.res_embed(y)#[B,768,14,14]
            B,D,_,_=y.shape
            y=y.view(B,D,-1)
            y=y.permute(0,2,1)#[B,196,768]
        x = self.model(x,y)  # [B,768]
        if self.return_attn:
            x,attn=x
            embed=x
            x=self.head(x)
            return x,attn, embed
        else:
            embed=x
            x = self.head(x)
            return x,embed
