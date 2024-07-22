# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
import torch.nn as nn
#from timm.models.layers.mlp import Mlp
from timm.layers.mlp import Mlp

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 vocab_size=30522, text_max_length=128, weight_text_loss=0.5,
                 sim_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        self.sim_loss = sim_loss
        self.weight_text_loss = weight_text_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim) #Tom: Tom: layer norm = True ?
        num_patches = self.patch_embed.num_patches

        # vocab_size - The size of the vocabulary in BERT-base
        self.text_embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.text_max_length = text_max_length
        self.text_pos_embed = nn.Parameter(torch.zeros(1, text_max_length, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.mask_token_text = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        # text decoder positional embedding
        self.text_decoder_pos_embed = nn.Parameter(torch.zeros(1, text_max_length, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.decoder_pred_text = nn.Linear(decoder_embed_dim, vocab_size)

        # --------------------------------------------------------------------------

        self.mlp_img = Mlp(in_features=decoder_embed_dim, hidden_features=int(decoder_embed_dim * mlp_ratio))
        self.mlp_text = Mlp(in_features=decoder_embed_dim, hidden_features=int(decoder_embed_dim * mlp_ratio))
        
        # image/text type embedding
        self.encoder_image_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_image_type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.encoder_text_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_text_type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # text encoder embedding
        text_pos_embed = get_1d_sincos_pos_embed(self.text_pos_embed.shape[-1], self.text_max_length)
        self.text_pos_embed.data.copy_(torch.from_numpy(text_pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # text decoder embedding
        text_decoder_pos_embed = get_1d_sincos_pos_embed(self.text_decoder_pos_embed.shape[-1], self.text_max_length)
        self.text_decoder_pos_embed.data.copy_(torch.from_numpy(text_decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_token_text, std=.02)

        torch.nn.init.normal_(self.encoder_image_type_embedding, std=.02)
        torch.nn.init.normal_(self.decoder_image_type_embedding, std=.02)
        torch.nn.init.normal_(self.encoder_text_type_embedding, std=.02)
        torch.nn.init.normal_(self.decoder_text_type_embedding, std=.02)

        # init text embedding layer
        self.text_embedding_layer.weight.data.normal_(mean=0.0, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, t, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # add image type embed
        x = x + self.encoder_image_type_embedding

        # text embed vectors
        t = self.text_embedding_layer(t)
        # add pos embed
        t = t + self.text_pos_embed[:, :t.shape[1], :]
        # add text type embed
        t = t + self.encoder_text_type_embedding

        # print('x shape (original): ', x.shape)
        # print('t shape (original): ', t.shape)
        # print('input x (img + text, unmasked) shape: ', torch.cat((x, t), axis=1).shape)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        t, mask_text, ids_restore_text = self.random_masking(t, mask_ratio)

        # merge image (x) and text (t)
        length_img_tokens = x.shape[1] + 1
        # print('input x shape: ', x.shape)
        x = torch.cat((x, t), axis=1)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print('input x (cls + img + text) shape: ', x.shape)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, mask_text, ids_restore_text, length_img_tokens

    def forward_decoder_captioning(self, x, ids_restore, ids_restore_text, length_img_tokens, nb_masks=1):
        #add nb_masks tokens at the end of the sentence part. Used for zero shot.
         
        # embed tokens
        x = self.decoder_embed(x)

        # print(ids_restore)
        # print(ids_restore_text)
        # print(length_img_tokens)
        # print(x.shape)
        # raise RuntimeError

        # append mask tokens to sequence (image)
        x_img = x[:, :length_img_tokens, :]
        mask_tokens = self.mask_token.repeat(x_img.shape[0], ids_restore.shape[1] + 1 - x_img.shape[1], 1)
        x_img_ = torch.cat([x_img[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_img_ = torch.gather(x_img_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_img.shape[2]))  # unshuffle

        # append mask tokens to sequence (text)
        x_text = x[:, length_img_tokens:, :]
        mask_tokens_text = self.mask_token_text.repeat(x_text.shape[0], ids_restore_text.shape[1] - x_text.shape[1], 1)
        x_text_ = torch.cat([x_text, mask_tokens_text], dim=1)  # no cls token
        x_text_ = torch.gather(x_text_, dim=1, index=ids_restore_text.unsqueeze(-1).repeat(1, 1, x_text.shape[2]))  # unshuffle
        x_text_ = torch.cat([x_text_, self.mask_token_text.repeat(x_img.shape[0], nb_masks, 1)], dim=1)
        # print('x_text_.shape: ', x_text_.shape)
        # print('x_img_.shape: ', x_img_.shape)
        # x = torch.cat([x[:, :1, :], x_img_, x_text_], dim=1)  # append cls token

        x_img_cls_ = torch.cat([x[:, :1, :], x_img_], dim=1)  # append cls token
        # add pos embed (img)
        x_img_cls_ = x_img_cls_ + self.decoder_pos_embed
        # add type embed (img)
        x_img_cls_ = x_img_cls_ + self.decoder_image_type_embedding
        
        # add pos embed (text)
        x_text_ = x_text_ + self.text_decoder_pos_embed[:, :x_text_.shape[1], :]
        # add type embed (img)
        x_text_ = x_text_ + self.decoder_text_type_embedding

        # merge (image) and (text)
        x = torch.cat([x_img_cls_, x_text_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        num_img_tokens = self.patch_embed.num_patches
        # seperate image tokens and text tokens
        # z_img -> MLP -> img decoder -> input patches
        z_img = x[:, :num_img_tokens, :]
        z_img = self.mlp_img(z_img)
        z_img = self.decoder_pred(z_img)

        # z_text -> MLP -> text decoder -> input ids
        z_text = x[:, num_img_tokens:, :]
        z_text = self.mlp_text(z_text)
        z_text = self.decoder_pred_text(z_text)

        return z_img, z_text

    def forward_decoder(self, x, ids_restore, ids_restore_text, length_img_tokens):
        # embed tokens
        x = self.decoder_embed(x)

        # print(ids_restore)
        # print(ids_restore_text)
        # print(length_img_tokens)
        # print(x.shape)
        # raise RuntimeError

        # append mask tokens to sequence (image)
        x_img = x[:, :length_img_tokens, :]
        mask_tokens = self.mask_token.repeat(x_img.shape[0], ids_restore.shape[1] + 1 - x_img.shape[1], 1)
        x_img_ = torch.cat([x_img[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_img_ = torch.gather(x_img_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_img.shape[2]))  # unshuffle

        # append mask tokens to sequence (text)
        x_text = x[:, length_img_tokens:, :]
        mask_tokens_text = self.mask_token_text.repeat(x_text.shape[0], ids_restore_text.shape[1] - x_text.shape[1], 1)
        x_text_ = torch.cat([x_text, mask_tokens_text], dim=1)  # no cls token
        x_text_ = torch.gather(x_text_, dim=1, index=ids_restore_text.unsqueeze(-1).repeat(1, 1, x_text.shape[2]))  # unshuffle
        # print('x_text_.shape: ', x_text_.shape)
        # print('x_img_.shape: ', x_img_.shape)
        # x = torch.cat([x[:, :1, :], x_img_, x_text_], dim=1)  # append cls token

        x_img_cls_ = torch.cat([x[:, :1, :], x_img_], dim=1)  # append cls token
        # add pos embed (img)
        x_img_cls_ = x_img_cls_ + self.decoder_pos_embed
        # add type embed (img)
        x_img_cls_ = x_img_cls_ + self.decoder_image_type_embedding
        
        # add pos embed (text)
        x_text_ = x_text_ + self.text_decoder_pos_embed[:, :x_text_.shape[1], :]
        # add type embed (img)
        x_text_ = x_text_ + self.decoder_text_type_embedding

        # merge (image) and (text)
        x = torch.cat([x_img_cls_, x_text_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        # x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        num_img_tokens = self.patch_embed.num_patches
        # seperate image tokens and text tokens
        # z_img -> MLP -> img decoder -> input patches
        z_img = x[:, :num_img_tokens, :]
        z_img = self.mlp_img(z_img)
        z_img = self.decoder_pred(z_img)

        # z_text -> MLP -> text decoder -> input ids
        z_text = x[:, num_img_tokens:, :]
        z_text = self.mlp_text(z_text)
        z_text = self.decoder_pred_text(z_text)

        return z_img, z_text

    def forward_loss_image(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_text(self, pred_logits, input_ids, mask):
        # will manually reduce the loss afterwards
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # reshaping the tensors
        # reshape input_ids to [batch_size*seq_length]
        input_ids = input_ids.view(-1)
        # reshape pred_logits to [batch_size*seq_length, vocab_size]
        pred_logits = pred_logits.view(-1, pred_logits.size(-1))
        # reshape mask to [batch_size*seq_length]
        mask = mask.view(-1)

        # calculate the losses
        losses = loss_fct(pred_logits, input_ids)

        # apply the mask - simply multiply it with loss vectors
        masked_losses = losses * mask.float()

        # calculate mean loss - but only for the masked tokens
        loss = masked_losses.sum() / mask.float().sum()

        return loss

    def forward_loss_sim_img_text(self, img_features, text_features):
        cos_criterion = nn.CosineSimilarity(dim=1)
        loss = 1.0 - cos_criterion(img_features, text_features).mean()
        return loss
        
    def forward_text(self, texts):
        # text embed vectors
        t = self.text_embedding_layer(texts)
        # add pos embed
        t = t + self.text_pos_embed[:, :t.shape[1], :]
        # add text type embed
        t = t + self.encoder_text_type_embedding

        # apply Transformer blocks
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t

    def forward_img(self, imgs):
        # embed patches
        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # add image type embed
        x = x + self.encoder_image_type_embedding

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
        
    def forward_loss(self, imgs, texts, mask_ratio=0.75):
        # latent, mask, ids_restore = self.forward_encoder(imgs, texts, mask_ratio)
        # encoder
        latent, mask, ids_restore, mask_text, ids_restore_text, length_img_tokens = self.forward_encoder(imgs, texts, mask_ratio)
        # decoder
        pred_img, pred_text = self.forward_decoder(latent, ids_restore, ids_restore_text, length_img_tokens)  # [N, L, p*p*3]

        # compute loss on images
        loss_img = self.forward_loss_image(imgs, pred_img, mask)
        # compute loss on text
        loss_text = self.forward_loss_text(pred_text, texts, mask_text)
        if self.sim_loss:
            # forward unmasked image/text input and obtain features
            image_features = torch.mean(self.forward_img(imgs), dim=1)
            text_features = torch.mean(self.forward_text(texts), dim=1)
            # compute loss on similarity between text and image
            loss_sim = self.forward_loss_sim_img_text(image_features, text_features)
            # compute overall loss
            loss = loss_img + self.weight_text_loss * loss_text + loss_sim
        else:
            # compute overall loss
            loss = loss_img + self.weight_text_loss * loss_text
        return loss, (loss_img, pred_img, mask), (loss_text, pred_text, mask_text) ##easier for functorchloss, pred_img, mask
    def forward(self, imgs, texts, mask_ratio=0.75):
        # latent, mask, ids_restore = self.forward_encoder(imgs, texts, mask_ratio)
        # encoder
        latent, mask, ids_restore, mask_text, ids_restore_text, length_img_tokens = self.forward_encoder(imgs, texts, mask_ratio)
        # decoder
        pred_img, pred_text = self.forward_decoder(latent, ids_restore, ids_restore_text, length_img_tokens)  # [N, L, p*p*3]

        # compute overall loss
        target_img = self.patchify(imgs)
        target_text = texts#.view(-1)
        pred_text=pred_text.view(-1, pred_text.size(-1))
        mask_text = mask_text#.view(-1)
        return (pred_img, mask, target_img), (pred_text, mask_text, target_text) ##easier for functorch


######AUTOREGRESSIVE#####


from model.torch_transformer import decoder
from einops import rearrange
import torch.nn.functional as F
def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class MaskedAutoencoderViT_autoregressive(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 vocab_size=30522, text_max_length=128, weight_text_loss=0.5,
                 sim_loss=False, activation="relu", bias=True, qkv_bias=True):
        super().__init__()

        # --------------------------------------------------------------------------
        self.sim_loss = sim_loss
        self.weight_text_loss = weight_text_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim) #Tom: Tom: layer norm = True ?
        num_patches = self.patch_embed.num_patches

        # vocab_size - The size of the vocabulary in BERT-base
        #self.text_embedding_layer = nn.Embedding(vocab_size, embed_dim)

        #13/06 TOM
        self.text_embedding_layer_SimVLM = nn.Embedding(vocab_size, decoder_embed_dim)

        self.text_max_length = text_max_length
        self.text_pos_embed = nn.Parameter(torch.zeros(1, text_max_length, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)#already Gelu
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=False)

        self.mask_token_text = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=False)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        # text decoder positional embedding
        self.text_decoder_pos_embed = nn.Parameter(torch.zeros(1, text_max_length, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_blocks = nn.ModuleList([
        #     # Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])
        self.decoder=decoder(d_model=decoder_embed_dim, nhead=decoder_num_heads, depth=decoder_depth,
            dim_feedforward=mlp_ratio*decoder_embed_dim, dropout=0, activation=activation, bias=bias)

        #self.decoder_norm = norm_layer(decoder_embed_dim)
        #self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.decoder_pred_text = nn.Sequential(nn.LayerNorm(decoder_embed_dim),nn.Linear(decoder_embed_dim,  vocab_size))

        # --------------------------------------------------------------------------

        #self.mlp_img = Mlp(in_features=decoder_embed_dim, hidden_features=int(decoder_embed_dim * mlp_ratio))
        #self.mlp_text = Mlp(in_features=decoder_embed_dim, hidden_features=int(decoder_embed_dim * mlp_ratio))
        
        # image/text type embedding
        self.encoder_image_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        self.decoder_image_type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=False)
        self.encoder_text_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)
        self.decoder_text_type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=False)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # text encoder embedding
        text_pos_embed = get_1d_sincos_pos_embed(self.text_pos_embed.shape[-1], self.text_max_length)
        self.text_pos_embed.data.copy_(torch.from_numpy(text_pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # text decoder embedding
        text_decoder_pos_embed = get_1d_sincos_pos_embed(self.text_decoder_pos_embed.shape[-1], self.text_max_length)
        self.text_decoder_pos_embed.data.copy_(torch.from_numpy(text_decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.mask_token_text, std=.02)

        torch.nn.init.normal_(self.encoder_image_type_embedding, std=.02)
        torch.nn.init.normal_(self.decoder_image_type_embedding, std=.02)
        torch.nn.init.normal_(self.encoder_text_type_embedding, std=.02)
        torch.nn.init.normal_(self.decoder_text_type_embedding, std=.02)

        # init text embedding layer
        #self.text_embedding_layer.weight.data.normal_(mean=0.0, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio != 0:
            x, mask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore, ids_shuffle = None, None, None

        # merge image (x) and text (t)
        length_img_tokens = x.shape[1] + 1

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add image type embed
        x = x + self.encoder_image_type_embedding

        # save length of non-masked tokens
        length_img_tokens = x.shape[1]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_shuffle, length_img_tokens

    def loglikelyhood(self, imgs, texts, mask_ratio=0.75, target_text_length=77, device=None):
        # latent, mask, ids_restore = self.forward_encoder(imgs, texts, mask_ratio)
        # encoder
        latent, mask, ids_restore, ids_shuffle, length_img_tokens = self.forward_encoder(imgs, mask_ratio)
        # decoder
        x = self.decoder_embed(latent)
        
        t = self.text_embedding_layer_SimVLM(texts)
        t = t + self.text_decoder_pos_embed[:, :t.shape[1], :]
        t = t + self.decoder_text_type_embedding
        assert mask_ratio==0
        #Adding Image Posiotion embedding
        x = x + self.decoder_pos_embed
        x = x + self.decoder_image_type_embedding 
        tgt_mask = generate_square_subsequent_mask(target_text_length).to(device)
        key_padding_mask=torch.zeros(x.shape[0],x.shape[1],dtype=torch.bool, device=device)#useless
        tgt_key_padding_mask=texts==0#shouldnt be used bause we do not use padding tokens
        # t = rearrange(t, 'b n c->n b c')
        # x = rearrange(x, 'b n c->n b c')
        output = self.decoder(t, x, is_causal=1, 
                              tgt_mask=tgt_mask, memory_mask=None,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=key_padding_mask)
        output = self.decoder_pred_text(output)
        # output = rearrange(output, 'n b c -> b c n')
        output = rearrange(output, 'b n c -> b c n')
        target_text = texts[:,1:]
        loss=F.cross_entropy(output[:,:,:-1],target_text)
        return loss
    def linear_prob_forward(self, imgs):
        latent, mask, ids_restore, ids_shuffle, length_img_tokens = self.forward_encoder(imgs, 0)
        x = self.decoder_embed(latent)
        x = x + self.decoder_pos_embed
        x = x + self.decoder_image_type_embedding
        return(x)
    # @staticmethod
    #@torch.cuda.amp.custom_fwd
    def forward(self, imgs, texts, mask_ratio=0.75,prob_parallel_decoding=0.0, target_text_length=77, device=None, reduction="none"):        
        # encoder
        latent, mask, ids_restore, ids_shuffle, length_img_tokens = self.forward_encoder(imgs, mask_ratio)
        # decoder
        x = self.decoder_embed(latent)

        #Adding Image Posiotion embedding
        if mask_ratio != 0:
            ### append mask tokens to sequence (used in MAE paper)
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            # add pos embed
            x = x + self.decoder_pos_embed
            
            # ### do not append mask tokens 
            # image_pos_embed = self.decoder_pos_embed.repeat(x.shape[0], 1, 1)
            # indicies = ids_shuffle[:, :length_img_tokens].unsqueeze(-1).repeat(1, 1, image_pos_embed.shape[2])
            # # add pos embed
            # x = x + image_pos_embed.gather(1, indicies)

        else:
            x = x + self.decoder_pos_embed
        # add image type embedding
        x = x + self.decoder_image_type_embedding 

        # get text embedding
        

        tgt_mask = generate_square_subsequent_mask(target_text_length).to(device)
        key_padding_mask=torch.zeros(x.shape[0],x.shape[1],dtype=torch.bool, device=device)#useless
        tgt_key_padding_mask=texts==0#.zeros(texts.shape[0],texts.shape[1],dtype=torch.bool, device=device)#useless
        
        # t = rearrange(t, 'b n c->n b c')
        # x = rearrange(x, 'b n c->n b c')
        t = self.text_embedding_layer_SimVLM(texts)
        import random
        if random.uniform(0,1)< prob_parallel_decoding:
            # prepare t_mask
            t_mask = torch.zeros(t.shape).to(device) + self.mask_token_text +0*t #otherwise text_embedding_layer_SimVLM is problematic for ghostnorm
            t_mask = t_mask + self.text_decoder_pos_embed[:, :t_mask.shape[1], :]
            t_mask = t_mask + self.decoder_text_type_embedding
            # parallel decoding loss
            output = self.decoder(t_mask, x, 
                                  is_causal=0,
                                  tgt_mask=tgt_mask, 
                                  memory_mask=None, 
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=key_padding_mask)
        else:
            t = self.text_embedding_layer_SimVLM(texts)
            t = t + self.text_decoder_pos_embed[:, :t.shape[1], :]
            t = t + self.decoder_text_type_embedding
            output = self.decoder(t, x, 
                                is_causal=1,
                                tgt_mask=tgt_mask, 
                                memory_mask=None, 
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=key_padding_mask)
        # next word prediction
        output = self.decoder_pred_text(output)
        output = rearrange(output, 'b n c -> b c n')
        labels = texts[:, 1:, ]
        logits = output[:, :, :-1]
        nb_pad = (labels!=0).sum(dim=1)
        loss = F.cross_entropy(logits, labels,ignore_index=0,reduction="none").sum(dim=1)/(nb_pad)
        if reduction=="none":
            return(loss)
        else:
            return loss.mean()
        target_text = texts[:, 1:]
        loss = F.cross_entropy(output[:, :, :-1], target_text, ignore_index=0)
        return loss

    def forward_next_word(self, imgs, texts, mask_ratio=0.75,target_text_length=77, device=None):
        # only outputs the next word. To be used for generation/zero-shot
        # encoder
        latent, mask, ids_restore, ids_shuffle, length_img_tokens = self.forward_encoder(imgs, mask_ratio)
        # decoder
        x = self.decoder_embed(latent)
        t = self.text_embedding_layer_SimVLM(texts)
        t = t + self.text_decoder_pos_embed[:, :t.shape[1], :]
        t = t + self.decoder_text_type_embedding
        if mask_ratio != 0:
            ### append mask tokens to sequence (used in MAE paper)
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            # add pos embed
            x = x + self.decoder_pos_embed

            # ### do not append mask tokens 
            # image_pos_embed = self.decoder_pos_embed.repeat(x.shape[0], 1, 1)
            # indicies = ids_shuffle[:,:length_img_tokens].unsqueeze(-1).repeat(1,1,image_pos_embed.shape[2])
            # x = x + image_pos_embed.gather(1, indicies)
        else:
            x = x + self.decoder_pos_embed
        x = x + self.decoder_image_type_embedding
        tgt_mask = generate_square_subsequent_mask(target_text_length).to(device)
        key_padding_mask=torch.zeros(x.shape[0],x.shape[1],dtype=torch.bool, device=device)#useless
        output = self.decoder(t, x, is_causal=1, tgt_mask=tgt_mask, memory_mask=None,tgt_key_padding_mask=None,memory_key_padding_mask=key_padding_mask)
        output = self.decoder_pred_text(output)
        output = rearrange(output, 'b n c -> b c n')
        return(output[:,:,-1])
        #return (pred_img, mask, target_img), (pred_text, mask_text, target_text) ##easier for functorch

    def forward_caption(self, imgs, mask_ratio=0,target_text_length=77, device=None):
        # only outputs the next word. To be used for generation/zero-shot
        # encoder
        latent, mask, ids_restore, ids_shuffle, length_img_tokens = self.forward_encoder(imgs, mask_ratio)
        x = self.decoder_embed(latent)
        x = x + self.decoder_pos_embed
        x = x + self.decoder_image_type_embedding
        text = []
        next_token = 101
        iter = 0
        text.append(next_token)
        while next_token !=102 and iter < 40:
            t = self.text_embedding_layer_SimVLM(torch.Tensor([text]).int().cuda())
            t = t + self.text_decoder_pos_embed[:, :t.shape[1], :]
            t = t + self.decoder_text_type_embedding
            tgt_mask = generate_square_subsequent_mask(target_text_length).to(device)
            key_padding_mask=torch.zeros(x.shape[0],x.shape[1],dtype=torch.bool, device=device)#useless
            output = self.decoder(t, x, is_causal=1, tgt_mask=tgt_mask, memory_mask=None,tgt_key_padding_mask=None,memory_key_padding_mask=key_padding_mask)
            output = self.decoder_pred_text(output)
            output = rearrange(output, 'b n c -> b c n')
            next_token = int(torch.argmax(output[:,:,-1]))
            iter+=1
            text.append(next_token)
        return(text)



def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_autoregressive_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, #change decoder_depth to 6 to get the same archtiecture as DM paper
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_autoregressive_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_autoregressive_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_autoregressive_nobias_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=12, #change decoder_depth to 6 to get the same archtiecture as DM paper
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), activation="gelu", bias=False, qkv_bias=False,**kwargs)
    return model


def mae_vit_large_patch16_autoregressive_nobias_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=768, decoder_depth=6, decoder_num_heads=12, #change decoder_depth to 6 to get the same archtiecture as DM paper
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), activation="gelu", bias=False, qkv_bias=False,**kwargs)
    return model


def mae_vit_small_patch16_autoregressive_nobias_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=576, depth=12, num_heads=12,
        decoder_embed_dim=576, decoder_depth=6, decoder_num_heads=12, #change decoder_depth to 6 to get the same archtiecture as DM paper
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), activation="gelu", bias=False, qkv_bias=False,**kwargs)
    return model


def mae_vit_tiny_patch16_autoregressive_nobias_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=384, decoder_depth=6, decoder_num_heads=6, #change decoder_depth to 6 to get the same archtiecture as DM paper
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), activation="gelu", bias=False, qkv_bias=False,**kwargs)
    return model

def mae_vit_nano_patch16_autoregressive_nobias_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT_autoregressive(
        patch_size=16, embed_dim=192, depth=12, num_heads=6,
        decoder_embed_dim=192, decoder_depth=6, decoder_num_heads=6, #change decoder_depth to 6 to get the same archtiecture as DM paper
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), activation="gelu", bias=False, qkv_bias=False,**kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_autoregressive = mae_vit_base_patch16_autoregressive_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16_autoregressive_nobias = mae_vit_base_patch16_autoregressive_nobias_dec512d8b  # decoder: 768 dim, 6 blocks
mae_vit_large_patch16_autoregressive = mae_vit_large_patch16_autoregressive_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_autoregressive = mae_vit_huge_patch14_autoregressive_dec512d8b 

mae_vit_large_patch16_autoregressive_nobias = mae_vit_large_patch16_autoregressive_nobias_dec512d8b  # decoder: 768 dim, 6 blocks
mae_vit_small_patch16_autoregressive_nobias = mae_vit_small_patch16_autoregressive_nobias_dec512d8b  # decoder: 576 dim, 6 blocks
mae_vit_tiny_patch16_autoregressive_nobias = mae_vit_tiny_patch16_autoregressive_nobias_dec512d8b  # decoder: 384 dim, 6 blocks
mae_vit_nano_patch16_autoregressive_nobias = mae_vit_nano_patch16_autoregressive_nobias_dec512d8b  # decoder: 384 dim, 6 block