# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#script for 0shot eval of M3AE

#import models_mae
import models_mae
import torch
import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
from transformers import BertTokenizer
import json
import argparse
from tqdm import tqdm
import torchvision

from common.utils import bool_flag
from util.trie import get_trie, RowGuide

from model.torch_transformer import decoder
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn


def generate_square_subsequent_mask(sz: int):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_args_parser():
    parser = argparse.ArgumentParser('One shot eval throught captionning', add_help=False)
    parser.add_argument('--checkpoint_path', default='/checkpoint/yaodongyu/experiments/M3AE/9219026/checkpoint-25.pth', type=str,help='checkpoint')
    parser.add_argument('--dataset', default="ImageNet", choices = ["ImageNet", "CIFAR10", "CIFAR100"])
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str, help='ImageNet path')
    parser.add_argument('--partition', default='val', type=str, help='train/val/test...')
    parser.add_argument('--type', default="trie", type=str, choices = ["trie", "likelihood", "1token"], help='what technique to use for 0shot')
    #parser.add_argument('--trie', default=True, type=bool_flag, help='using prefix tree. If false, using the old version to only try 1-token labels.')
    parser.add_argument('--seed', type=int, default=0, help="seed")
    parser.add_argument('--mask_ratio', type=float, default=0, help="mask ratio")
    parser.add_argument('--model', type=str, default='mae_vit_base_patch16_autoregressive_nobias', help="what model to use")
    parser.add_argument('--nb_eval', type=int, default=10000, help="nb of images to evaluate")
    parser.add_argument('--target_txt_len', type=int, default=40, help="target_txt_len")

    return parser

def labels_lengths(tokenizer, path, dataset="ImageNet"):
    print(f"evaluating zero-shot for {dataset}")
    lengths = {}
    f = open(path)
    dic = json.load(f)
    dic2 = {}
    mask = [0 for i in range(30522)]#vocab size
    for key, value in dic.items():
        aux = [tokenizer.batch_encode_plus([x],add_special_tokens=True, return_tensors='pt')['input_ids'] for x in value.split(",")]
        dic2[key] = [x[0][1:-1].tolist() for x in aux]
        aux2 = []
        for x in dic2[key]:
            aux2.append(len(x))
            if len(x)==1:
                mask[x[0]]=1
        lengths[key]=aux2
    mask = torch.Tensor(mask)
    return lengths, dic2, mask

def labels_lengths_with_BOS(tokenizer, path, dic = None):
    lengths = {}
    if dic==None:
        f = open(path)
        dic = json.load(f)
    dic2 = {}
    mask = [0 for i in range(30522)]#vocab size
    for key, value in dic.items():
        aux = [tokenizer.batch_encode_plus([x],add_special_tokens=True, return_tensors='pt')['input_ids'] for x in value.split(",")]
        dic2[key] = [x[0].tolist() for x in aux]
        aux2 = []
        for x in dic2[key]:
            aux2.append(len(x))
            if len(x)==1:
                mask[x[0]]=1
        lengths[key]=aux2
    mask = torch.Tensor(mask)
    return lengths, dic2, mask

#

def main(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    #model
    model = models_mae.__dict__[args.model](text_max_length=args.target_txt_len)
    model.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu') #why cpu?
    model.load_state_dict(checkpoint['model'])
    model.eval()

    #tokinizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.dataset=="ImageNet":
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, args.partition), transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths(tokenizer, "labels/imagenet_labels.json")
    elif args.dataset=="CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths(tokenizer, "labels/cifar10_labels.json")
    elif args.dataset=="CIFAR100":
        dataset_train = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths(tokenizer, "labels/cifar100_labels.json")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        num_workers=10,
        pin_memory="store_true",
        drop_last=True,
    )
    proportions = {'1': 0, '2': 0}

    good=0
    count = 0

    text = ['this is a photo of a ']
    texts_tokenized_input = tokenizer.batch_encode_plus(text,add_special_tokens=True,return_tensors='pt')
    input_ids_ = texts_tokenized_input['input_ids'].to(device)
    input_ids_ = input_ids_[0][:-1].unsqueeze(0)
    t = get_trie(dic2)
    guide = RowGuide(t)
    for (image,label) in tqdm(data_loader_train):
        image = image.squeeze(0).to(device)
        next_token, end = 0, False
        while not(end):
            assert "autoregressive" in args.model
            input = torch.Tensor([input_ids_[0].tolist()+guide.values]).to(device).int()
            distribution = model.forward_next_word(image.unsqueeze(0), input, mask_ratio=args.mask_ratio,target_text_length=len(input[0]), device=device)[0]
            next_token, end = guide.next(distribution)
        if guide.values[:-1] in dic2[str(label.item())]:
            good+=1
        count+=1
        guide = RowGuide(t)
        if count % 500 == 0:
            print('good/count: ', good/count)
    print(good/count)


def input_ids_from_text(text, tokenizer):
    texts_tokenized_input = tokenizer.batch_encode_plus(text,add_special_tokens=True,return_tensors='pt')
    input_ids_ = texts_tokenized_input['input_ids'].cuda()
    input_ids_ = input_ids_[0][:-1].unsqueeze(0)
    return input_ids_

def predicted_label(model, input_ids_, image, dic2, device):
    res = -1
    loss_res = 1000000
    image = image.squeeze(0).to(device)
    latent, _, _, _, _ = model.forward_encoder(image.unsqueeze(0), 0)
    # decoder
    x = model.decoder_embed(latent)
    #Adding Image Posiotion embedding
    x = x + model.decoder_pos_embed
    x = x + model.decoder_image_type_embedding
    key_padding_mask=torch.zeros(x.shape[0],x.shape[1],dtype=torch.bool, device=device)#useless
    # x = rearrange(x, 'b n c->n b c') 
    for key, values in dic2.items():
        for s in values:
            #texts = torch.Tensor([s]).to(device).long()
            texts = torch.Tensor([input_ids_[0].tolist()+s[1:]]).to(device).long()
            t = model.text_embedding_layer_SimVLM(texts)
            t = t + model.text_decoder_pos_embed[:, :t.shape[1], :]
            t = t + model.decoder_text_type_embedding
            target_text_length=len(texts[0])
            tgt_mask = generate_square_subsequent_mask(target_text_length).to(device)
            tgt_key_padding_mask=texts==0#shouldnt be used bause we do not use padding tokens
            # t = rearrange(t, 'b n c->n b c')
            output = model.decoder(t, x, is_causal=1, 
                                    tgt_mask=tgt_mask, memory_mask=None,
                                    tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=key_padding_mask)
            output = model.decoder_pred_text(output)
            # output = rearrange(output, 'n b c -> b c n')
            output = rearrange(output, 'b n c -> b c n')
            target_text = texts[:,1:]
            loss=F.cross_entropy(output[:,:,:-1],target_text)
            if loss.item() < loss_res:
                loss_res = loss.item()
                res = int(key)
    return res

def loglikelyhood(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    #model
    model = models_mae.__dict__[args.model](text_max_length=args.target_txt_len)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu') #why cpu?
    model.load_state_dict(checkpoint['model'])
    #tokinizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.dataset=="ImageNet":
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, args.partition), transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths_with_BOS(tokenizer, "labels/imagenet_labels.json")
    elif args.dataset=="CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths_with_BOS(tokenizer, "labels/cifar10_labels.json")
    elif args.dataset=="CIFAR100":
        dataset_train = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths_with_BOS(tokenizer, "labels/cifar100_labels.json")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        num_workers=10,
        pin_memory="store_true",
        drop_last=True,
    )
    proportions = {'1': 0, '2': 0}

    good=0
    count = 0
    text = ['this is a photo of a ']
    input_ids_ = input_ids_from_text(text, tokenizer)
    for i, (image,label) in enumerate(tqdm(data_loader_train)):
        res = predicted_label(model,input_ids_,image, dic2, device)

        if res ==label.item():
            good+=1
        count+=1
        if i%10==0:
            print(good/count)

def main_old(args):
    print("ONLY EVALUATING 1-token labels")
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    #model
    model = models_mae.__dict__[args.model](text_max_length=args.target_txt_len)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu') #why cpu?
    model.load_state_dict(checkpoint['model'])
    #tokinizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    if args.dataset=="ImageNet":
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, args.partition), transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths(tokenizer, "labels/imagenet_labels.json")
    elif args.dataset=="CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths(tokenizer, "labels/cifar10_labels.json")
    elif args.dataset=="CIFAR100":
        dataset_train = torchvision.datasets.CIFAR100(root='./data', train=True,download=True, transform=transform_test)
        sampler_train = torch.utils.data.RandomSampler(dataset_train, num_samples=args.nb_eval)
        lengths, dic2, mask_ = labels_lengths(tokenizer, "labels/cifar100_labels.json")
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=1,
        num_workers=10,
        pin_memory="store_true",
        drop_last=True,
    )
    proportions = {'1': 0, '2': 0}

    saved = 0
    one_tok =0
    two_toks =0

    void = 0
    two_only=0
    for (image,label) in tqdm(data_loader_train):
        #Other design possibilities: Append the mask token 1 by 1. Majority vote. 
        #compare text embedding/
        #text decoder and text encoder
        #still masking most of the patches
        #Image to text translation
        #forward only the unmasked tokens
        if (1 in lengths[str(label.item())]):
            image = image.squeeze(0).to(device)
            text = ['this is an image of a']
            texts_tokenized_input = tokenizer.batch_encode_plus(text,add_special_tokens=True,return_tensors='pt')
            input_ids_ = texts_tokenized_input['input_ids'].to(device)
            input_ids_ = input_ids_[0][:-1].unsqueeze(0)
            if not "autoregressive" in args.model:
                latent, mask, ids_restore, mask_text, ids_restore_text, length_img_tokens = model.forward_encoder(image.unsqueeze(0), input_ids_, 0)
            #latent, mask, ids_restore, mask_text, ids_restore_text, length_img_tokens = model.forward_encoder(image.unsqueeze(0), input_ids_, 0)
            #x, mask, ids_restore, length_img_tokens = model.forward_encoder(image.unsqueeze(0), 0)
            if 1 in lengths[str(label.item())]:
                proportions['1']+=1
                if not "autoregressive" in args.model:
                    _, pred_text = model.forward_decoder_captioning(latent, ids_restore, ids_restore_text, length_img_tokens, 1)
                    pred = torch.argmax(pred_text*mask_.view(30522).to(device), dim=2)[0][-1:].cpu().tolist()
                else:
                    logits = model.forward_next_word(image.unsqueeze(0), input_ids_, mask_ratio=0,target_text_length=len(input_ids_[0]), device=device)
                    pred = [torch.argmax(logits.squeeze(0)*mask_.view(30522).to(device)).cpu().tolist()]
                if pred in dic2[str(label.item())]:
                    one_tok+=1
        else:
            void +=1
    print(f"number of samples for which the corresponding label can be translated in 1 or 2 tokens: {args.nb_eval-void}")
    print(f"More precisely, {proportions['1']} can be tranlated in 1 token, and {proportions['2']} can be tranlated in 2 tokens")
    print(f"{100*(one_tok+saved+two_toks)/(args.nb_eval-void):.4f}% were correcly classified")
    print(f"{100*one_tok/proportions['1']:.4f}% were correcly classified with the one token mask")
    if two_only!=0: print(f"{100*two_toks/two_only:.4f}% among these that can only be translated with two tokens")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.type == "trie":
        print("prefix tree")
        main(args)
    elif args.type == "likelihood":
        print("likelihood")
        loglikelyhood(args)
    else:
        print("1-token")
        main_old("args")