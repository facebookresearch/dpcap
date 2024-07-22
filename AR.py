# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

root_dir="Path to your VGA and VGR and Flickr"
root_dir_coco="Path to your COCO dataset"
import sys


import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.PILToTensor()
])

from PIL import Image
image_preprocess = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

import models_mae
from transformers import BertTokenizer
import tqdm

import torch
device = torch.device('cuda')
#model

#tokinizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#data


def flickr(model):
    total=0
    good = 0
    for sample in flickr_order_dataset:
        total+=1
        if total>len(flickr_order_dataset):
            break
        #print(sample)
        image = image_preprocess(sample['image_options'][0])
        text1 = sample['caption_options'][0]
        text2 = sample['caption_options'][1]
        text3 = sample['caption_options'][2]
        text4 = sample['caption_options'][3]
        text5 = sample['caption_options'][4]
        #print(image)
        # print(text1)
        # print(text2)
        texts_tokenized_input_1 = tokenizer.batch_encode_plus(
            [text1],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_2 = tokenizer.batch_encode_plus(
            [text2],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_3 = tokenizer.batch_encode_plus(
            [text3],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_4 = tokenizer.batch_encode_plus(
            [text4],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_5 = tokenizer.batch_encode_plus(
            [text5],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids_1 = texts_tokenized_input_1['input_ids']
        input_ids_2 = texts_tokenized_input_2['input_ids']
        input_ids_3 = texts_tokenized_input_3['input_ids']
        input_ids_4= texts_tokenized_input_4['input_ids']
        input_ids_5 = texts_tokenized_input_5['input_ids']

        # print(input_ids_1)
        # print(input_ids_2)
        imgs_ = image.to(device, non_blocking=True)
        input_ids_1 = input_ids_1.to(device, non_blocking=True)
        input_ids_2 = input_ids_2.to(device, non_blocking=True)
        input_ids_3 = input_ids_3.to(device, non_blocking=True)
        input_ids_4 = input_ids_4.to(device, non_blocking=True)
        input_ids_5 = input_ids_5.to(device, non_blocking=True)
        loss1 = model.forward(imgs_.unsqueeze(0), input_ids_1, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss2 = model.forward(imgs_.unsqueeze(0), input_ids_2, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss3 = model.forward(imgs_.unsqueeze(0), input_ids_3, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss4 = model.forward(imgs_.unsqueeze(0), input_ids_4, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss5 = model.forward(imgs_.unsqueeze(0), input_ids_5, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        if loss1<min(loss2, loss3, loss4, loss5):
            good+=1
        if total%10==0:
            print(f"---------ITERATION {total}----------------")
            print(good/total)
    return(good/total)


def coco(model):
    total=0
    good = 0
    for sample in coco_order_dataset:
        total+=1
        if total>len(coco_order_dataset):
            break
        #print(sample)
        image = image_preprocess(sample['image_options'][0])
        text1 = sample['caption_options'][0]
        text2 = sample['caption_options'][1]
        text3 = sample['caption_options'][2]
        text4 = sample['caption_options'][3]
        text5 = sample['caption_options'][4]
        #print(image)
        # print(text1)
        # print(text2)
        texts_tokenized_input_1 = tokenizer.batch_encode_plus(
            [text1],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_2 = tokenizer.batch_encode_plus(
            [text2],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_3 = tokenizer.batch_encode_plus(
            [text3],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_4 = tokenizer.batch_encode_plus(
            [text4],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_5 = tokenizer.batch_encode_plus(
            [text5],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids_1 = texts_tokenized_input_1['input_ids']
        input_ids_2 = texts_tokenized_input_2['input_ids']
        input_ids_3 = texts_tokenized_input_3['input_ids']
        input_ids_4= texts_tokenized_input_4['input_ids']
        input_ids_5 = texts_tokenized_input_5['input_ids']

        # print(input_ids_1)
        # print(input_ids_2)
        imgs_ = image.to(device, non_blocking=True)
        input_ids_1 = input_ids_1.to(device, non_blocking=True)
        input_ids_2 = input_ids_2.to(device, non_blocking=True)
        input_ids_3 = input_ids_3.to(device, non_blocking=True)
        input_ids_4 = input_ids_4.to(device, non_blocking=True)
        input_ids_5 = input_ids_5.to(device, non_blocking=True)
        loss1 = model.forward(imgs_.unsqueeze(0), input_ids_1, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss2 = model.forward(imgs_.unsqueeze(0), input_ids_2, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss3 = model.forward(imgs_.unsqueeze(0), input_ids_3, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss4 = model.forward(imgs_.unsqueeze(0), input_ids_4, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss5 = model.forward(imgs_.unsqueeze(0), input_ids_5, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        if loss1<min(loss2, loss3, loss4, loss5):
            good+=1
        if total%10==0:
            print(f"---------ITERATION {total}----------------")
            print(good/total)
    return(good/total)


def vga(model):
    total=0
    good = 0
    for sample in vga_dataset:
        total+=1
        if total>len(vga_dataset):
            break
        #print(sample)
        image = image_preprocess(sample['image_options'][0])
        text1 = sample['caption_options'][0]
        text2 = sample['caption_options'][1]
        texts_tokenized_input_1 = tokenizer.batch_encode_plus(
            [text1],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_2 = tokenizer.batch_encode_plus(
            [text2],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )


        input_ids_1 = texts_tokenized_input_1['input_ids']
        input_ids_2 = texts_tokenized_input_2['input_ids']

        # print(input_ids_1)
        # print(input_ids_2)
        imgs_ = image.to(device, non_blocking=True)
        input_ids_1 = input_ids_1.to(device, non_blocking=True)
        input_ids_2 = input_ids_2.to(device, non_blocking=True)

        loss1 = model.forward(imgs_.unsqueeze(0), input_ids_1, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss2 = model.forward(imgs_.unsqueeze(0), input_ids_2, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)

        if loss1>loss2:
            good+=1
        if total%10==0:
            print(f"---------ITERATION {total}----------------")
            print(good/total)
    return(good/total)

def vgr(model):
    total=0
    good = 0
    for sample in vgr_dataset:
        total+=1
        if total>len(vgr_dataset):
            break
        #print(sample)
        image = image_preprocess(sample['image_options'][0])
        text1 = sample['caption_options'][0]
        text2 = sample['caption_options'][1]
        texts_tokenized_input_1 = tokenizer.batch_encode_plus(
            [text1],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        texts_tokenized_input_2 = tokenizer.batch_encode_plus(
            [text2],
            add_special_tokens=True,
            max_length=40, #We can do shorter. see https://pypi.org/project/open-clip-torch/#files 
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )


        input_ids_1 = texts_tokenized_input_1['input_ids']
        input_ids_2 = texts_tokenized_input_2['input_ids']

        # print(input_ids_1)
        # print(input_ids_2)
        imgs_ = image.to(device, non_blocking=True)
        input_ids_1 = input_ids_1.to(device, non_blocking=True)
        input_ids_2 = input_ids_2.to(device, non_blocking=True)

        loss1 = model.forward(imgs_.unsqueeze(0), input_ids_1, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)
        loss2 = model.forward(imgs_.unsqueeze(0), input_ids_2, mask_ratio=0, prob_parallel_decoding=0, target_text_length=40, device=device)

        if loss1>loss2:
            good+=1
        if total%10==0:
            print(f"---------ITERATION {total}----------------")
            print(good/total)
    return(good/total)

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('AROeval', add_help=False)
    parser.add_argument('--checkpoints',nargs="+", type=str,
                        help='checkpoints to evaluate') 
    parser.add_argument('--index', default=0, type=int,
                        help='for name of dic') 
    parser.add_argument('--flickr_only', default=0, type=int,
                        help='Only evaluate flickr') 
    return parser

def main(args):
    checkpoints=args.checkpoints
    index = args.index
    import numpy as np
    hello = {'hello':'world'}
    np.save('output_dir/hello.npy', hello) 
    np.save('hello.npy', hello) 
    res = {}
    for checkpoint_path in checkpoints:
        if ("tiny" in checkpoint_path) or ("ViTTiny" in checkpoint_path):
            name = "mae_vit_tiny_patch16_autoregressive_nobias"
        elif "small" in checkpoint_path:
            name = "mae_vit_small_patch16_autoregressive_nobias"
        elif "large" in checkpoint_path:
            name = "mae_vit_large_patch16_autoregressive_nobias"
        else:
            name = "mae_vit_base_patch16_autoregressive_nobias"
        model = models_mae.__dict__[name](text_max_length=40)
        model.to(device)
        model.eval()
        checkpoint = torch.load(checkpoint_path, map_location='cpu') #why cpu?
        model.load_state_dict(checkpoint['model'])
        model.to(device)
    if not args.flickr_only:
        res[checkpoint_path+"_coco"] = coco(model)
        res[checkpoint_path+"_vga"] = vga(model)
        res[checkpoint_path+"_vgr"] = vgr(model)
    res[checkpoint_path+"_flickr"] = flickr(model)
    print(res)
    np.save(f'output_dir/res_{index}_f{args.flickr_only}.npy', res)

if __name__ == '__main__':
    args = get_args_parser()
    print(args)
    args = args.parse_args()
    print("args.checkpoints",args.checkpoints)
    sys.path.insert(0, './vision-language-models-are-bows_submodule')
    from dataset_zoo import VG_Relation, VG_Attribution, COCO_Order
    import torch
    from dataset_zoo import COCO_Order, Flickr30k_Order
    if not args.flickr_only:
        vga_dataset = VG_Attribution(image_preprocess=None, download=True, root_dir=root_dir)
        vgr_dataset = VG_Relation(image_preprocess=None, download=True, root_dir=root_dir)
        coco_order_dataset = COCO_Order(image_preprocess=None, download=True, root_dir=root_dir_coco) 
        print("len of VGA dataset", len(vga_dataset))
        print("len of VGR dataset", len(vgr_dataset))
        print("len of COCO_Order", len(coco_order_dataset))
    flickr_order_dataset = Flickr30k_Order(image_preprocess=None, root_dir=root_dir, split="test")
    sys.path=sys.path[1:]
    print("len of flckr", len(flickr_order_dataset))
    main(args)