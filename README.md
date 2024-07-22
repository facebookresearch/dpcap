# DP-Cap - Differentially Private Representation Learning via Image Captioning

Codebase for the ICML 2024 paper "Differentially Private Representation Learning via Image Captioning" (https://arxiv.org/abs/2403.02506)

TL;DR: A Differetentially Private Image Captionner trained on 233M image-text pairs, with $\varepsilon=8$ and better image representations than regular MAE trained on the same dataset.

Tom Sander* (Ecole polytechnique, Meta FAIR), Yaodong Yu*<sup>*</sup>* (UC Berkeley, Meta FAIR), Maziar Sanjabi (Meta), Alain Durmus (Ecole polytechnique), Yi Ma (UC Berkeley),  Kamalika Chaudhuri (Meta FAIR), Chuan Guo (Meta FAIR).

![Performance overview](images/DP-MULTIMODAL.png "Overview of DP-CAP")

## Set-up

1. Clone this respository and go to the `dpcap` folder, then clone and patch required submodules

```
git clone https://github.com/facebookresearch/dpcap.git
cd dpcap
git submodule update --init --recursive

cd private-transformers
git checkout 18ccc4eab7355e4ac96051a82434796f6aa4624b
git apply ../patch/private-transformers.patch

cd ../msn
git apply ../patch/msn.patch

cd ..
```


2. Install requirements

```
conda create -n dpcap python==3.8
conda activate dpcap
pip install -r requirements.txt

cd private-transformers
pip install -e
```

## How to load a DP-Cap Model?


**WARNING:** We are releasing the weights of models trained on the same dataset as <a href="https://github.com/facebookresearch/MetaCLIP">Meta-CLIP</a>, which differs from the DEDUP-LAION-233M dataset used in our paper. 
Notably, this dataset contains 2 billion samples.
Currently, we are providing the weights for a DP-Cap base model with a privacy budget of $\varepsilon=7$.
After downloading using this <a href="https://dl.fbaipublicfiles.com/dpcap/base-eps7.pth"> link</a>, please place the weights in the ./checkpoints/DP-Cap/ directory.

***Performance***

The vision encoder's performance is approximately equivalent to the model trained with $\varepsilon=8$ in the paper:

| Performance on ImageNet | Full Linear Prob | 10 shot | 5 shot | 1-shot |
|-------------------------|------------------|---------|--------|--------|
| Percentage              | 31.9%            | 24.8%   | 16.3%  | 10.7%  |

 
We have trained this model for 5000 steps, using $\sigma=0.527$ and the same 1.3M batch size, without parameter tuning (yet!). 

To load the base $\varepsilon=7$ model, you can do:

```python
import models_mae
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda")

model = models_mae.__dict__["mae_vit_base_patch16_autoregressive_nobias"](text_max_length=77) #40 can be changed by anything; you can aldo replace "base" by "large", "small" or "tiny"
model = model.cuda()
checkpoint = torch.load("./checkpoints/DP-Cap/base-eps7.pth", map_location='cpu')
msg = model.load_state_dict(checkpoint['model'], strict=False)
print(msg)
```

## How to get an image representation from the encoder?

```python
image = Image.open("./images/00.png")
image_preprocess = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.Lambda(lambda x: x.convert("RGB")),  # Convert RGBA to RGB
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
image_tensor = image_preprocess(image).cuda().unsqueeze(0)

##Representation (197*768)
x = model.linear_prob_forward(image_tensor)
```

## How to perform 0 shot?

For zero-shot learning, we calculate the loss for each label and select the one with the smallest loss. Here's an example of how this process works when there are only five labels:

```python
##Caption
from captioning import predicted_label, labels_lengths_with_BOS, input_ids_from_text

labels = {0:"zebra", 1:"dog", 2:"cat", 3:"building", 4:"city"}

text = ['this is a photo of a']
input_ids_ = input_ids_from_text(text, tokenizer)
_, dic, _ = labels_lengths_with_BOS(tokenizer,None, dic=labels)

key = predicted_label(model, input_ids_, image_tensor, dic, device)
print(labels[key])
```

As outlined in the paper, the computational expense increases significantly with the number of labels. We propose a novel method to address this issue by implementing a tree search, as detailed in the `captioning.py` file.

## How to generate a caption?

Since DP-Cap was trained with captioning, it's possible to directly prompt the model to generate a caption from a given image.

```python
##Caption
caption = model.forward_caption(image_tensor)
print(tokenizer.decode(caption))
```

## Pre-training DP-Cap

To pre-train DP-Cap  with **multi-node distributed training** on your dataset of image-caption pairs, run the following on 16 nodes with 8 GPUs each:

```
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --folder ${EXP_NAME}\
    --dataset_path ${DIR_PATH}\
    --nb_samples ${NB_SAMPLES}\
    --nodes 16 \
    --ngpus 8 \
    --use_volta32 \
    --num_workers 10 \
    --batch_size 1310720\
    --TAN_batch_size 1310720\
    --max_physical_B 128 \
    --model mae_vit_base_patch16_autoregressive_nobias \
    --resume "./checkpoints/DP-Cap/base-init.pth" \
    --init True\
    --DP "ghost" \
    --amp True \
    --target_txt_len 40 \
    --mask_ratio 0 \
    --blr 1.0e-07 \
    --partition ${PARTITION_NAME} \
    --weight_decay 0.005 \
    --sigma 0.728 \
    --max_grad_norm 1 \
    --warmup_iterations 2000 \
    --overall_iterations 10000 \
    --target_step 5700\
    --save_freq 1000
```

- Here the batch size is 128 per GPU. If you want to simulate training at lower computational cost, you can decrease the TAN_batch_size (then the $\sigma$ that will be used will automatically be divided by batch_size/TAN_batch_size. See (Sander et al., 2023) for more details about this simulation). If memory or # gpus is limited, use `--max_device_batch_size` to reduce the batch size in each (accumulation) gradient step on each gpu.
- Specify ``DIR_PATH`` variable with the path to your dataset.
- Specify ``JOB_DIR`` variable to define the directory for saving logs and checkpoints.
- Specify ``NB_SAMPLES`` the number of samples in your dataset (important for logging)
- Specify ``EXP_NAME`` variable to define the name of the experiment.
- Specify ``PARTITION_NAME`` variable with the slurm partition name.

Use the `--init` option set to True if you wish to begin training from scratch, using the weights provided in the `--resume` checkpoint. If the `--init` option is not specified, the system will search for additional elements in the checkpoint (such as optimizer state, learning rate, etc.) and resume training from where it last stopped.

## Evaluating DP-Cap

After applying the msn patch as advised above, use the template provided in the run_linear_prob.sh file to perform few-shot evaluation.


## Code Acknowledgements

The majority of DP-Cap is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [private-transformers](https://github.com/lxuechen/private-transformers) is licensed under the Apache 2.0 license; [vision-language-models-are-bows](https://github.com/mertyg/vision-language-models-are-bows) is licensed under the MIT license. Note that due to the non-commercial nature of the CC-BY-NC license, this code is **not** ready for production use.

## How to Cite Us

If you use DP-Cap in your research or wish to refer to the baseline results published in the repository, please use the following BibTeX entry.

```bibtex
@misc{sander2024differentiallyprivaterepresentationlearning,
      title={Differentially Private Representation Learning via Image Captioning}, 
      author={Tom Sander and Yaodong Yu and Maziar Sanjabi and Alain Durmus and Yi Ma and Kamalika Chaudhuri and Chuan Guo},
      year={2024},
      eprint={2403.02506},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.02506}, 
}
```

