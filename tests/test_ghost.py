# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functorch

def test_functorch(model_functorch, imgs_, input_ids_, args, device, reduction):
    #(pred_img, mask, target_img), (pred_text, mask_text, target_text) = func_model(weights, imgs_.unsqueeze(0), input_ids_.unsqueeze(0), args.mask_ratio)
    func_model, weights, buffers = functorch.make_functional_with_buffers(model_functorch)
    def compute_loss_stateless_model(weights,buffers, imgs_, input_ids_):
        return func_model(weights, buffers, imgs_.unsqueeze(0), input_ids_.unsqueeze(0), mask_ratio=args.mask_ratio, prob_parallel_decoding=0, target_text_length=args.target_txt_len, device=device, reduction=reduction)
    compute_grad_and_loss = lambda weights, buffers, img, txt: functorch.vmap(functorch.grad_and_value(compute_loss_stateless_model), (None, None, 0, 0), randomness='different')(weights,buffers, img, txt)
    grads, loss = compute_grad_and_loss(weights, buffers, imgs_, input_ids_)
    return grads, loss