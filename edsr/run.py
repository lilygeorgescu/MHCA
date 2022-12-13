"""
This is the script that runs the inference.

- Scale 2x:
-- Single input
python run.py --model EDSR --scale 2  --n_resblocks 16 --n_feats 64 --res_scale 0.1   --shift_mean=False  --use_attention_resblock --use_mhca_3 --ratio=0.5  --pre_train=pretrained_models/model_single_input_IXI_x2.pt
-- Multi input
python run.py --model EDSR_Nav --scale 2  --n_resblocks 16 --n_feats 64 --res_scale 0.1   --shift_mean=False --use_nav --use_mhca_3 --ratio=0.5  --pre_train=pretrained_models/model_multi_input_IXI_x2.pt

- Scale 4x:
-- Single input
python run.py --model EDSR --scale 4  --n_resblocks 16 --n_feats 64 --res_scale 0.1   --shift_mean=False  --use_attention_resblock --use_mhca_2 --ratio=0.5  --pre_train=pretrained_models/model_single_input_IXI_x4.pt
-- Multi input
python run.py --model EDSR_Nav --scale 4  --n_resblocks 16 --n_feats 64 --res_scale 0.1   --shift_mean=False --use_nav --use_mhca_2 --ratio=0.5  --pre_train=pretrained_models/model_multi_input_IXI_x4.pt

"""

import matplotlib.pyplot as plt
import model as model_def
import numpy as np
import pickle
import torch
from options import args as args_env


def prepare(*args):
    device = torch.device('cpu' if args_env.cpu else 'cuda')

    def _prepare(numpy_array):
        tensor = torch.Tensor(numpy_array)
        if args_env.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    if type(args[0]) == list:
        return [_prepare(a) for a in args[0]]
    else:
        return _prepare(args[0])

def inference(lr):
    # Create the EDSR model.
    model = model_def.Model(args_env)

    torch.set_grad_enabled(False)

    lr = prepare(lr)

    sr = model(lr, 0)

    # Visualisation.
    hr_image = sr[0, 0].cpu().numpy()

    if type(lr) == list:
        input_lr_image = lr[0][0, 0].cpu().numpy()
    else:
        input_lr_image = lr[0, 0].cpu().numpy()


    fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, int(args_env.scale[0])]})

    axs[0].imshow(input_lr_image, cmap='gray')
    axs[0].set_title('LR')

    axs[1].imshow(hr_image, cmap='gray')
    axs[1].set_title('HR')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    with open(f'test_samples\HR_T2_500_0_x{args_env.scale[0]}.pt', 'rb') as _f:
        t2w_lr = np.expand_dims(pickle.load(_f), axis=(0, 1))
    with open(f'test_samples\HR_PD_500_0_x{args_env.scale[0]}.pt', 'rb') as _f:
        pd_lr = np.expand_dims(pickle.load(_f),  axis=(0, 1))

    # The model for IXI was trained using the T2w as the first image and PD as the second image.
    if args_env.use_nav:
        lr_input = [t2w_lr, pd_lr]
    else:
        lr_input = t2w_lr

    inference(lr_input)
