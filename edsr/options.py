"""
Forked from https://github.com/sanghyun-son/EDSR-PyTorch.
"""

import argparse

parser = argparse.ArgumentParser(description='EDSR with MHCA')


# Hardware specifications
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')

# Model specifications
parser.add_argument('--model', default='EDSR',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='',
                    help='pre-trained model directory')

parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--scale', type=str, default='4',
                    help='super resolution scale')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB; It is not used for medical image.')

# MHCA specifications
parser.add_argument('--ratio', type=str, default='4',
                    help='Channel reduction ratio.')

parser.add_argument('--use_nav', action='store_true',
                    help='Use more inputs for the network.')

parser.add_argument('--use_attention_resblock', action='store_true',
                    help='Use MHCA in the resnetblocks.')

parser.add_argument('--use_mhca_2', action='store_true',
                    help='Use MHCA with 2 heads.')

parser.add_argument('--use_mhca_3', action='store_true',
                    help='Use MHCA with 3 heads.')


args = parser.parse_args()
args.scale = list(map(lambda x: int(x), args.scale.split('+')))


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

