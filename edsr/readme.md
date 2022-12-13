This code is mostly built on: [EDSR (Pytorch)](https://github.com/sanghyun-son/EDSR-PyTorch). We thank 🙏 the authors for sharing their code of EDSR.

This folder contains the following:
- EDSR architecture with MHCA.
- EDSR architecture with MMHCA (multiple input).
- Code to run the inference.

Demo:
```
python run.py --model EDSR --scale 2  --n_resblocks 16 \
     --n_feats 64 --res_scale 0.1 --shift_mean=False \
     --use_attention_resblock --use_mhca_3 --ratio=0.5 \
     --pre_train=pretrained_models/model_single_input_IXI_x2.pt
```

Check the ```run.py``` script for more Demo commands.

### 🔒 License
The MHCA/MMHCA code is released under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.
