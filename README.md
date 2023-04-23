# AgileGAN
Unofficial implementation for paper "AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning"

The code is mainly based on these two repo: [AgileGAN-inference](https://github.com/flyingbread-elon/AgileGAN) and [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel)

## Installation

Please refer to [AgileGAN#requirements](https://github.com/flyingbread-elon/AgileGAN#requirements) and [pixel2style2pixel#prerequisites](https://github.com/eladrich/pixel2style2pixel#prerequisites).


## Train

1. Prepare the dataset and pretrained models.

2. Start training using

```
python train.py \
    --exp_dir=ffhq_encode \
    --workers=8 \
    --batch_size=4 \
    --test_batch_size=4 \
    --test_workers=8 \
    --val_interval=2500 \
    --save_interval=5000 \
    --start_from_latent_avg
```


## Test

Please refer to [AgileGAN-inference](https://github.com/flyingbread-elon/AgileGAN)

`
python test.py --path examples/29899.jpg --style cartoon
`

