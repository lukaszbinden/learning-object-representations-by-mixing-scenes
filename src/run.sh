#!/bin/bash

# nohup python main.py -c="Exp13@node02: Same as exp13; enc s4: df_dim * 8 (3); dec h3: gf_dim*4(2); image size 64; small dataset v2;b=" > nohup_exp13.out &
# nohup python main.py -c="Exp14@node03: Same as exp13 but without linear layer at end of encoder but instead a conv layer " > nohup_exp14.out & 
# nohup python main.py -c="Exp15@node04: Same as exp13; enc s4: df_dim * 8 (3); dec h3: gf_dim*4(2); image size 64; small dataset v2;b=32" > nohup_exp15.out &
# nohup python -u main.py -c="Exp18@node04: same as exp16 but with 500 epochs " > nohup_exp18.out &
# nohup python -u main_dcgan_coco.py -c="Exp20@node04: plain DCGAN training with denseblocks in both encoder and decoder" > nohup_exp20.out &
# nohup python -u main_dcgan_coco.py -c="Exp21@node02: as exp20 but without dense-blocks; enc/dec as before but incr cap in enc, decr cap in dec (i.e. balanced cap in enc/dec)" > nohup_exp21.out &
# nohup python -u main_dcgan_coco.py -c="Exp22@node04: same as exp20; plus some issues fixed; uses testset v4 for cherrypicking" > nohup_exp22.out &
# nohup python -u main_dcgan_coco.py -c="Exp23@node03: same as exp22 but with weight=5 (instead of 20 in g_loss_comp)  " > nohup_exp23.out &
# nohup python -u main_dcgan_coco.py -c="Exp24@node02: Same as exp22; additionally: adds PSNR, growth_rate = 24 (from 16)" > nohup_exp24.out &
# nohup python -u main_dcgan_coco.py -c="Exp25@node03: Same as exp23; additionally: adds PSNR, growth_rate = 24 (from 16)" > nohup_exp25.out &
# nohup python -u main_dcgan_coco.py -c="Exp26@node03: Same as exp25 but with weight=40 (instead of orig 20 in g_loss_comp or 5 in exp25)" > nohup_exp26.out &
#nohup python -u main_dcgan_coco.py -c="Exp27@node02: Same as exp24 but adds self-attention layer in discriminator (i.e. make dsc stronger)" > nohup_exp27.out &
# nohup python -u main_dcgan_coco.py -c="Exp28@hp: Same as exp24 but changed learning rates: discriminator=0.0004, generator=0.0001" 
# nohup python -u main_dcgan_coco.py -c="Exp30@node02: Rerun exp21 for comparison; also using PSNR; based on exp27 (w/ SA layer in DSC); use weight 40 (from exp26)" > nohup_exp30.out &
# nohup python -u main_dcgan_coco.py -c="Exp31@node03: builds on exp27; uses LR decay" > nohup_exp31.out &
# nohup python -u main_dcgan_coco.py -c="Exp33@node02: same as exp27, but with SN in decoder" > nohup_exp33.out &
# nohup python -u main_dcgan_coco.py -c="Exp35@node04: based on exp33; use global local discriminator" > nohup_exp35.out &
# nohup python -u main_dcgan_coco.py -c="Exp36@node03: same as exp33; weight 0.998 for L1-loss, 0.002 for g_loss" > nohup_exp36.out & 
# nohup python -u main_dcgan_coco.py -c="Exp38@node02: based on exp36: remove dropout from AE; TB log V(D,G) and 2 logits of DSC;increase feature size=1024" > nohup_exp38.out &
nohup python -u main_dcgan_coco.py -c="Exp40@node05: same as exp39;use resize-convolution (in decoder);log weights for TB" > nohup_exp40.out &


