cd ..
python3 train_finetuning.py --gpu 0 --exp sam_ft --label_num 16 --patch_size 112 --rdmrotflip
python3 train_retraining.py --gpu 0 --exp 3d_seg --label_num 16 --patch_size 112 --pre_exp sam_ft
# test
python3 test_singleclass.py --gpu 0 --model 3d_seg --iteration 6000




