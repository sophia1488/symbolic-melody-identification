name="2021"     # specify the name for checkpoint path

# train
python3 main.py --name=$name

# evaluation on test set
python3 predict.py --ckpt=result/$name/CNN-melody-identification.pth

# plot pianoroll of 343.mid
python3 pianoroll.py --ckpt=result/$name/CNN-melody-identification.pth
