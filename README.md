# Symbolic Melody Identification
This repository is an unofficial PyTorch implementation of the paper:[A Convolutional Approach to Melody Line Identification in Symbolic Scores](https://arxiv.org/abs/1906.10547).

Its official version can be found [here](https://github.com/LIMUNIMI/Symbolic-Melody-Identification), which is written in Theano and Lasagne.

This repo is trained and evaluated on POP909 Dataset (in 4/4 time signature), to serve as a baseline of [MIDI-BERT](https://github.com/wazenmai/MIDI-BERT).  

## Evaluation
`python3 predict.py --ckpt=result/$name/CNN-melody-identification.pth`

The result (`result.txt`) will be saved, you may analyze its accuracy, precision, recall, f1-score later.

I've already provided the checkpoint at `result/1118/CNN-melody-identification.pth` and its result at `result/1118/result.txt`.

### Plot the predicted pianoroll from test set 343.mid
`python3 pianoroll.py --ckpt=result/$name/CNN-melody-identification.pth`

<img src="https://user-images.githubusercontent.com/47291963/142834993-fe1f1d12-d6a7-41c8-9946-f5ce4e4596c4.jpg" alt="pianoroll" width=400 height=240>

## Train
`python3 main.py --name=$name`
