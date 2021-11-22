import pickle
import numpy as np
from miditoolkit.midi import parser as mid_parser
from utils import midi_to_pianoroll
from note import NOTE
import random


WIN_LEN = 64 
WIN_HEIGHT = 128


def load_split(filename):
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return x['train_data'], x['valid_data'], x['test_data']


def overlap_split(score, WIN_WIDTH):
    '''
    PARAMETER:
        score: (win_height, score_len)
        WIN_WIDTH: default 64
    RETURN:
        [ numpy array of shape [win_height, win_width] * N ]
    '''
    if WIN_WIDTH % 2: return None
    HOP = WIN_WIDTH // 2
    ret = []
    score_len = score.shape[1] + HOP

    # pad number of HOP at the left
    score = np.pad(score, ((0,0), (HOP,0)), 'constant', constant_values=0)

    centers = [i for i in range(HOP, score_len+HOP, HOP)]

    # pad at the right 
    end = centers[-1] + HOP
    pad_len = end - score_len
    score = np.pad(score, ((0,0), (0, pad_len)), 'constant', constant_values=0)

    for center in centers:
        ret.append( score[:, center-HOP : center+HOP] )
    return ret


def split_window(score, WIN_WIDTH):
    '''
    PARAMETER:
        score: (win_height, score_len)
        WIN_WIDTH: default 64
    RETURN:
        [ win_height, np.array(WIN_WIDTH) ] * num_of_windows
    '''
    if WIN_WIDTH > score.shape[1]:
        # pad right
        pad_len = WIN_WIDTH - score.shape[1]
        return [ np.pad(score, ((0,0), (0,pad_len)), 'constant', constant_values=0) ]
    else:
        # split (overlap 50%), and pad at 1st / last few frames
        return overlap_split(score, WIN_WIDTH)


def read_file(filename, score_list, melody_list):
    # get notelist
    midi_obj = mid_parser.MidiFile(filename)
    
    # separate melody & accompaniment notelist: set melody = 0, accompaniment = 1
    melody = [NOTE(i.start, i.end, i.pitch, i.velocity, 0) for i in midi_obj.instruments[0].notes]
    acc = midi_obj.instruments[1].notes + midi_obj.instruments[2].notes
    acc = [NOTE(i.start, i.end, i.pitch, i.velocity, 1) for i in acc]
    notelist = melody+acc
    notelist = sorted(notelist, key=lambda n: n.start)
    
    score_pr, melody_pr = midi_to_pianoroll(notelist)   # (win_height, 2754)
    
    # split window 
    melody_pr_splitted = split_window(melody_pr, WIN_LEN)
    score_pr_splitted = split_window(score_pr, WIN_LEN)

    melody_list += melody_pr_splitted
    score_list += score_pr_splitted
    return


def preprocess_split(root, split, train):
    score_list, melody_list = [], []
    subset = 'train' if train else 'valid'

    for subfile in split:
        read_file(f'{root}/{subset}/{subfile}', score_list, melody_list)

    score_list = np.array(score_list)
    score_list = np.expand_dims(score_list, axis=1)
    melody_list = np.array(melody_list)
    melody_list = np.expand_dims(melody_list, axis=1)

    return score_list, melody_list


def preprocess(split_file, root):
    train_data, valid_data, _ = load_split(split_file)

    # get train & valid data
    train_score, train_melody = preprocess_split(root, train_data, train=True)
    valid_score, valid_melody = preprocess_split(root, valid_data, train=False)

    return train_score, train_melody, valid_score, valid_melody


def data_augmentation(X, y):
    ''' 
    Perform data augmentation in 50% of window
    PARAMETERS:
        X: numpy.ndarray with shape (num_of_windows, 1, win_height, win_len), this contains the input pianoroll window (melody + accompaniment)
        y: numpy.ndarray with shape (num_of_windows, 1, win_height, win_len), this contains the output pianoroll window (melody)

    RETURN:
        numpy.ndarray with shape (int(num_of_windows / 2), 1, win_height, win_len)
    '''
    indices = X.shape[0]
    extracted_indices = np.random.choice(range(indices), int(indices/2), replace=False)
    X_add, y_add = [], []

    for i in extracted_indices:
        score = X[i].copy()
        melody = y[i].copy()

        idx = np.argwhere(melody > 0.5)
        for j in idx:
            score[j[0], j[1], j[2]] = 0
            melody[j[0], j[1], j[2]] = 0

            # shift melody 1 or 2 octave lower or 1 octave higher
            k = random.choice([-1, -2, 1])
            new_pitch = j[1] - k*12 if 0 <= j[1]-k*12 < WIN_HEIGHT else j[1]
            score[j[0], new_pitch, j[2]] = 1
            melody[j[0], new_pitch, j[2]] = 1

        X_add.append(score)
        y_add.append(melody)

    return np.array(X_add), np.array(y_add)
