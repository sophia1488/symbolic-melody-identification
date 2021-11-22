import numpy as np
import argparse
import torch
from miditoolkit.midi import parser as mid_parser

from scipy.sparse import csgraph
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import colorConverter
import matplotlib.patches as mpatches

from preprocess import split_window
from note import NOTE
from utils import midi_to_pianoroll
from model import CNN_Net
from predict import get_predicted_pianoroll, set_threshold, get_probability, build_graph, get_predicted_melody_ind


WIN_LEN = 64  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


################################################
#                PIANOROLL                     #
################################################
def make_note_dict():
    note_dict = {}
    note = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    note_dict[21] = 'A0'
    note_dict[23] = 'B0'
    base_number = 24
    for i in range(1, 9):
        for n in note:
            note_dict[base_number] = n + str(i)
            if n == 'E' or n == 'B':
                base_number += 1
            else:
                base_number += 2
    return note_dict


def notes_to_pianoroll(notes, predicted_melody_ind, length):
    predicted_melody_ind = set(predicted_melody_ind)
    melody_pr = np.zeros((128, length))
    accomp_pr = np.zeros((128, length))
    
    for i, note in enumerate(notes):
        s, e = int(note.start/120), int(note.end/120)       # 4 pixel per beat
        if s > length:
            break
        e = min(length, e)
        if i in predicted_melody_ind:
            melody_pr[note.pitch, s:e] = np.ones((1, e-s))
        else:
            accomp_pr[note.pitch, s:e] = np.ones((1, e-s)) 

    return melody_pr, accomp_pr


def plot_roll(pianoroll, pixels):
    # pianoroll[idx, msg['pitch'], note_on_start_time:note_on_end_time] = intensity   # idx => 0: melody, 1: non-melody
    
    # build and set figure object
    plt.ioff()
    fig = plt.figure(figsize=(17, 11))
    a1 = fig.add_subplot(111)
    a1.axis("equal")
    a1.set_facecolor("white")
    a1.set_axisbelow(True)
    a1.yaxis.grid(color='gray', linestyle='dashed')
    
    # set colors
    channel_nb = 2 
    transparent = colorConverter.to_rgba('white')
    colors = [mpl.colors.to_rgba('lightcoral'), mpl.colors.to_rgba('cornflowerblue')] 
    cmaps = [mpl.colors.LinearSegmentedColormap.from_list('my_cmap', [transparent, colors[i]], 128) for i in range(channel_nb)]

    # build color maps
    for i in range(channel_nb):
        cmaps[i]._init()
        # create your alpha array and fill the colormap with them
        alphas = np.linspace(0, 1, cmaps[i].N + 3)
        # create the _lut array, with rgba value
        cmaps[i]._lut[:, -1] = alphas

    label_name = ['melody', 'non-melody']
    for i in range(channel_nb):
        try:
            a1.imshow(pianoroll[i], origin="lower", interpolation="nearest", cmap=cmaps[i], aspect='auto', label=label_name[i])
        except IndexError:
            pass
    note_dict = make_note_dict()

    # set scale and limit of axis
    interval = 64
    plt.xticks([i*interval for i in range(13)], [i*4 for i in range(13)])
    plt.yticks([(24 + y*12) for y in range(8)], [note_dict[24 + y*12] for y in range(8)])
    plt.ylim([36, 96])      # C1 to C8

    # show legend, and create a patch (proxy artist) for every color 
    patches = [ mpatches.Patch(color=colors[i], label=label_name[i] ) for i in range(channel_nb) ]
    # put those patched as legend-handles into the legend
    first_legend = plt.legend(handles=[patches[0]], loc=2, fontsize=40)
    ax = plt.gca().add_artist(first_legend)
    plt.legend(handles=[patches[1]], loc=1, fontsize=40)
    
    # save pianoroll to figure
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("bars", fontsize=40)
    plt.ylabel("note name", fontsize=40)

    plt.savefig('pianoroll.jpg')
    return


################################################
#           FOR MODEL INFERENCE                #
################################################

def get_notelist(filename, mulinstr=True):
    # get notelist
    midi_obj = mid_parser.MidiFile(filename)
    
    # separate melody & accompaniment notelist
    melody = [NOTE(i.start, i.end, i.pitch, i.velocity, 0) for i in midi_obj.instruments[0].notes]
    acc = midi_obj.instruments[1].notes + midi_obj.instruments[2].notes
    acc = [NOTE(i.start, i.end, i.pitch, i.velocity, 1) for i in acc]
    notelist = melody+acc
    notelist = sorted(notelist, key=lambda n: n.start)
    return notelist


def cal_acc(predicted_melody_ind, notelist):
    predicted_melody_ind = set(predicted_melody_ind)
    tp, fp, fn, tn = 0, 0, 0, 0

    for i in range(len(notelist)):
        note = notelist[i]
        if i in predicted_melody_ind:
            # predicted as melody
            if note.Type == 0:
                tp += 1
            else:
                fp += 1
        else:
            # predicted as accompaniment
            if note.Type == 0:
                fn += 1
            else:
                tn += 1
    print(f'tp:{tp}, fp:{fp}, fn:{fn}, tn:{tn}')
    print(f'acc: {(tp+tn)/(tp+fp+fn+tn)}\n')
    return 


def get_melody_prediction(filename, model, cluster_method, min_th, mulinstr=True):
    notelist = get_notelist(filename, mulinstr=mulinstr)
    full_pianoroll, melody_pianoroll = midi_to_pianoroll(notelist)   # (128, length)
    width = full_pianoroll.shape[1]
    
    # split window
    split_pianoroll = split_window(full_pianoroll, WIN_LEN)
    split_pianoroll = np.array(split_pianoroll)
    split_pianoroll = np.expand_dims(split_pianoroll, axis=1)

    melody_pianoroll = split_window(melody_pianoroll, WIN_LEN)
    melody_pianoroll = np.array(melody_pianoroll)
    melody_pianoroll = np.expand_dims(melody_pianoroll, axis=1)

    # prediction & average probabilities in overlapping window
    pred_pianoroll = get_predicted_pianoroll(split_pianoroll, melody_pianoroll, model)
    pred_pianoroll = pred_pianoroll[:, : width]

    # each pop song has its own threshold by clustering algorithm
    threshold = set_threshold(pred_pianoroll, cluster_method, min_th)

    # build graph
    graph = build_graph(notelist, pred_pianoroll, full_pianoroll, threshold)
    dist_mat, predecessors = csgraph.shortest_path(graph, method='BF', directed=True, indices=[0], return_predecessors=True)
    predicted_melody_ind = get_predicted_melody_ind(predecessors)

    predicted_melody_ind = []
    for i in range(len(notelist)):
        prob = get_probability(notelist[i], pred_pianoroll, full_pianoroll, threshold)
        if prob > threshold:
            predicted_melody_ind.append(i)

    return threshold, predicted_melody_ind



def get_args():
    parser = argparse.ArgumentParser()

    ### path ###
    parser.add_argument('--split_file', type=str, default='pop909_datasplit.pkl')
    parser.add_argument('--root', type=str, default='../Dataset/pop909_aligned', help='path to pop909 dataset')
    parser.add_argument('--ckpt', type=str, required=True)
    
    ### parameter ###
    parser.add_argument('--cluster', type=str, default="centroid")
    parser.add_argument('--min_threshold', type=float, default=1e-15)
    parser.add_argument('--monophonic', type=bool, default=True)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    print(f'device: {DEVICE}\n')

    # load model
    best_mdl = args.ckpt
    print(f"Loading model from {best_mdl.split('/')[-2]}")
    model = CNN_Net()
    model.load_state_dict(torch.load(best_mdl, map_location='cpu'))
    model = model.to(DEVICE)
    model.eval()
    
    # load test set 
    print('Loading test file 343.mid')
    filename = f'{args.root}/test/343.mid'
   
    notelist = get_notelist(filename)
    th, predicted_ind = get_melody_prediction(filename, model, args.cluster, args.min_threshold)
    cal_acc(predicted_ind, notelist)

    # plot pianoroll
    bars = 48
    pixels = 48*4*4     # 4 beat per bar, 4 pixel per beat
    melody_pr, accomp_pr = notes_to_pianoroll(notelist, predicted_ind, pixels)
    print(melody_pr.shape, accomp_pr.shape, pixels)
    pianoroll = [melody_pr, accomp_pr]
    plot_roll(pianoroll, pixels)


if __name__ == '__main__':
    main()
