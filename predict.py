import numpy as np
import argparse
import torch
from miditoolkit.midi import parser as mid_parser

from sklearn.cluster import KMeans
import fastcluster
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.sparse import csgraph

from preprocess import load_split, split_window
from note import NOTE
from utils import iqr, midi_to_pianoroll
from model import CNN_Net


WIN_LEN = 64 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def recreate_pianoroll(windows, overlap=True):
    '''
    Recreate pianoroll from windows of shape (NUM_WIN, 1, WIN_HEIGHT, WIN_WIDTH)
    RETURN:
        2D array with shape (128, pianoroll length)
    '''
    windows = windows.detach().cpu().numpy()
    WIN_WIDTH = windows.shape[3]
    WIN_HEIGHT = windows.shape[2]
    NUM_WIN = windows.shape[0]
    pianoroll_width = WIN_WIDTH * NUM_WIN
    if overlap:
        pianoroll_width /= 2
        pianoroll_width += WIN_WIDTH / 2
    pianoroll_width = int(pianoroll_width)
    output = np.zeros((WIN_HEIGHT, pianoroll_width))

    for i in range(NUM_WIN):
        window = windows[i][0]
        if overlap:
            output[:, i*WIN_WIDTH//2 : (i+2)*WIN_WIDTH//2] += window/2
        else:
            output[:, i*WIN_WIDTH: (i+1)*WIN_WIDTH] += window

    HOP = WIN_WIDTH // 2
    return output[:, HOP : -HOP]


def get_predicted_pianoroll(X, y, model):
    # predict by CNN
    X, y = torch.tensor(X), torch.tensor(y)
    X, y = X.to(DEVICE).float(), y.to(DEVICE).float()
    with torch.no_grad():
        y_hat = model(X)        # (batch, 1, 128, 64)
        attn = (X!=0).float()
        
        y_hat *= attn
    
    pianoroll = recreate_pianoroll(y_hat, overlap=True)
    return pianoroll


def set_threshold(pianoroll, cluster, min_threshold):
    print(f"start clustering (min threshold: {min_threshold})")
    mask = (pianoroll > min_threshold).astype(np.int)
    pianoroll *= mask
    pianoroll = pianoroll.reshape((-1,1))
    N_CLUSTER = 2
    target_cluster = 1
    print(f"max: {np.max(pianoroll)}, min: {np.min(pianoroll)}")

    pianoroll = pianoroll[iqr(pianoroll)]
    if cluster == 'kmeans':
        kmeans = KMeans(n_clusters=N_CLUSTER, init=np.array([min_threshold, pianoroll.max()]).reshape(-1,1))
        labels = kmeans.fit_predict(arr.reshape(-1,1))
    else:
        Z = pdist(pianoroll.reshape(-1,1))
        if cluster == 'single':
            X = fastcluster.single(Z)
        elif cluster == 'average':
            X = fastcluster.average(Z)
        elif cluster == 'centroid':
            X = fastcluster.centroid(Z)
        else:
            return 0.5
        labels = N_CLUSTER - fcluster(X, N_CLUSTER, 'maxclust')

    index = {}
    for i, l in enumerate(labels):
        index[l] = pianoroll[i]
        if len(index.keys()) == N_CLUSTER:
            break
    index = sorted(index.items(), key=lambda kv: kv[1])
    target_label = index[target_cluster-1][0]
    th = np.max(pianoroll[np.flatnonzero(labels == target_label)])
    print(f"find threshold: {th}")
    return th


def get_probability(note, pred_pianoroll, full_pianoroll, threshold, mode='median'):
    '''
    PARAMETER:
        note: class NOTE (start, end, pitch, velocity, Type)
        pred_pianoroll: pianoroll of shape (128, width) with probability range 0-1
        full_pianoroll: pianoroll of shape (128, width)
        threshold: decided by clustering
        mode: 'mean' or 'median'
    RETURN:
        probability
    '''
    s, e = int(note.start/60), int(note.end/60)-1
    while full_pianoroll[note.pitch][s] == 0:
        s += 1
    if s >= e:
        print(f'Error! Note with start {s} and end {e} (init start: {int(note.start/60)})')
        exit(1)
    m = pred_pianoroll[note.pitch, s:e]
    if mode == 'mean':
        p = m.mean()
    elif mode == 'median':
        p = np.median(m)
    return p if p > threshold else 0


def build_graph(notelist, pred_pianoroll, full_pianoroll, threshold):
    graph = np.full((len(notelist)+2, len(notelist)+2), np.inf)
    print('build graph')

    first_onset = min([note.start for note in notelist])
    # connect dummy start_node to 1st note
    connect = False
    i = 0
    while i < len(notelist):
        note = notelist[i]
        if note.start == first_onset:
            prob = get_probability(note, pred_pianoroll, full_pianoroll, threshold)
            if prob:
                connect = True
                # set probability from 0 to {i+1}
                graph[0, i+1] = -prob
            i += 1
        elif not connect:
            # reset onset to note.start
            first_onset = note.start
        else:
            break

    # connect notes
    for i, n1 in enumerate(notelist):
        connect = False
        j = i + 1
        if j < len(notelist):
            new_onset = notelist[j].start
        while j < len(notelist):
            if notelist[j].start >= n1.end:
                if notelist[j].start == new_onset:
                    prob = get_probability(notelist[j], pred_pianoroll, full_pianoroll, threshold)
                    if prob:
                        connect = True
                        graph[i+1, j+1] = -prob
                    j += 1
                elif not connect:
                    # reset onset to note.start
                    new_onset = notelist[j].start
                else:
                    break
            else:
                j += 1
        if not connect:
            graph[i+1, -1] = -0.5
    return graph


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


def get_predicted_melody_ind(predecessors):
    predicted_melody_ind = []
    last_pred = predecessors[0, -1]
    while last_pred != -9999:
        predicted_melody_ind.append(last_pred)
        last_pred = predecessors[0, last_pred]
    predicted_melody_ind = predicted_melody_ind[::-1][1:]
    return predicted_melody_ind


def cal_acc(predicted_melody_ind, notelist, file_ptr):
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
    print(f'tp:{tp}, fp:{fp}, fn:{fn}, tn:{tn}', file=file_ptr)
    print(f'acc: {(tp+tn)/(tp+fp+fn+tn)}\n')
    print(f'acc: {(tp+tn)/(tp+fp+fn+tn)}\n', file=file_ptr)
    return tp, fp, fn, tn


def get_melody_prediction(filename, model, cluster_method, min_th, mulinstr=True):
    notelist = get_notelist(filename, mulinstr=mulinstr)
    full_pianoroll, melody_pianoroll = midi_to_pianoroll(notelist)   # (128, 2754)
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

    # each pop song has its own threshold
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
    _, _, test_data = load_split(args.split_file)
    print(f'Loaded testset with {len(test_data)}')
   
    file_ptr = open('result.txt','w')
    TP, FP, FN, TN = 0, 0, 0, 0
    for i, subfile in enumerate(test_data):
        filename = f'{args.root}/test/{subfile}'
        print(f'=== {i}: {filename} ===')
        print(f'=== {i}: {filename} ===', file=file_ptr)
        notelist = get_notelist(filename)
        th, predicted_ind = get_melody_prediction(filename, model, args.cluster, args.min_threshold)
        print(f'threshold: {th}')
        print(f'threshold: {th}', file=file_ptr)
        tp, fp, fn, tn = cal_acc(predicted_ind, notelist, file_ptr)
        TP += tp; FP += fp; FN += fn; TN += tn
    
    print('\n OVERALL')
    print('\n OVERALL', file=file_ptr)
    print(f'tp:{TP}, fp:{FP}, fn:{FN}, tn:{TN}')
    print(f'tp:{TP}, fp:{FP}, fn:{FN}, tn:{TN}', file=file_ptr)
    print(f'acc: {(TP+TN)/(TP+FP+FN+TN)}\n')
    print(f'acc: {(TP+TN)/(TP+FP+FN+TN)}\n', file=file_ptr)

    file_ptr.close()


if __name__ == '__main__':
    main()
