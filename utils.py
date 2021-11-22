import numpy as np
import os
import torch


#########################################################
#                    PIANOROLL                          #
#########################################################

def midi_to_pianoroll(notes, tick=60, win_height=128):
    max_end = max([note.end for note in notes])
    max_end = int(max_end/tick)
    score_pr = np.zeros((win_height, max_end))
    melody_pr = np.zeros((win_height, max_end))
    for note in notes:
        s, e = int(note.start/tick), int(note.end/tick)
        if note.Type == 0:
            melody_pr[note.pitch, s:e] = np.ones((1, e-s)) 
        score_pr[note.pitch, s:e] = np.ones((1, e-s)) 

    return score_pr, melody_pr
    

#########################################################
#                       MATH                            # 
#########################################################

def iqr(ys):
    q1, q3 = np.percentile(ys, [25,75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr*1.5)
    upper_bound = q3 + (iqr*1.5)
    return np.where((ys<upper_bound) | (ys>lower_bound))[0]


#########################################################
#                    MODEL TRAINING                     #
#########################################################

def save_checkpoint(state, is_best, path, target):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, best_loss = None):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = best_loss
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
