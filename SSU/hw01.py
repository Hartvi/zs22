import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from copy import copy
import string
import sklearn

npr = np.array

alphabet = 'abcdefghijklmnopqrstuvwxyz'
np_alphabet = np.array('abcdefghijklmnopqrstuvwxyz')


def nums_to_letters(arr):
    return np_alphabet[arr]


def letter_to_number(a):
    return alphabet.index(a)


def letters_to_numbers(arr):
    return np.array([letter_to_number(_) for _ in arr])


def expan(x):
    return np.vstack([x, np.ones(x.shape[1])])  # (d+1, Xwidth)


def big_phi(Xheight, y):
    # 26 x a_lot
    ret = np.zeros((26, Xheight))
    ret[y, :] = 1
    return ret


def generalized_perceptron(X, Y, max_iterations=10):
    # X = list[np.ndarray[feature_height, length_of_sequence]]
    A = 26
    d = X[0].shape[0]
    V = np.zeros((A, d+1))
    aug_Y = np.hstack([letters_to_numbers(_) for _ in Y])  # (1, total_dataset_size = Xwidth below)
    aug_X = np.hstack(X)
    aug_X = np.vstack([aug_X, np.ones(aug_X.shape[1])])  # (d+1, Xwidth)
    Xwidth = aug_X.shape[1]
    Xheight = aug_X.shape[0]

    for i in range(max_iterations):
        print(f"On {i}-th iteration", end="\r")
        shattered = True
        for k in range(Xwidth):
            tmpX = aug_X[:, k]
            mul = V @ tmpX
            pred = np.argmax(mul)  # integer 0-25
            y = aug_Y[k]
            if pred != y:
                V += (big_phi(Xheight, y) - big_phi(Xheight, pred))*tmpX
                if shattered:
                    shattered = False
        if shattered:
            print("\nTASK 1 SHATTERED")
            break

    W = V[:, :-1]
    B = V[:, -1]
    return W, B


def classify_perceptron(X, Y, W, B):
    V = np.hstack([W, np.expand_dims(B, axis=1)])
    aug_Y = [letters_to_numbers(_) for _ in Y]  # (1, total_dataset_size = Xwidth below)
    m = len(Y)
    err_sum = 0.0
    char_err = 0
    num_of_chars = 0
    for sequence, label in zip(X, aug_Y):
        sequence = np.vstack([sequence, np.ones(sequence.shape[1])])  # (d+1, Xwidth)
        Y_hat = np.argmax(V @ sequence, axis=0)
        match = label == Y_hat
        if not match.all():
            err_sum += 1
        char_err += np.sum(match.astype(int))
        num_of_chars += len(label)
    return err_sum/m, 1 - char_err/num_of_chars


def evaltask2(sequence, V, G):
    """
    sequence: (L, )
    V = [W, b]:  (26, num_of_features+1)
    G: (26, 26)
    """
    L = sequence.shape[1]
    qs = V @ sequence  # A x L
    F = np.zeros((26, L))
    F[:, 0] = qs[:, 0]
    for l in range(1, L):
        for k in range(26):
            q_k_l = qs[k, l]
            max_F = np.max(F[:, l-1] + G[:, k])  # max over y so over column
            new_F_k_l = q_k_l + max_F
            F[k, l] = new_F_k_l
    Y = lambda i, y: np.argmax(F[:, i] + G[:, y])  # argmax over y' => column in G and column in F
    pred = np.zeros(L, dtype=int)
    pred[-1] = np.argmax(F[:, -1])
    for i in range(L-1, 0, -1):
        pred[i-1] = Y(i-1, pred[i])
    return tuple(pred)


def huge_phi(Y: np.ndarray, Xheight):
    V_update = np.zeros((26, Xheight))
    V_update[Y, :] = 1
    G_update = np.zeros((26,26))
    for i in range(1, len(Y)):
        G_update[Y[i-1], Y[i]] = 1
    return V_update, G_update


def task2(X, Y, max_iterations=2):
    aug_X = [expan(_) for _ in X]
    aug_Y = [tuple(letters_to_numbers(_)) for _ in Y]  # (1, total_dataset_size = Xwidth below)

    A = 26
    d = X[0].shape[0]
    Xheight = d+1
    V = np.zeros((A, Xheight))
    G = np.zeros((A, A))  # 26x26
    correct = 0
    incorrect = 1
    for i in range(max_iterations):
        print(f"On {i}-th iteration: {correct/(incorrect+correct)}/1", end="\r")
        correct = 0
        incorrect = 0
        shattered = True
        for sequence, label in zip(aug_X, aug_Y):
            pred = evaltask2(sequence, V, G)
            if pred != label:
                if shattered:
                    shattered = False
                for ith_letter in range(len(label)):
                    pred_letter = pred[ith_letter]
                    label_letter = label[ith_letter]
                    if pred_letter != label_letter:

                        V[pred_letter, :] -= sequence[:, ith_letter]
                        V[label_letter, :] += sequence[:, ith_letter]
                        if ith_letter > 0:
                            prev_letter = label[ith_letter - 1]
                            G[prev_letter, pred[ith_letter]] -= 0.1
                            G[prev_letter, label[ith_letter]] += 0.1
                incorrect += 1
            else:
                correct += 1

        if shattered:
            print("\nTASK 2 SHATTERED")
            break
    W = V[:, :-1]
    B = V[:, -1]
    return W, B, G


def classify_task2(X, Y, W, B, G):
    aug_X = [expan(_) for _ in X]
    aug_Y = [letters_to_numbers(_) for _ in Y]  # (1, total_dataset_size = Xwidth below)
    A = 26
    d = X[0].shape[0]
    Xheight = d+1
    V = np.hstack([W, np.expand_dims(B, axis=1)])

    m = len(Y)
    err_sum = 0.0
    char_err = 0
    num_of_chars = 0
    preds = []
    for sequence, label in zip(aug_X, aug_Y):
        Y_hat = evaltask2(sequence, V, G)
        preds.append(Y_hat)
        # Y_hat = np.argmax(V @ sequence, axis=0)
        match = label == Y_hat
        if not match.all():
            err_sum += 1
        char_err += np.sum(match.astype(int))
        num_of_chars += len(label)
    # rnd_choice = np.random.choice(len(aug_X), 5)
    # for i in range(len(rnd_choice)):
    #     print("label:", aug_Y[rnd_choice[i]], " pred:", preds[rnd_choice[i]])
    return err_sum/m, 1-char_err/num_of_chars


def large_phi(Y, Xheight):
    V_update = np.zeros((26, Xheight))
    V_update[Y, :] = 1
    return V_update


def evaltask3(sequence, V, v):
    Y_L = v[sequence.shape[1]]
    pred = None
    max_score = -float('inf')
    for letters in Y_L:
        score = np.sum(V[letters, :] @ sequence) + Y_L[letters]
        if score > max_score:
            pred = letters
            max_score = score
    return pred


def task3(X, Y, max_iterations=2):
    aug_X = [expan(_) for _ in X]
    aug_Y = [tuple(letters_to_numbers(_).tolist()) for _ in Y]  # (1, total_dataset_size = Xwidth below)

    A = 26
    d = X[0].shape[0]
    Xheight = d+1
    V = np.zeros((A, Xheight))
    v = dict()
    for y in aug_Y:
        len_y = len(y)
        if len_y not in v:
            v[len_y] = dict()
        v_len_y = v[len_y]
        if y not in v_len_y:
            v_len_y[y] = 0

    for i in range(max_iterations):
        print(f"On {i}-th iteration", end="\r")
        shattered = True
        for sequence, label in zip(aug_X, aug_Y):
            seq_len = sequence.shape[1]
            pred = evaltask3(sequence, V, v)
            np_pred = np.array(pred)
            np_label = np.array(label)
            if not (np_pred == np_label).all():
                V_update_gt = large_phi(label, Xheight)
                V_update_pred = large_phi(pred, Xheight)
                for k in range(seq_len):
                    V_update_gt[label[k], :] = V_update_gt[label[k], :]*sequence[:, k]
                    V_update_pred[pred[k], :] = V_update_pred[pred[k], :]*sequence[:, k]
                V += (V_update_gt - V_update_pred)
                v[seq_len][pred] -= 1
                v[seq_len][pred] += 1
                if shattered:
                    shattered = False
        if shattered:
            print("\nTASK 3 SHATTERED")
            break
    W = V[:, :-1]
    B = V[:, -1]
    return W, B, v


def classify_task3(X, Y, W, B, v):
    aug_X = [expan(_) for _ in X]
    aug_Y = [letters_to_numbers(_) for _ in Y]  # (1, total_dataset_size = Xwidth below)
    A = 26
    d = X[0].shape[0]
    Xheight = d+1
    V = np.hstack([W, np.expand_dims(B, axis=1)])

    m = len(Y)
    err_sum = 0.0
    char_err = 0
    num_of_chars = 0
    for sequence, label in zip(aug_X, aug_Y):
        Y_hat = evaltask3(sequence, V, v)
        # Y_hat = np.argmax(V @ sequence, axis=0)
        match = label == Y_hat
        if not match.all():
            err_sum += 1
        char_err += np.sum(match.astype(int))
        num_of_chars += len(label)
    return err_sum/m, 1 - char_err/num_of_chars


