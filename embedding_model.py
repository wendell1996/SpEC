import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import time


def sppmi(adj, k):
    G = adj.copy()
    nodeDegree = G.sum(axis=1)
    W = np.sum(nodeDegree)
    SPPMI = G
    [col, row] = np.nonzero(G)
    weights = G[col, row].reshape(len(col), 1)
    for i in range(len(col)):
        score = np.log(weights[i] * W / nodeDegree[col[i]] /
                       nodeDegree[row[i]]) - np.log(k)
        SPPMI[col[i], row[i]] = score if score > 0 else 0
    return SPPMI


def sparse_embedding_iteration(A, X, U, C, beta, alpha, n_iter=500, log=True, log_iter=50):
    eps = 2.22e-30
    lossPre = 1.79e30
    lossCur = lossPre - (lossPre * 1e-6)
    s_mode = sp.issparse(A)

    i = 1
    while (abs(lossPre - lossCur) / lossPre) > 1e-7 or i < n_iter:
        deno = U.transpose().dot(U).dot(C) + 2 * alpha * C
        if s_mode:
            deno[deno <= eps] = eps
            C = C.multiply((U.transpose().dot(X)).multiply(deno.power(-1)))
        else:
            deno[np.where(deno <= eps)] = eps
            C = np.multiply(C, ((U.transpose().dot(X)) / deno))

        nume = X.dot(C.transpose()) + 2 * beta * A.dot(U)
        if s_mode:
            nume[nume <= eps] = eps
        else:
            nume[np.where(nume <= eps)] = eps
        deno = U.dot(C.dot(C.transpose())) + 2 * alpha * U * sp.csr_matrix(np.ones((U.shape[1],U.shape[1]))) + 2 * beta * U.dot(
                    ((U.transpose()).dot(U)))
        if s_mode:
            deno[deno <= eps] = eps
            U = U.multiply(nume).multiply(deno.power(-1))
        else:
            deno[np.where(deno <= eps)] = eps
            U = np.multiply(U, np.power((nume / deno), 1))

        lossPre = lossCur
        if s_mode:
            residual = (X - U.dot(C)).data ** 2
            L1 = residual.sum()
        else:
            # residual = np.power((X - U.dot(C)).ravel(), 2)
            L1 = np.linalg.norm(X - U.dot(C),ord='fro')
        if s_mode:
            residual = (A - U.dot(U.transpose())).data ** 2
            L2 = residual.sum()
        else:
            # residual = np.power((A - U.dot(U.transpose())).ravel(), 2)
            L2 = np.linalg.norm(A - U.dot(U.transpose()),ord='fro')
        L3 = alpha * np.power(C.sum(0), 2).sum()
        L4 = alpha * np.power(U.sum(0), 2).sum()
        lossCur = L1 + L2 + L3 + L4
        if log and not i % log_iter:
            print(lossCur)
        i = i + 1
        if i > n_iter or (abs(lossPre - lossCur) / lossPre) < 5e-6:
            if log:
                print(i, lossPre, lossCur)
            break
    return U, C


def non_overlapping_detection(U):
    if sp.issparse(U):
        U[U < 1e-50] = 0
        predicted_labels = U.argmax(axis=1)
    else:
        U[np.where(U < 1e-50)] = 0
        predicted_labels = np.argmax(U, axis=1)
    return np.array(predicted_labels).flatten()


def overlapping_detection(U, threshold=None):
    if threshold is None:
        threshold = 0.1
    predicted_matrix = deepcopy(U)
    if sp.issparse(U):
        predicted_matrix[predicted_matrix < 1e-50] = 0
        predicted_matrix[predicted_matrix >= threshold] = 1
        predicted_matrix[predicted_matrix < threshold] = 0
    else:
        predicted_matrix[np.where(predicted_matrix < 1e-50)] = 0
        predicted_matrix[np.where(predicted_matrix >= threshold)] = 1
        predicted_matrix[np.where(predicted_matrix < threshold)] = 0
    return predicted_matrix
