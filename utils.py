# based on https://github.com/dawenl/vae_cf
# ImplicitSLIM is taken from https://github.com/ilya-shenbin/ImplicitSLIM

import numpy as np
from scipy import sparse
import pandas as pd
import os
import bottleneck as bn




def load_train_data(csv_file, n_items, n_users, global_indexing=False):
    tp = pd.read_csv(csv_file)
    
    n_users = n_users if global_indexing else tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items, n_users, global_indexing=False):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    if global_indexing:
        start_idx = 0
        end_idx = len(unique_uid) - 1
    else:
        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


def get_data(dataset, global_indexing=False):
    unique_sid = list()
    with open(os.path.join(dataset, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    
    unique_uid = list()
    with open(os.path.join(dataset, 'unique_uid.txt'), 'r') as f:
        for line in f:
            unique_uid.append(line.strip())
            
    n_items = len(unique_sid)
    n_users = len(unique_uid)
    
    train_data = load_train_data(os.path.join(dataset, 'train.csv'), n_items, n_users, global_indexing=global_indexing)


    vad_data_tr, vad_data_te = load_tr_te_data(os.path.join(dataset, 'validation_tr.csv'),
                                               os.path.join(dataset, 'validation_te.csv'),
                                               n_items, n_users, 
                                               global_indexing=global_indexing)

    test_data_tr, test_data_te = load_tr_te_data(os.path.join(dataset, 'test_tr.csv'),
                                                 os.path.join(dataset, 'test_te.csv'),
                                                 n_items, n_users, 
                                                 global_indexing=global_indexing)
    
    data = train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te
    data = (x.astype('float32') for x in data)
    
    return data


def ndcg(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.getnnz(axis=1)])

    # Handle cases where IDCG is zero
    # Because then the NDCG calculation is undefined
    # In practice, we only see 3-10 users per batch (out of 500) so 
    # it's not an issue. However, this should ideally be dealt with in preprocessing.
    ndcg_scores = np.zeros(batch_users)
    valid_users = IDCG > 0
    ndcg_scores[valid_users] = DCG[valid_users] / IDCG[valid_users]

    # Warning for users with zero IDCG
    zero_idcg_users = np.where(IDCG == 0)[0]
    if len(zero_idcg_users) > 0:
        # print(f"Found {len(zero_idcg_users)} users out of {batch_users} with zero IDCG (no relevant items). Their NDCG scores are set to 0.")
        # print(f"Example user indices with zero IDCG: {zero_idcg_users[:3]}")  # Print first 3 examples
        pass 
    return ndcg_scores


def recall(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def implicit_slim(W, X, λ, α, thr):
    A = W.copy().astype(np.float16)
    
    D = 1 / (np.array(X.sum(0)) + λ)
    
    ind = (np.array(X.sum(axis=0)) < thr).squeeze()
    A[:, ind.nonzero()[0]] = 0
    
    M = (λ * A + A @ X.T @ X) * D * D
    
    AinvC = λ * M + M @ X.T @ X
    AinvCAt = AinvC @ A.T

    AC = AinvC - AinvCAt @ np.linalg.inv(np.eye(A.shape[0]) / α + AinvCAt) @ AinvC
    
    return α * W @ A.T @ AC
