import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
import ipdb
from sklearn.manifold import TSNE


class kReciprocalReRanking(object):
    """
    k-reciprocal re-ranking from https://arxiv.org/pdf/1701.08398.pdf
    reranks an initial ranking of embeddings based on shared nearest neighbors
    """
    def __init__(self, queries, samples, cosine=False):
        # ipdb.set_trace()
        vecs = torch.cat([queries, samples], dim=0)
        if cosine:
            # cosine distance = 1 - cosine similarity
            self.original_dist = 1 - torch.matmul(vecs, vecs.T)
        else:
            self.original_dist = torch.cdist(vecs, vecs, p=2)

        self.query_num = queries.shape[0]
        gallery_num = vecs.shape[0]
        self.all_num = gallery_num

    def forward(self, k1=20, k2=5, l=0.1): # finetuning of k1, k2 and lambda

        '''

        k1: number of initial neighbors to consider for each sample
        k2: number of reciprocal neighbors to consider for each neighbor
        l: lambda, weight of original distance

        '''

        initial_rank = torch.argsort(self.original_dist, dim=1)
        V = torch.zeros_like(self.original_dist).type(torch.float32).cuda()
        # ipdb.set_trace()
        for i in range(self.original_dist.shape[0]):
            # k-reciprocal neighbors
            k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, i, k1)
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_k_reciprocal_index = self.k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
                # ipdb.set_trace()
                if len(np.intersect1d(candidate_k_reciprocal_index.cpu().numpy(), k_reciprocal_index.cpu().numpy())) > 2. / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))


            k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
            # # check the number of extra samples added to the original neighbors
            # extra = len(k_reciprocal_expansion_index) - len(k_reciprocal_index)
            # if extra > 0:
            #     print(f"Added {extra} extra samples to the original {k1} neighbors.")
            weight = torch.exp(-self.original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = 1. * weight / torch.sum(weight)

        self.original_dist = self.original_dist[:self.query_num,]

        if k2 != 1:
            V_qe = torch.zeros_like(V, dtype=torch.float32)
            for i in range(self.all_num):
                V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], dim=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(self.all_num):
            invIndex.append(torch.where(V[:, i] != 0)[0])

        jaccard_dist = torch.zeros_like(self.original_dist, dtype=torch.float32).cuda()

        for i in range(self.query_num):
            temp_min = torch.zeros(1, self.all_num, dtype=torch.float32).cuda()
            indNonZero = torch.where(V[i, :] != 0)[0]
            # ipdb.set_trace()
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.min(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

        final_dist = jaccard_dist * (1 - l) + self.original_dist * l
        # del self.original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:self.query_num, self.query_num:]
        return final_dist, self.original_dist

    @staticmethod
    def k_reciprocal_neigh(initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]

        fi = torch.where(backward_k_neigh_index == i)[0]
        return forward_k_neigh_index[fi]



def alpha_query_expansion(vecs, alpha = 3, n = 10):
    """
        Update query vectors by alpha query expansion from https://arxiv.org/pdf/1711.02512.pdf
        :param vecs: embedding vectors, shape (n, d), normalized to unit length
        :param alpha: power weight for weighted mean
        :param n: number of top vectors to use
        """

    # normalize vectors for cosine similarity
    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

    # similarity scores, closer to 1 means more similar
    scores = torch.matmul(vecs, vecs.T)
    ranks = torch.argsort(-scores, dim=1) # from more similar to less similar

    newvecs = torch.zeros_like(vecs)
    for i in range(vecs.shape[0]):
        nqe = ranks[i, :n]
        weights = scores[i, nqe]
        # ipdb.set_trace()
        weights = torch.pow(weights, alpha)
        newvecs[i] = torch.sum(weights[:, None].repeat(1, vecs.shape[1]) * vecs[nqe], dim=0)
    # ipdb.set_trace()
    newqvecs = torch.nn.functional.normalize(newvecs, p=2, dim=1)
    return newqvecs


import torch.nn.functional as F
import scipy
import numpy as np
from scipy.spatial.distance import cdist


class Diffuser(object):
    """
    Diffuse the labels on the graph created from the data points
    """
    def __init__(self, cfg):
        self.alpha = cfg.diffuse_alpha
        self.gamma = cfg.diffuse_gamma
        self.k_adap = cfg.diffuse_k_adap
        self.k = cfg.diffuse_k # k nearest neighbors for constructing the graph
        self.k_value = cfg.diffuse_k_value
        self.diffusion_value_use_entropy = cfg.diffusion_value_use_entropy
        self.max_iter = 20
        self.n_classes = cfg.n_ranks
        self.class_weights = np.ones((self.n_classes,))

    def diffuse(self, X, labels, len_labeled):
        """
        X: N x D matrix of data points
        labels: N x 1 vector of labels
        """

        X = X/np.linalg.norm(X, axis=1)[:, None]

        N = X.shape[0]
        # cosine similarity
        D = np.dot(X, X.T)
        I = np.argsort(-D, axis=1)

        # Create the graph
        I = I[:, 1:]
        W = np.zeros((N, N))
        for i in range(N):
            W[i, I[i, :self.k]] = D[i, I[i, :self.k]] ** self.gamma
        W = W + W.T

        # Normalize the graph
        W = W - np.diag(np.diag(W))
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, self.n_classes))
        A = np.eye(Wn.shape[0]) - self.alpha * Wn
        for i in range(self.n_classes):
            y = np.zeros((N,))
            this_class = labels == i
            y[this_class] = 1 / np.sum(this_class)
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=self.max_iter)
            Z[:, i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 < 0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(self.n_classes)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)

        p_labels[:len_labeled] = labels[:len_labeled] # do not update the labels of the labeled data
        weights[:len_labeled] = 1.0

        self.p_weights = weights.tolist()
        self.p_labels = p_labels

        # Compute the weight for each class
        for i in range(self.n_classes):
            cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
            self.class_weights[i] = (float(labels.shape[0]) / self.n_classes) / cur_idx.size

        return



    def diffuse_pred(self, X, label_ranks, label_values):
        """
        X: N x D matrix of data points, both labeled and pred data
        label_ranks: n x 1 vector of rank labels generated from values with narrow gap, use -1 for validation data

        """

        self.n_classes = len(np.unique(label_ranks))
        n = len(label_ranks) # labeled data
        label_ranks = np.concatenate((label_ranks, np.ones((X.shape[0]-n,))*-1))
        label_values = np.concatenate((label_values, np.ones((X.shape[0]-n,))*-1))

        X = X / np.linalg.norm(X, axis=1)[:, None]
        N = X.shape[0]
        # cosine similarity
        D = np.dot(X, X.T)
        I = np.argsort(-D, axis=1)

        # Create the graph
        I = I[:, 1:]
        W = np.zeros((N, N))

        ks = []

        for i in range(N):
            if self.k_adap:
                pec = np.percentile(D[i, :], self.k)
                kk = np.sum(D[i, :] > pec)
                W[i, I[i, :kk]] = D[i, I[i, :kk]] ** self.gamma
                ks.append(kk)
            else:
                # using fixed k
                W[i, I[i, :self.k]] = D[i, I[i, :self.k]] ** self.gamma

        W = W + W.T

        if self.k_adap:
            print(f"Average adaptive k: {np.mean(ks)}"
                    f"\nMax adaptive k: {np.max(ks)}"
                    f"\nMin adaptive k: {np.min(ks)}")


        # Normalize the graph
        W = W - np.diag(np.diag(W))
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        Wn = D * W * D

        Z = np.zeros((N, self.n_classes))
        A = np.eye(Wn.shape[0]) - self.alpha * Wn
        for i in range(self.n_classes):
            y = np.zeros((N,))
            this_class = label_ranks == i
            y[this_class] = 1 / np.sum(this_class)
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=self.max_iter)
            Z[:, i] = f

        Z_value = np.zeros((N, n))
        for i in range(n):
            y_value = np.zeros((N,))

            y_value[i] = 1
            f_value, _ = scipy.sparse.linalg.cg(A, y_value, tol=1e-6, maxiter=self.max_iter)
            Z_value[:, i] = f_value

        # Handle numberical errors
        Z[Z < 0] = 0
        Z_value[Z_value < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 <= 0] = 1e-7
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(self.n_classes)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)

        probs_value = F.normalize(torch.tensor(Z_value), 1).numpy()
        # weighted sum of top k values
        probs_value[probs_value < 0] = 0

        if self.diffusion_value_use_entropy:
            # using entropy as weights
            indx = np.argsort(-probs_value, axis=1)[:, :self.k_value]
            entropy_value = scipy.stats.entropy(probs_value.T)

            extracted_entropy = entropy_value[indx]
            weights_value = 1 - extracted_entropy / self.k_value

            # normalize the weights
            weights_value = weights_value / np.sum(weights_value, axis=1)[:, None]
            p_values = np.sum(label_values[indx] * weights_value, axis=1)


        else:
            # using prob as weights
            # using fixed k
            indx = np.argsort(-probs_value, axis=1)[:, :self.k_value]
            extract_probs = [probs_value[i, indx[i]] for i in range(probs_value.shape[0])]

            extract_probs = np.array(extract_probs)
            # normalize the probs
            extract_probs = extract_probs / (np.sum(extract_probs, axis=1)+ 1e-8)[:, None]
            p_values = np.sum(label_values[indx] * extract_probs, axis=1)


            # update embs based on the weights
            p_embs = np.sum(X[indx] * extract_probs[:, :, None], axis=1)

        p_labels[:n] = label_ranks[:n]  # do not update the labels of the labeled data
        weights[:n] = 1.0

        self.p_weights = weights.tolist()
        self.p_labels = p_labels
        self.p_values = p_values
        self.p_embs = p_embs
        self.X = X
        self.indx = indx
        self.extract_probs = extract_probs
        self.probs_value = probs_value


        return



