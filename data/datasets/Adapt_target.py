import numpy as np
import random

import numpy as np
import scipy
import skimage
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.util import load_one_image


class Train(Dataset):
    def __init__(self, conf, imgs, labels, ranks):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs
        self.labels = labels # list of labels
        self.n_imgs = len(self.imgs)
        self.labels = np.array(self.labels)
        self.min_target_bf_norm = int(self.labels.min())
        self.ranks = ranks

    def __getitem__(self, item):

        order_label, ref_idx = self.find_reference(self.ranks[item], self.ranks, min_rank=min(self.ranks),
                                                   max_rank=max(self.ranks))

        base_img = np.asarray(load_one_image(self.imgs[item]))
        ref_img = np.asarray(load_one_image(self.imgs[ref_idx]))

        base_img = transform_img(base_img, self.conf, mode='train')
        ref_img = transform_img(ref_img, self.conf, mode='train')

        base_target = self.labels[item]
        ref_target = self.labels[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        return base_img, ref_img, [base_target, ref_target], [base_rank, ref_rank], item # need target in finetuning
    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=9, epsilon=1e-4):
        """
        randomly choose ref matching one of the order (0, 1, 2)

        """

        def get_indices_in_range(search_range, targets):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= targets, targets <= search_range[1]))

        rng = np.random.default_rng()
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = min_rank
                ref_range_max = base_rank - self.conf.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.conf.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.conf.tau - epsilon
                ref_range_max = base_rank + self.conf.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx


class Train_DSS(Dataset):

    '''dataset containing a combined dataset of labeled data and unlabeled data

    enable pseudo label update for unlabeled data during training

    keep track of labeled and unlabeled indices

    to be sampled with two stream batch sampler

    set weight for each sample (=1 for labeled data, to be updated for unlabeled data) for loss calculation

    '''

    def __init__(self, conf, imgs, values, ranks, labeled_idx, unlabeled_idx):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs # list of all images (labeld + unlabeled)
        self.values = values # list of values (real + pseudo)
        self.n_imgs = len(self.imgs)
        self.min_target_bf_norm = int(min(self.values))
        self.ranks = ranks # list of ranks (real + pseudo)
        self.n_ranks = len(np.unique(self.ranks))
        self.p_weights = np.ones((len(self.imgs),))
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.class_weights = np.ones((self.n_ranks,), dtype=np.float32)


    def __getitem__(self, item):

        order_label, ref_idx = self.find_reference(self.ranks[item], self.ranks, min_rank=min(self.ranks),
                                                   max_rank=max(self.ranks))

        base_img = np.asarray(load_one_image(self.imgs[item]))
        ref_img = np.asarray(load_one_image(self.imgs[ref_idx]))

        base_img = transform_img(base_img, self.conf, mode='train')
        ref_img = transform_img(ref_img, self.conf, mode='train')

        base_target = self.values[item]
        ref_target = self.values[ref_idx]

        # gt ranks
        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        return base_img, ref_img, order_label, [base_rank, ref_rank], item

    def __len__(self):
        return self.n_imgs

    def find_reference(self, base_rank, ref_ranks, min_rank=0, max_rank=9, epsilon=1e-4):
        """
        randomly choose ref matching one of the order (0, 1, 2)

        """

        def get_indices_in_range(search_range, targets):
            """find indices of values within range[0] <= x <= range[1]"""
            return np.argwhere(np.logical_and(search_range[0] <= targets, targets <= search_range[1]))

        rng = np.random.default_rng()
        order = np.random.randint(0, 3)
        ref_idx = -1
        debug_flag = 0
        while ref_idx == -1:
            if debug_flag == 3:
                raise ValueError(f'Failed to find reference... base_score: {base_rank}')
            if order == 0:  # base_rank > ref_rank + tau
                ref_range_min = min_rank
                ref_range_max = base_rank - self.conf.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.conf.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.conf.tau - epsilon
                ref_range_max = base_rank + self.conf.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx


    def update_pseudo_label(self, X, max_iter = 20):
        """update pseudo label for unlabeled data, keep original label for labeled data

        update sample weights for unlabeled data, keep original sample weights (=1) for labeled data

        """

        ranks = np.asarray(self.ranks)
        labeled_idx = np.asarray(self.labeled_idx)
        # unlabeled_idx = np.asarray(self.unlabeled_idx)
        X = X / np.linalg.norm(X, axis=1)[:, None]
        N = X.shape[0]
        # cosine similarity
        D = np.dot(X, X.T)
        I = np.argsort(-D, axis=1)

        # Create the graph
        I = I[:, 1:]
        W = np.zeros((N, N))
        for i in range(N):
            # W[i, I[i, :self.k]] = D[i, :self.k]
            W[i, I[i, :self.conf.diffuse_k]] = D[i, I[i, :self.conf.diffuse_k]] ** self.conf.diffuse_gamma
            # ipdb.set_trace()
        W = W + W.T

        # Normalize the graph
        W = W - np.diag(np.diag(W))
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, self.n_ranks))
        A = np.eye(Wn.shape[0]) - self.conf.diffuse_alpha * Wn
        for i in range(self.n_ranks):
            cur_idx = labeled_idx[np.where(ranks[labeled_idx] == i)]
            y = np.zeros((N,))
            y[cur_idx] = 1 / len(cur_idx)
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 < 0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(self.n_ranks)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)

        p_labels[labeled_idx] = ranks[labeled_idx]
        weights[labeled_idx] = 1

        self.p_weights = weights.tolist()
        self.p_labels = p_labels

        # Compute the weight for each class
        for i in range(self.n_ranks):
            cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
            self.class_weights[i] = (float(ranks.shape[0]) / self.n_ranks) / cur_idx.size


class Valid_DSS(Dataset):

    '''dataset containing a combined dataset of labeled data and unlabeled data

    enable pseudo label update for unlabeled data during training

    keep track of labeled and unlabeled indices

    to be sampled with two stream batch sampler

    set weight for each sample (=1 for labeled data, to be updated for unlabeled data) for loss calculation

    '''

    def __init__(self, conf, imgs, values, ranks, labeled_idx, unlabeled_idx):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs # list of all images (labeld + unlabeled)
        self.values = values # list of values (real + pseudo)
        self.n_imgs = len(self.imgs)
        self.min_target_bf_norm = int(min(self.values))
        self.ranks = ranks # list of ranks (real + pseudo)
        self.n_ranks = len(np.unique(self.ranks))
        self.p_weights = np.ones((len(self.imgs),))
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.class_weights = np.ones((self.n_ranks,), dtype=np.float32)


    def __getitem__(self, item):
        if self.conf.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item]))
        else:
            img = np.asarray(self.imgs[item])
        img = transform_img(img, self.conf, mode='valid')
        return img, self.values[item], self.ranks[item], item

    def __len__(self):
        return self.n_imgs


    def update_pseudo_label(self, X, max_iter = 20):
        """update pseudo label for unlabeled data, keep original label for labeled data

        update sample weights for unlabeled data, keep original sample weights (=1) for labeled data

        """

        ranks = np.asarray(self.ranks)
        labeled_idx = np.asarray(self.labeled_idx)
        # unlabeled_idx = np.asarray(self.unlabeled_idx)
        X = X / np.linalg.norm(X, axis=1)[:, None]
        N = X.shape[0]
        # cosine similarity
        D = np.dot(X, X.T)
        I = np.argsort(-D, axis=1)

        # Create the graph
        I = I[:, 1:]
        W = np.zeros((N, N))
        for i in range(N):
            # W[i, I[i, :self.k]] = D[i, :self.k]
            W[i, I[i, :self.conf.diffuse_k]] = D[i, I[i, :self.conf.diffuse_k]] ** self.conf.diffuse_gamma
            # ipdb.set_trace()
        W = W + W.T

        # Normalize the graph
        W = W - np.diag(np.diag(W))
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, self.n_ranks))
        A = np.eye(Wn.shape[0]) - self.conf.diffuse_alpha * Wn
        for i in range(self.n_ranks):
            cur_idx = labeled_idx[np.where(ranks[labeled_idx] == i)]
            y = np.zeros((N,))
            y[cur_idx] = 1 / len(cur_idx)
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 < 0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(self.n_ranks)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)
        p_labels[labeled_idx] = ranks[labeled_idx]
        weights[labeled_idx] = 1

        self.p_weights = weights.tolist()
        self.p_labels = p_labels

        # Compute the weight for each class
        for i in range(self.n_ranks):
            cur_idx = np.where(np.asarray(self.p_labels) == i)[0]
            self.class_weights[i] = (float(ranks.shape[0]) / self.n_ranks) / cur_idx.size



class Valid(Dataset):
    def __init__(self, conf, imgs, labels, ranks):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs
        self.labels = labels
        if conf.logscale:
            self.labels = np.log(np.array(labels).astype(np.float32)+conf.offset)
        self.n_imgs = len(self.imgs)
        self.ranks = ranks


    def __getitem__(self, item):
        if self.conf.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item]))
        else:
            img = np.asarray(self.imgs[item])
        img = transform_img(img, self.conf, mode='valid')

        return img, self.labels[item], self.ranks[item], item

    def __len__(self):
        return len(self.imgs)


class Train_vanilla(Dataset):
    def __init__(self, conf, imgs, labels, norm_target=False, is_filelist=True):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs
        self.labels = labels
        self.labels = np.array(self.labels)
        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if conf.logscale:
            self.labels = np.log(np.array(labels).astype(np.float32) + conf.offset)
        if norm_target:
            self.labels = self.labels - min(self.labels)


    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')
        img = transform_img(img, self.conf, mode='train')


        return img, self.labels[item], item

    def __len__(self):
        return len(self.imgs)

class Valid_vanilla(Dataset):
    def __init__(self, conf, imgs, labels):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs
        self.labels = labels
        if conf.logscale:
            self.labels = np.log(np.array(labels).astype(np.float32)+conf.offset)
        self.n_imgs = len(self.imgs)

    def __getitem__(self, item):
        if self.conf.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item]))
        else:
            img = np.asarray(self.imgs[item])
        img = transform_img(img, self.conf, mode='valid')

        return img, self.labels[item], item

    def __len__(self):
        return len(self.imgs)


def transform_img(rawImg, arg, mode='train'):
    rawImg = skimage.transform.resize(rawImg, (arg.inputchannel, arg.infer_croplength, arg.infer_croplength),
                                      preserve_range=True, anti_aliasing=True).astype(np.uint8)
    if arg.infer_center_cp:
        center = center_crop(rawImg, (arg.infer_croplength, arg.infer_croplength))
    else:
        center = rawImg

    center = np.transpose(center, axes=(1, 2, 0)) # channels last
    p_transform = random.random()
    Transform = []
    if (mode == 'train') and p_transform <= arg.augmentation_prob:
        Transform.append(T.ToPILImage())
        Transform.append(T.RandomHorizontalFlip())
        Transform.append(T.RandomVerticalFlip())
        Transform.append(T.RandomEqualize(p=0.5))
        # gaussian blur
        Transform.append(T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
        if arg.infer_center_cp:
            Transform.append(T.RandomRotation(180))
        Transform.append(T.RandomAdjustSharpness(sharpness_factor=1.5))
        Transform.append(T.ColorJitter(brightness=.5, contrast=.5, saturation=.3, hue=.3))

        Transform = T.Compose(Transform)
        center = Transform(center)
        Transform = []

    Transform.append(T.ToTensor())
    Transform.append(T.Normalize(mean=arg.mean, std=arg.std))
    # transform resize to input shape
    Transform.append(T.Resize((arg.infer_inputlength, arg.infer_inputlength)))
    Transform = T.Compose(Transform)
    img_final = Transform(center)
    if arg.infer_circle_cp:
        # inner circle cropping
        rad = int(img_final.shape[-1] / 2)
        mask = create_circular_mask(rad * 2, rad * 2, radius=rad)
        img_final[:, ~mask] = 0

    return img_final


def center_crop(img, crop_size, channel_last = 0):
    # Note: image_data_format is 'channel_last'
    # assert img.shape[2] == 4
    if channel_last:
        height, width = img.shape[0], img.shape[1]
    else:

        height, width = img.shape[1], img.shape[2]
    dy, dx = crop_size

    sx = (height - dx) // 2 # would be 10
    sy = (width - dy) // 2
    if channel_last:
        return img[sy:(sy+dy), sx:(sx+dx), :]
    else:
        return img[:, sy:(sy+dy), sx:(sx+dx)]

def center_random_crop(img, crop_size, channel_last = 0):
    # cropping center with random offset less than 25 pixel (5 m location error)
    if channel_last:
        height, width = img.shape[0], img.shape[1]
    else:
        height, width = img.shape[1], img.shape[2]
    dy, dx = crop_size
    # random offset with probability higher when closer to 10 and lower when closer to 0 or closer to 20
    prob = np.append(np.linspace(0, 1, 10), np.linspace(1, 0, 10))
    prob = prob / np.sum(prob)
    sx = np.random.choice(np.arange(0, 20), p=prob)
    sy = np.random.choice(np.arange(0, 20), p=prob)
    if channel_last:
        return img[sy:(sy+dy), sx:(sx+dx), :]
    else:
        return img[:, sy:(sy+dy), sx:(sx+dx)]




def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    dist_from_center = torch.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def map_rank(cls):
    rank = 0
    mapping = dict()
    for c in np.unique(cls):
        mapping[c] = rank
        rank += 1
    ranks = np.array([mapping[l] for l in cls])
    return ranks, mapping