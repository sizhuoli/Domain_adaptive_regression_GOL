
import random

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils.util import load_one_image


class Train(Dataset):
    def __init__(self, conf, imgs, labels, sample_weights, ranks, tau, norm_target=False,
                 logscale=False, is_filelist=True):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs
        self.labels = labels
        self.ranks = ranks
        self.sample_weights = sample_weights
        self.n_imgs = len(self.imgs)
        self.labels = np.array(self.labels)
        self.min_target_bf_norm = int(self.labels.min())
        if logscale:
            self.labels = np.log(labels.astype(np.float32) + conf.offset)
        else:
            if norm_target:
                self.labels = self.labels - min(self.labels)

        self.tau = tau
        self.is_filelist = is_filelist

    def __getitem__(self, item):
        order_label, ref_idx = self.find_reference(self.ranks[item], self.ranks, min_rank=min(self.ranks),
                                                   max_rank=max(self.ranks))


        if self.is_filelist:
            base_img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
            ref_img = np.asarray(load_one_image(self.imgs[ref_idx])).astype('uint8')
        else:
            base_img = np.asarray(self.imgs[item]).astype('uint8')
            ref_img = np.asarray(self.imgs[ref_idx]).astype('uint8')

        base_img = transform_img(base_img, self.conf, mode='train')
        ref_img = transform_img(ref_img, self.conf, mode='train')

        base_target = self.labels[item]
        ref_target = self.labels[ref_idx]

        base_rank = self.ranks[item]
        ref_rank = self.ranks[ref_idx]

        return base_img, ref_img, order_label, [base_rank, ref_rank], [base_target, ref_target], item

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
                ref_range_max = base_rank - self.tau - epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue
            elif order == 1:  # base_rank < ref_rank - tau
                ref_range_min = base_rank + self.tau + epsilon
                ref_range_max = max_rank
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
                    continue

            else:  # |base_rank - ref_rank| <= tau
                ref_range_min = base_rank - self.tau - epsilon
                ref_range_max = base_rank + self.tau + epsilon
                candidates = get_indices_in_range([ref_range_min, ref_range_max], ref_ranks)
                if len(candidates) > 0:
                    ref_idx = candidates[rng.choice(len(candidates), 1)[0]][0]
                else:
                    order = (order + 1) % 3
                    debug_flag += 1
        return order, ref_idx


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



class Valid(Dataset):
    def __init__(self, conf, imgs, labels, ranks, norm_target=False, is_filelist=True, return_ranks=True):
        super(Dataset, self).__init__()
        self.conf = conf
        self.imgs = imgs
        self.labels = labels
        self.labels = np.array(self.labels)
        self.ranks = ranks
        self.n_imgs = len(self.imgs)
        self.is_filelist = is_filelist
        if conf.logscale:
            self.labels = np.log(np.array(labels).astype(np.float32) + conf.offset)
        if norm_target:
            self.labels = self.labels - min(self.labels)
        self.return_ranks = return_ranks


    def __getitem__(self, item):
        if self.is_filelist:
            img = np.asarray(load_one_image(self.imgs[item])).astype('uint8')
        else:
            img = np.asarray(self.imgs[item]).astype('uint8')
        img = transform_img(img, self.conf, mode='valid')

        if self.return_ranks:
            return img, self.labels[item], self.ranks[item], item
        else:
            return img, self.labels[item], item

    def __len__(self):
        return len(self.imgs)



def transform_img(rawImg, arg, mode='train'):
    # for only
    if arg.center_cp:
        # center cropping
        center = center_crop(rawImg, (arg.croplength, arg.croplength))
    else:
        center = rawImg

    if arg.circle_cp:
        rad = int(center.shape[-1] / 2)
        mask = create_circular_mask(rad * 2, rad * 2, radius=rad)

        center[:, ~mask] = 0
    circle = np.transpose(center, axes=(1, 2, 0))
    p_transform = random.random()
    Transform = []
    if (mode == 'train') and p_transform <= arg.augmentation_prob:
        Transform.append(T.ToPILImage())
        Transform.append(T.RandomHorizontalFlip())
        Transform.append(T.RandomVerticalFlip())
        Transform.append(T.RandomEqualize(p=0.5))
        # gaussian blur
        Transform.append(T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
        if arg.center_cp:
            Transform.append(T.RandomRotation(180))
        Transform.append(T.RandomAdjustSharpness(sharpness_factor=1.5))
        Transform.append(T.ColorJitter(brightness=.5, contrast=.5, saturation=.3, hue=.3))

        Transform = T.Compose(Transform)
        circle = Transform(circle)
        Transform = []

    Transform.append(T.ToTensor())
    Transform.append(T.Normalize(mean=arg.mean, std=arg.std))
    Transform.append(T.Resize((arg.inputlength, arg.inputlength)))
    Transform = T.Compose(Transform)
    img_final = Transform(circle)
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

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

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







