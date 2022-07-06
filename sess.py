from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import cv2
from utils import apply_transforms, load_image, save_img_with_heatmap, sliding_window, resize_img, save_heatmap, tensor_to_img

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

class Patch:
    def __init__(self, source_img, coordinate, main_id=-1, sub_id=-1):
        """

        Parameters
        ----------
        source_img: tensor(cpu, torch.float32, 1x3x224x224)
        coordinate: tuple (tuple(int), x1, y1, x2, y2)
        main_id: int
        sub_id: int
        """
        self.source_img = source_img
        self.coordinate = coordinate # Format: x1, y1, x2, y2
        self.main_id = main_id
        self.sub_id = sub_id

    def patch(self):
        # print(self.source_img.shape, self.coordinate)
        x1, y1, x2, y2 = self.coordinate
        return self.source_img[:, :, y1:y2, x1:x2]

class SESS:
    def __init__(self, visualiser, pre_filter_ratio=0, theta=0,
                window_size=224, step_size=224, min_overlap_ratio=1, pool='mean',
                requires_grad =True, scales = None, smooth=True, output=None, verbose=0, device=None):
        """

        """
        self.window_size = window_size
        self.step_size = step_size
        self.visualiser = visualiser
        self.pre_filter_ratio = pre_filter_ratio
        self.theta = theta
        self.scales = scales
        self.output = output
        self.requires_grad = requires_grad
        self.verbose = verbose
        self.pool = pool
        self.min_overlap_ratio = min_overlap_ratio
        self.base_width = None
        self.base_height = None
        self.smooth = smooth
        assert min_overlap_ratio <= 1
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if scales is None:
            self.scales = [224 + 64 * i for i in range(12)]

    def forward(self, img, class_idx=None, retain_graph=False):
        """

        Parameters
        ----------
        img: tensor (cpu, shape: 1x3x224x224)
        class_idx: int
        retain_graph

        Returns
        -------

        """
        self.img = img
        self.base_height, self.base_width = self.img.shape[-2:]
        if class_idx is None:
            class_idx = self.get_id(self.img.to(self.device))
            if self.verbose: print('The target class id is ', class_idx)
        patches = self.collect_patches(self.img)
        if self.verbose: print('Total number of all extracted patches: ', len(patches))
        probs = self.get_scores(patches, class_idx)
        # print(probs.shape)
        heatmaps, probs = self.get_saliency(patches, probs, class_idx, self.pre_filter_ratio)
        if self.verbose: print('Total number of channels: ', len(heatmaps))
        # for i, item in enumerate(zip(heatmaps, probs)):
        #     heatmap, prob = item
        #     cam = save_img_with_heatmap(img, heatmap, 'temp/{}.jpg'.format(i), style='zhou')
        if self.pool == 'max':
            merged_heatmap = self.max_pool(heatmaps)
        elif self.pool == 'mean':
            merged_heatmap = self.mean_pool(heatmaps, self.theta)
        else:
            raise('Wrong pool type')

        if self.smooth:
            merged_heatmap = self.smoother(merged_heatmap)
        return merged_heatmap, class_idx

    def get_id(self, img_tensor):
        # predication on raw x
        logit = self.visualiser.model(img_tensor.to(self.device))
        class_idx = logit.max(1)[-1].item()
        return class_idx

    def collect_patches(self, img):
        patches = []
        window_size = self.window_size
        for _scale in self.scales:
            assert _scale >= window_size
            cur_img = transforms.Resize(_scale)(img)
            cur_height, cur_width = cur_img.shape[-2:]
            for window_pos in sliding_window(cur_height, cur_width, self.step_size, (window_size, window_size)):
                x1, y1, x2, y2 = window_pos
                patch =  Patch(cur_img, (x1, y1, x1 + window_size, y1 + window_size), len(patches), 0)
                patches.append(patch)
        return patches

    def get_scores(self, patches, class_idx=None):
        # _patches = []

        class CustomDataset(Dataset):
            """Custom dataset."""

            def __init__(self, patches):
                self.patches = patches

            def __len__(self):
                return len(self.patches)

            def __getitem__(self, idx):
                # print(idx, self.patches[idx].patch()[0].shape)
                return idx, self.patches[idx].patch()[0]

        num_workers = 4
        batch_size = 128
        loader = DataLoader(
            CustomDataset(patches),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        logits = []
        with torch.no_grad():
            for i, item in enumerate(loader):
                idx, patch_img = item
                logit = self.visualiser.model(patch_img.to(self.device))
                logits.append(logit.detach())

        if len(logits) > 1:
            logits = torch.squeeze(torch.cat(logits, axis=0))
        else:
            logits = torch.cat(logits, axis=0)
        # print(logits.shape)
        scores = F.softmax(logits, dim=1)[:, class_idx]
        # print('score shape:', scores.shape)
        return scores

    def get_saliency(self, patches, probs, class_idx=None, theta=0.):
        heatmaps = []
        selected_patches = []
        probs = probs.cpu().numpy()
        idx = np.argsort(probs)
        max_prob = probs[idx[-1]]
        norm_probs = probs / max_prob
        selected_norm_probs = []
        start = int(len(probs) * theta)
        ideal_theta = 1
        if theta < ideal_theta:
            ideal = int(len(probs) * ideal_theta)
        else:
            ideal = start

        if ideal == len(probs):
            ideal = ideal -1

        if ideal < 0:
            ideal = 0
        # start_prob = norm_probs[idx[start]] * 0.9  # To avoid numerical error
        # ideal_prob = norm_probs[idx[ideal]]*  0.9  # To avoid numerical error
        start_prob = norm_probs[idx[start]]
        ideal_prob = norm_probs[idx[ideal]]
        if self.verbose:
            print('Total number of remained patches:', (len(idx) - start))
        main_patches = []
        norm_main_probs = []
        for i in range(start, len(idx)):
            # print(idx[i])
            main_patches.append(patches[idx[i]])
            norm_main_probs.append(norm_probs[idx[i]])

        for i, cur_patch in enumerate(main_patches):
            patch_tensor = cur_patch.patch()
            heatmap, softmax = self.visualiser(patch_tensor.to(self.device), class_idx=class_idx)
            cur_norm_prob = torch.squeeze(softmax)[class_idx]/ max_prob
            cur_norm_prob = cur_norm_prob.cpu().numpy()
            heatmaps.append(heatmap)
            selected_norm_probs.append(cur_norm_prob)
            selected_patches.append(cur_patch)

        heatmaps = np.concatenate(heatmaps, axis=1)[0]
        # Normalise
        heatmaps = heatmaps - np.min(heatmaps, axis=(1, 2), keepdims=True)
        heatmaps = heatmaps / np.max(heatmaps, axis=(1, 2), keepdims=True)

        if True:
            # Channel-wise weight
            heatmaps = np.expand_dims(selected_norm_probs, axis=(1, 2)) * heatmaps

        full_heatmaps = []
        for i in range(len(heatmaps)):
            cur_patch = selected_patches[i]
            # print(cur_patch.coordinate)
            x1, y1, x2, y2 = cur_patch.coordinate
            assert x2 - x1 == 224
            assert y2 - y1 == 224
            height, width = cur_patch.source_img.shape[-2:]
            full_heatmap = np.zeros((height, width))
            full_heatmap[y1:y2, x1:x2] = heatmaps[i]
            full_heatmap = cv2.resize(full_heatmap, (self.base_width, self.base_height))
            full_heatmaps.append(full_heatmap)

            if self.output is not None:
                main_id, sub_id = cur_patch.main_id, cur_patch.sub_id
                if False:
                    save_heatmap(heatmaps[i], (self.output + '/{}_{}_cam.jpg').format(main_id, sub_id))
                cur_img = tensor_to_img(cur_patch.patch().detach())
                if True:
                    cur_img.save((self.output + '/{}_{}_img.jpg').format(main_id, sub_id))
                if False:
                    save_img_with_heatmap(cur_img ,heatmaps[i], (self.output + '/{}_{}_cam_patch.jpg').format(main_id, sub_id),
                                      style='zhou', normalise=False)
                if True:
                    save_img_with_heatmap(tensor_to_img(self.img.detach()), full_heatmap,
                                          (self.output + '/{}_{}_cam_img.jpg').format(main_id, sub_id),
                                      style='zhou', normalise=False)
        return full_heatmaps, norm_probs

    def mean_pool(self, heatmaps, theta):
        # weighted average
        heatmaps = np.array(heatmaps)
        total = np.sum(heatmaps > theta, axis=0)
        heatmaps[heatmaps <= theta] = 0
        mean_heatmap = np.true_divide(np.sum(heatmaps, axis=0), total)
        mean_heatmap = np.nan_to_num(mean_heatmap)
        return mean_heatmap

    def max_pool(self, heatmaps):
        heatmaps = np.array(heatmaps)
        return np.max(heatmaps, axis=0)

    def smoother(self, heatmap):
        return cv2.GaussianBlur(heatmap, (11, 11), 5.0, 0)

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)


