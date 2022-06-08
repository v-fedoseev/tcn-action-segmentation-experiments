import numpy as np
from torch.utils.data import Dataset
from os.path import join


class ActionVideoDataset(Dataset):
    def __init__(self, features_list_path, action_class_to_num, features_folder, gt_folder):
        with open(features_list_path, 'r') as f:
            videos = [line.split('.')[0] for line in f.readlines()[:-1]][:32]
        self.videos = videos  # list of video titles

        self.action_class_to_num = action_class_to_num  # dict mapping action class name to its index
        self.num_classes = len(self.action_class_to_num)
        self.features_folder = features_folder
        self.gt_folder = gt_folder

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        # read features from features file
        features_path = join(self.features_folder, f'{self.videos[idx]}.npy')
        features = np.load(features_path)
        # read labels from gt file
        gt_path = join(self.gt_folder, f'{self.videos[idx]}.txt')
        with open(gt_path, 'r') as gt_file:
            labels = [line.rstrip() for line in gt_file.readlines()]
        # convert labels to class indices with the mapping dict
        gt = np.zeros(len(labels), dtype=np.int32)
        for i, label in enumerate(labels):
            gt[i] = self.action_class_to_num[label]

        return {'features': features, 'gt': gt}
