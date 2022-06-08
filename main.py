import torch
import numpy as np
from torch.utils.data import DataLoader
from os.path import join
from os import makedirs

from model import TrainerTask1, TrainerTask2, TrainerTask3, TrainerTask4
from dataset import ActionVideoDataset


feature_len = 2048  # length of the feature vector for each video
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
data_folder = '../data'
train_features_list_path = join(data_folder, 'train.bundle')
test_features_list_path = join(data_folder, 'test.bundle')
features_folder = join(data_folder, 'features')
gt_folder = join(data_folder, 'groundTruth')
classes_list_path = join(data_folder, 'mapping.txt')

# Load class mapping file and get number of classes
action_class_to_num = dict()
with open(classes_list_path, 'r') as f:
    lines = f.readlines()
    for l in lines:
        action_class_to_num[l.split()[1]] = int(l.split()[0])
num_classes = len(action_class_to_num)

# init an instance of the dataset class from dataset.py
train_dataset = ActionVideoDataset(train_features_list_path, action_class_to_num, features_folder, gt_folder)

# Create output folders: save for model checkpoints, res for inference outputs
# and create subfolders for tasks, i.e. res/task1
save_folder = '../save'
save_dirs = [join(save_folder, f'task{i}') for i in range(1, 5)]
for d in save_dirs:
    try:
        makedirs(d)
    except:
        pass
res_folder = '../res'
res_dirs = [join(res_folder, f'task{i}') for i in range(1, 5)]
for d in res_dirs:
    try:
        makedirs(d)
    except:
        pass


def collate_videos(batch_list):
    # Custom collate_fn function for DataLoader to construct batches.
    # To process videos of different length in a batch:
    # 1) create batch tensors to fit the longest video (init features w/ 0 and gt w/ non-existent class -10)
    # 2) populate them with the feature and gt data for each video
    # 3) create a mask tensor to additionally store which indices were populated (1) and which were beyond a video (0)

    features_list = [item['features'] for item in batch_list]  # list of features in the batch
    gt_list = [item['gt'] for item in batch_list]  # list of ground truth vectors in the batch
    max_frames = max([len(gt) for gt in gt_list])  # maximum length of a video in the batch
    feature_length = features_list[0].shape[0]  # length of the feature vector

    features_batch = np.zeros((len(batch_list), feature_length, max_frames), dtype=float)  # features, 0s
    gt_batch = np.ones((len(batch_list), max_frames), dtype=int) * (-10)  # gt, -10s to distinguish the padding
    gt_video_batch = np.zeros((len(batch_list), num_classes), dtype=int)  # video-level gt, 0s for not present classes
    mask = np.zeros((len(batch_list), num_classes, max_frames), dtype=float)  # mask, 0s for padding

    # populate the tensors for each video
    for i, item in enumerate(batch_list):
        features_batch[i, :, :features_list[i].shape[1]] = features_list[i]
        gt_batch[i, :gt_list[i].shape[0]] = gt_list[i]
        present_classes = np.unique(gt_list[i]).tolist()  # classes present in the video
        gt_video_batch[i, present_classes] = 1  # set video-level gt to 1 at the indices of present classes
        mask[i, :, :gt_list[i].shape[0]] = np.ones((num_classes, gt_list[i].shape[0]), dtype=float)

    # convert numpy arrays to tensors
    feature_tensor = torch.tensor(features_batch, dtype=torch.float)
    gt_tensor = torch.tensor(gt_batch, dtype=torch.long)
    gt_video_tensor = torch.tensor(gt_video_batch, dtype=torch.float)
    mask_tensor = torch.tensor(mask, dtype=torch.float)

    # return the batch as a dict
    return {'features': feature_tensor, 'gt': gt_tensor, 'mask': mask_tensor, 'gt_video': gt_video_tensor}


# Functions running the train and predict code for each task
# comment .train() or .predict() to run just one of them
def task1(num_layers, num_filters, num_epochs, batch_size, lr):
    trainer = TrainerTask1(num_layers, num_filters, feature_len, num_classes)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_videos)
    trainer.train(dataloader, num_epochs, lr, save_dirs[0], device)
    trainer.predict(save_dirs[0], num_epochs, res_dirs[0], features_folder, test_features_list_path,
                    action_class_to_num, device)


def task2(num_stages, num_layers, num_filters, num_epochs, batch_size, lr):
    trainer = TrainerTask2(num_stages, num_layers, num_filters, feature_len, num_classes)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_videos)
    trainer.train(dataloader, num_epochs, lr, save_dirs[1], device)
    trainer.predict(save_dirs[1], num_epochs, res_dirs[1], features_folder, test_features_list_path,
                    action_class_to_num, device)


def task3(num_stages, num_layers, num_filters, num_epochs, batch_size, lr):
    trainer = TrainerTask3(num_stages, num_layers, num_filters, feature_len, num_classes)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_videos)
    trainer.train(dataloader, num_epochs, lr, save_dirs[2], device)
    trainer.predict(save_dirs[2], num_epochs, res_dirs[2], features_folder, test_features_list_path,
                    action_class_to_num, device)


def task4(scales, num_layers, num_filters, num_epochs, batch_size, lr):
    trainer = TrainerTask4(scales, num_layers, num_filters, feature_len, num_classes)
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_videos)
    trainer.train(dataloader, num_epochs, lr, save_dirs[3], device)
    trainer.predict(save_dirs[3], num_epochs, res_dirs[3], features_folder, test_features_list_path,
                    action_class_to_num, device)


if __name__ == '__main__':
    task1(num_layers=10, num_filters=64, num_epochs=50, batch_size=4, lr=0.001)
    task2(num_stages=4, num_layers=10, num_filters=64, num_epochs=50, batch_size=4, lr=0.001)
    task3(num_stages=4, num_layers=10, num_filters=64, num_epochs=50, batch_size=4, lr=0.001)
    task4(scales=[1, 4, 8], num_layers=10, num_filters=64, num_epochs=50, batch_size=4, lr=0.001)
