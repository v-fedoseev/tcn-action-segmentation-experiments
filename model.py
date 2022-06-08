import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from os.path import join


def dilation_sequence(num_layers, dilation_type='linear'):
    # generating sequence of dilation factors
    if dilation_type == 'linear':
        return [i+1 for i in range(num_layers)]


class DilatedResidualLayer(nn.Module):
    # dilated conv -> relu -> 1x1 conv -> dropout (section 3.4) -> add input (residual)
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        # same padding as dilation, so border elements also contribute in all border convolutions
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out += x
        out = out * mask[:, 0:1, :]  # mask the output to set the results beyond the videos' boundaries to 0

        return out


class SSTCN(nn.Module):
    # Single-stage TCN
    # 1x1 conv -> DilatedResidualLayer (num_layers times) -> 1x1 conv (classification layer conv_class)
    def __init__(self, num_layers, num_filters, feature_len, num_classes, sample_rate=1, add_pre_class_out=False):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(in_channels=feature_len, out_channels=num_filters, kernel_size=1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(in_channels=num_filters, out_channels=num_filters, dilation=dilation)
            for dilation in dilation_sequence(num_layers)
        ])
        self.conv_class = nn.Conv1d(in_channels=num_filters, out_channels=num_classes, kernel_size=1)
        self.sample_rate = sample_rate  # sampling each Nth frame, 1 for full video resolution
        self.add_pre_class_out = add_pre_class_out  # True if should output pre-classification-layer result as well

    def forward(self, x, mask):
        if self.sample_rate > 1:
            # sample input and mask
            x = x[:, :, ::self.sample_rate]
            mask_sampled = mask[:, :, ::self.sample_rate]

        out = self.conv_1x1(x)  # first 1x1 conv
        for layer in self.layers:  # forward through layers
            out = layer(out, mask_sampled)

        if self.sample_rate > 1:
            # upsample the pre-class output
            out = out.repeat(1, 1, self.sample_rate)
            # trim to original length: down- and upsampling w/ slicing produce the next multiple of sample rate
            out = out[:, :, :mask.shape[-1]]

        class_out = self.conv_class(out)  # get the classification layer output
        class_out = class_out * mask[:, 0:1, :]  # mask the output (pre-class output masked already in the last layer)

        if self.add_pre_class_out:
            return {'out': class_out, 'pre_class_out': out}  # add the pre-class output
        else:
            return class_out


class MSTCN(nn.Module):
    # Multi-stage TCN
    # stage 1 (feature_len to num_filters) -> stages 2...num_stages (num_filters to num_filters)
    # outputs a concatenation of all stages' outputs
    def __init__(self, num_stages, num_layers, num_filters, feature_len, num_classes, add_input=True):
        super().__init__()
        self.stage1 = SSTCN(num_layers, num_filters, feature_len, num_classes)
        self.add_input = add_input  # True if should concatenate input features to output of each stage except last
        stages_feature_len = feature_len
        if add_input:
            stages_feature_len += num_classes  # increase length of input features by number of classes
        self.stages = nn.ModuleList([
            SSTCN(num_layers, num_filters, stages_feature_len, num_classes)
            for _ in range(num_stages - 1)
        ])

    def forward(self, x, mask):
        out = self.stage1(x, mask)  # first stage output
        output_stages = out.unsqueeze(0)  # constructing the concatenated output from all stages

        # forward through the rest of the stages
        for stage in self.stages:
            # get probabilities from logits before passing to the next stage, mask it
            out = F.softmax(out, dim=1) * mask[:, 0:1, :]
            if self.add_input:
                # concatenate the input features
                out = torch.cat((out, x), dim=1)
            out = stage(out, mask)  # forward through stage
            output_stages = torch.cat((output_stages, out.unsqueeze(0)), dim=0)  # concatenate stage output to others

        return output_stages


class MultiScaleTCN(nn.Module):
    # Forwards input through a 'branch' (TCN with sampling_rate = scale) for each scale in scales list
    # outputs 'average_output': average of pre-class outputs of branches passed through an extra classification layer
    # and 'branches_outputs': outputs of all branches as a list
    def __init__(self, scales, num_layers, num_filters, feature_len, num_classes):
        super().__init__()
        self.scales = scales
        self.branches = nn.ModuleList([
            SSTCN(num_layers, num_filters, feature_len, num_classes, sample_rate, add_pre_class_out=True)
            for sample_rate in scales
        ])
        self.conv_1x1 = nn.Conv1d(in_channels=num_filters, out_channels=num_classes, kernel_size=1)  # own class. layer

    def forward(self, x, mask):
        branches_outputs = [branch(x, mask) for branch in self.branches]

        # construct average output: add all outputs consecutively and divide by number of branches
        average_output = branches_outputs[0]['pre_class_out']
        for i, out in enumerate(branches_outputs[1:]):
            average_output = torch.add(average_output, branches_outputs[i]['pre_class_out'])
        average_output = torch.multiply(average_output, 1 / len(self.scales))

        average_output = self.conv_1x1(average_output)  # pass average output through own classification head
        branches_outputs = [out['out'] for out in branches_outputs]

        return {'average_output': average_output, 'branches_outputs': branches_outputs}


class TrainerTask1:
    # Trainer class for Task 1: single stage
    def __init__(self, num_layers, num_filters, feature_len, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-10)
        self.num_classes = num_classes
        self.model = SSTCN(num_layers, num_filters, feature_len, num_classes)

    def train(self, dataloader, num_epochs, lr,  save_dir, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            correct = 0
            total = 0
            for i_batch, batch in enumerate(dataloader):
                features = batch['features'].to(device)
                gt = batch['gt'].to(device)
                mask = batch['mask'].to(device)
                optimizer.zero_grad()
                prediction = self.model(features,  mask)

                # loss: cross-entropy
                loss = self.ce(prediction.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                gt = gt.detach()
                mask = mask.detach()
                prediction = prediction.detach()

                # leave only the most-scored class for each frame
                max_probabilities, predicted_classes = torch.max(prediction, 1)  # use class index, not probabilities
                correct += ((predicted_classes == gt) * mask[:, 0, :].squeeze()).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            print(f'epoch {epoch}: loss {epoch_loss}, acc {correct / total}')
            torch.save(self.model.state_dict(), join(save_dir, f'epoch{epoch}.model'))
            torch.save(optimizer.state_dict(), join(save_dir, f'epoch{epoch}.opt'))

    def predict(self, save_dir, epoch, results_dir, features_folder, video_list_path, action_class_to_num, device):
        self.model.eval()
        # revert the class dict to map class name to index
        action_num_to_class = {value: key for (key, value) in action_class_to_num.items()}
        with open(video_list_path, 'r') as f:
            video_list = [line.rstrip() for line in f.readlines()[:-1]]
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(join(save_dir, f'epoch{epoch}.model')))

            for video in video_list:
                features = np.load(join(features_folder, f'{video.split(".")[0]}.npy'))
                x = torch.tensor(features, dtype=torch.float).unsqueeze(0)
                x = x.to(device)
                x_mask = torch.ones(x.size())
                x_mask = x_mask.to(device)

                prediction = self.model(x, x_mask)
                prediction = prediction.detach().squeeze()

                predicted_probabilities, predicted_classes = torch.max(prediction, 0)
                predicted_classes = predicted_classes.tolist()
                predicted_labels = [action_num_to_class[c] for c in predicted_classes]
                out_path = join(results_dir, video)
                with open(out_path, 'w') as f:
                    f.writelines([f'{label}\n' for label in predicted_labels])


class TrainerTask2:
    # Trainer class for Task 2: multi stage
    def __init__(self, num_stages, num_layers, num_filters, feature_len, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-10)
        self.num_classes = num_classes
        self.model = MSTCN(num_stages, num_layers, num_filters, feature_len, num_classes)

    def train(self, dataloader, num_epochs, lr,  save_dir, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            correct = 0
            total = 0
            for i_batch, batch in enumerate(dataloader):
                features = batch['features'].to(device)
                gt = batch['gt'].to(device)
                mask = batch['mask'].to(device)
                optimizer.zero_grad()
                predictions = self.model(features,  mask)

                # loss: sum of cross-entropies between gt and each stage's prediction
                loss = 0
                for prediction in predictions:
                    loss += self.ce(prediction.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                gt = gt.detach()
                mask = mask.detach()
                predictions = predictions.detach()

                # leave only the most-scored class for each frame, use only the last stage's output
                max_probabilities, predicted_classes = torch.max(predictions[-1], 1)  # use class index
                correct += ((predicted_classes == gt) * mask[:, 0, :].squeeze()).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            print(f'epoch {epoch}: loss {epoch_loss}, acc {correct / total}')
            torch.save(self.model.state_dict(), join(save_dir, f'epoch{epoch}.model'))
            torch.save(optimizer.state_dict(), join(save_dir, f'epoch{epoch}.opt'))

    def predict(self, save_dir, epoch, results_dir, features_folder, video_list_path, action_class_to_num, device):
        self.model.eval()
        # revert the class dict to map class name to index
        action_num_to_class = {value: key for (key, value) in action_class_to_num.items()}
        with open(video_list_path, 'r') as f:
            video_list = [line.rstrip() for line in f.readlines()[:-1]]
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(join(save_dir, f'epoch{epoch}.model')))

            for video in video_list:
                features = np.load(join(features_folder, f'{video.split(".")[0]}.npy'))
                x = torch.tensor(features, dtype=torch.float).unsqueeze(0)
                x = x.to(device)
                x_mask = torch.ones(x.size())
                x_mask = x_mask.to(device)

                prediction = self.model(x, x_mask)[-1]  # use only the last stage's output
                prediction = prediction.detach().squeeze()

                predicted_probabilities, predicted_classes = torch.max(prediction, 0)
                predicted_classes = predicted_classes.tolist()
                predicted_labels = [action_num_to_class[c] for c in predicted_classes]
                out_path = join(results_dir, video)
                with open(out_path, 'w') as f:
                    f.writelines([f'{label}\n' for label in predicted_labels])


class TrainerTask3:
    # Trainer class for Task 3: multi stage with video-level loss
    def __init__(self, num_stages, num_layers, num_filters, feature_len, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-10)
        self.bce = nn.BCELoss()
        self.num_classes = num_classes
        self.model = MSTCN(num_stages, num_layers, num_filters, feature_len, num_classes)

    def train(self, dataloader, num_epochs, lr,  save_dir, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            correct = 0
            total = 0
            for i_batch, batch in enumerate(dataloader):
                features = batch['features'].to(device)
                gt = batch['gt'].to(device)
                mask = batch['mask'].to(device)
                gt_video = batch['gt_video'].to(device)  # 1-hot vector of classes present in gt

                optimizer.zero_grad()
                predictions = self.model(features,  mask)

                # loss: sum across all stages' outputs of sums of
                # 1) CE of output with gt
                # 2) BCE of 1-hot vector of classes present in gt with vector of the highest score for each class [0, 1]
                loss = 0
                for prediction in predictions:
                    # CE
                    loss += self.ce(prediction.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))
                    # BCE: MaxPool1d each class across all frames to get the highest probability that it reaches
                    # kernel_size = number of frames to get 1 number for each class
                    prediction_probabilities = F.softmax(prediction, dim=1)
                    max_pool = nn.MaxPool1d(kernel_size=prediction_probabilities.shape[2])
                    video_prediction = max_pool(prediction_probabilities)
                    loss += self.bce(video_prediction.squeeze().view(-1), gt_video.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                mask = mask.detach()
                gt = gt.detach()
                predictions = predictions.detach()

                # use only the last stage's output
                max_probabilities, predicted_classes = torch.max(predictions[-1], 1)  # use class index
                correct += ((predicted_classes == gt) * mask[:, 0, :].squeeze()).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            print(f'epoch {epoch}: loss {epoch_loss}, acc {correct / total}')
            torch.save(self.model.state_dict(), join(save_dir, f'epoch{epoch}.model'))
            torch.save(optimizer.state_dict(), join(save_dir, f'epoch{epoch}.opt'))

    def predict(self, save_dir, epoch, results_dir, features_folder, video_list_path, action_class_to_num, device):
        self.model.eval()
        action_num_to_class = {value: key for (key, value) in action_class_to_num.items()}
        with open(video_list_path, 'r') as f:
            video_list = [line.rstrip() for line in f.readlines()[:-1]]
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(join(save_dir, f'epoch{epoch}.model')))

            for video in video_list:
                features = np.load(join(features_folder, f'{video.split(".")[0]}.npy'))
                x = torch.tensor(features, dtype=torch.float).unsqueeze(0)
                x = x.to(device)
                x_mask = torch.ones(x.size())
                x_mask = x_mask.to(device)

                prediction = self.model(x, x_mask)[-1]  # use only the last stage's output
                prediction = prediction.detach().squeeze()

                predicted_probabilities, predicted_classes = torch.max(prediction, 0)
                predicted_classes = predicted_classes.tolist()
                predicted_labels = [action_num_to_class[c] for c in predicted_classes]
                out_path = join(results_dir, video)
                with open(out_path, 'w') as f:
                    f.writelines([f'{label}\n' for label in predicted_labels])


class TrainerTask4:
    # Trainer class for Task 4: Multi-scale TCN
    def __init__(self, scales, num_layers, num_filters, feature_len, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-10)
        self.num_classes = num_classes
        self.scales = scales
        self.model = MultiScaleTCN(scales, num_layers, num_filters, feature_len, num_classes)

    def train(self, dataloader, num_epochs, lr,  save_dir, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0
            correct = 0
            total = 0
            for i_batch, batch in enumerate(dataloader):
                features = batch['features'].to(device)
                gt = batch['gt'].to(device)
                mask = batch['mask'].to(device)

                optimizer.zero_grad()
                predictions = self.model(features,  mask)

                # loss: sum of CE of the final output with gt and CE of each branch output with gt
                loss = self.ce(predictions['average_output'].transpose(2, 1).contiguous().view(-1, self.num_classes),
                               gt.view(-1))
                for out in predictions['branches_outputs']:
                    loss += self.ce(out.transpose(2, 1).contiguous().view(-1, self.num_classes), gt.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                mask = mask.detach()
                gt = gt.detach()
                prediction = predictions['average_output'].detach()

                max_probabilities, predicted_classes = torch.max(prediction, 1)  # use class index
                correct += ((predicted_classes == gt) * mask[:, 0, :].squeeze()).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            print(f'epoch {epoch}: loss {epoch_loss}, acc {correct / total}')
            torch.save(self.model.state_dict(), join(save_dir, f'epoch{epoch}.model'))
            torch.save(optimizer.state_dict(), join(save_dir, f'epoch{epoch}.opt'))

    def predict(self, save_dir, epoch, results_dir, features_folder, video_list_path, action_class_to_num, device):
        self.model.eval()
        action_num_to_class = {value: key for (key, value) in action_class_to_num.items()}
        with open(video_list_path, 'r') as f:
            video_list = [line.rstrip() for line in f.readlines()[:-1]]
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(join(save_dir, f'epoch{epoch}.model')))

            for video in video_list:
                features = np.load(join(features_folder, f'{video.split(".")[0]}.npy'))
                x = torch.tensor(features, dtype=torch.float).unsqueeze(0)
                x = x.to(device)
                x_mask = torch.ones(x.size())
                x_mask = x_mask.to(device)

                predictions = self.model(x, x_mask)
                # use only the final (average) output to predict
                prediction = predictions['average_output'].detach().squeeze()

                predicted_probabilities, predicted_classes = torch.max(prediction, 0)
                predicted_classes = predicted_classes.tolist()
                predicted_labels = [action_num_to_class[c] for c in predicted_classes]
                out_path = join(results_dir, video)
                with open(out_path, 'w') as f:
                    f.writelines([f'{label}\n' for label in predicted_labels])
