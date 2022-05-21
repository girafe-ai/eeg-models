import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler

from eeg_models.datasets.demons import DemonsP300Dataset
from eeg_models.eegnet import EegNet
from eeg_models.transforms1 import (
    ButterFilter,
    ChannellwiseScaler,
    Decimator,
    MarkersTransformer,
)


class EEGtraining(object):
    def __init__(self, sample_per_epoch) -> None:
        self.loss_fn = None
        self.optimizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # window size : number of samples per epochs in output of dataset before decimation
        self.sample_per_epoch = sample_per_epoch
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

    def EEGmodel(self, n_samples):
        model = EegNet(2, 8, n_samples)
        return model

    def set_loaders(
        self, decimation_factor, sampling_rate, filter, batch_size, validation_split
    ):
        order, highpass, lowpass = filter
        print(
            ">> Decimation factor : ",
            decimation_factor,
            "- order : ",
            order,
            "-highpass : ",
            highpass,
            " - lowpass : ",
            lowpass,
        )
        labels_mapping = {1: 1, 2: 0, 0: 0}
        eeg_pipe = make_pipeline(
            Decimator(decimation_factor),
            ButterFilter(sampling_rate // decimation_factor, order, highpass, lowpass),
            ChannellwiseScaler(StandardScaler()),
        )
        markers_pipe = MarkersTransformer(labels_mapping, decimation_factor)
        n_samplesdecimated = self.sample_per_epoch // decimation_factor
        # model = EegNet(2, 8, n_samplesdecimated)
        self.model = self.EEGmodel(n_samplesdecimated)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer.zero_grad()
        # criterion
        self.loss_fn = nn.CrossEntropyLoss()
        # collate_fn

        def collate_adapt(batch):
            input = torch.stack([batch[i][0] for i in range(len(batch))], dim=0).reshape(
                -1, batch[0][0].shape[1], batch[0][0].shape[2]
            )
            label = (
                torch.stack([batch[i][1] for i in range(len(batch))], dim=0)
                .reshape(-1, batch[0][1].shape[1])
                .reshape(-1)
                .to(dtype=torch.int64)
            )
            return input, label

        # available device :
        print(self.device)
        # training and validation dataset from dataset
        sample_per_epoch = self.sample_per_epoch
        my_dataset = DemonsP300Dataset(
            transform=eeg_pipe,
            target_transform=markers_pipe,
            sample_per_epoch=sample_per_epoch,
        )
        # initialization
        shuffle_dataset = True
        random_seed = 42
        # Creating data indices for training and validation splits:
        dataset_size = len(my_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        self.train_loader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=batch_size,
            collate_fn=collate_adapt,
            sampler=train_sampler,
        )
        self.val_loader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=batch_size,
            collate_fn=collate_adapt,
            sampler=valid_sampler,
        )

    def metrics(self, epoch, target, pred):
        accuracy_skl = accuracy_score(target, pred, normalize=True)
        precision_skl = precision_score(target, pred, average="binary")
        recall_skl = recall_score(target, pred, average="binary")
        f1_score_skl = f1_score(target, pred)
        score_fpr, score_tpr, _ = roc_curve(target, pred)
        auc_score = auc(score_fpr, score_tpr)
        score_roc_auc = roc_auc_score(target, pred)
        return (
            accuracy_skl,
            precision_skl,
            recall_skl,
            f1_score_skl,
            score_fpr,
            score_tpr,
            auc_score,
            score_roc_auc,
        )

    def train_val(self, n_epochs):
        train_losses = []
        val_losses = []
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        optimizer = self.optimizer
        # compute
        for epoch in range(n_epochs):
            mini_batch_losses = []
            indice = 0
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # model train mode
                self.model.train()
                # outputs and loss
                outputs = self.model(x_batch)
                loss = self.loss_fn(outputs, y_batch)
                print("epoch -", epoch, " - train -", indice, " : ", loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # store loss
                mini_batch_losses.append(loss.item())
                indice += 1
            loss_mean = np.mean(mini_batch_losses)
            train_losses.append(loss_mean)
            print("train - epoch : ", epoch, "epoch loss mean = ", loss_mean)
            target_list = []
            pred_list = []

            with torch.no_grad():
                mini_batch_losses = []
                indice = 0
                for x_batch, y_batch in self.val_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    # eval mode
                    self.model.eval()
                    # outputs and loss
                    outputs = self.model(x_batch)
                    loss = self.loss_fn(outputs, y_batch)
                    print("epoch -", epoch, " - val -", indice, " - ", loss)
                    # store loss
                    mini_batch_losses.append(loss.item())
                    # store target and predicted value to compute metrics
                    target_list.append(y_batch)
                    pred_list.append(torch.argmax(outputs, dim=1))
                    indice += 1
            loss_mean = np.mean(mini_batch_losses)
            val_losses.append(loss_mean)
            print("val - epoch : ", epoch, "epoch loss mean = ", loss_mean)
            # compute metrics
            target = torch.cat(target_list)
            pred = torch.cat(pred_list)
            target = target.cpu().numpy()
            pred = pred.cpu().numpy()
            (
                accuracy_skl,
                precision_skl,
                recall_skl,
                f1_score_skl,
                score_fpr,
                score_tpr,
                auc_score,
                score_roc_auc,
            ) = self.metrics(epoch, target, pred)
            # print metrics
            print("epoch : ", epoch, " - accuracy :", accuracy_skl)
            print("epoch : ", epoch, "- precision : ", precision_skl)
            print("epoch : ", epoch, "- recall :", recall_skl)
            print("epoch : ", epoch, "- f1 score :", f1_score_skl)
            print("epoch : ", epoch, "- ROC : fpr :", score_fpr)
            print("epoch : ", epoch, "- ROC - tpr :", score_tpr)
            print("epoch : ", epoch, "- auc_score :", auc_score)
            print("epoch : ", epoch, "- roc_auc :", score_roc_auc)
        # print list of loss on train and val data
        print("  * train losses  :", train_losses)
        print("  * val losses :", val_losses)

    def searchgrid(
        self,
        decimator_pipeline,
        filter_pipeline,
        batch_size,
        validation_split,
        n_epochs,
        sampling_rate,
    ):
        for _, decimation_factor in enumerate(decimator_pipeline):
            for _, (order, highpass, lowpass) in enumerate(filter_pipeline):
                filter = (order, highpass, lowpass)
                self.set_loaders(
                    decimation_factor, sampling_rate, filter, batch_size, validation_split
                )
                self.train_val(n_epochs)
