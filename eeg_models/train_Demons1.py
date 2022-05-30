import numpy as np
import torch

# import torch.functional as F
import torch.nn as nn
import torch.optim as optim

# import torchmetrics.functional as M
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

# from torch.utils.data import DataLoader, TensorDataset, random_split
# from torch.utils.data import DataLoader, TensorDataset
# from torch.utils.data.dataset import random_split
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
    def __init__(self, nn_parameters, sample_per_epoch) -> None:
        self.loss_fn = None
        self.optimizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # window size : number of samples per epochs in output of dataset before decimation
        self.nn_parameters = nn_parameters
        self.sample_per_epoch = sample_per_epoch
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

    def EEGmodel(self):
        # def EEGmodel(self, nn_parameters):
        """
        Model definition : nn_parameters
            n_classes: int,
            n_channels: int = 64,
            n_samples: int = 128,
            dropout_rate: float = 0.5,
            rate: int = 128,
            f1: int = 8,
            d: int = 2,
            f2: Optional[int] = None,

        """
        # nn_parameters = {'n_classes' : 2, 'n_channels' : 8, 'n_samples' : sample_per_epoch // decimation_factor, 'dropout_rate' : 0.5, \
        #     'rate' : 128, 'f1' : 8, 'd' : 2, 'f2' : None}

        # model = EegNet(2, 8, n_samples)

        model = EegNet(
            self.nn_parameters["n_classes"],
            self.nn_parameters["n_channels"],
            self.nn_parameters["n_samples"],
        )

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

        # n_samplesdecimated = self.sample_per_epoch // decimation_factor

        # model = EegNet(2, 8, n_samplesdecimated)
        # model =  EegNet(nn_parameters)

        self.model = self.EEGmodel()
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer.zero_grad()
        # criterion
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.CrossEntropyLoss(
            weight=(torch.FloatTensor([0.1, 0.9])).to(self.device)
        )
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

        # sample_per_epoch = self.sample_per_epoch
        # my_dataset = DemonsP300Dataset(transform=eeg_pipe, target_transform=markers_pipe, sample_per_epoch=sample_per_epoch)
        my_dataset = DemonsP300Dataset(
            transform=eeg_pipe,
            target_transform=markers_pipe,
            sample_per_epoch=self.sample_per_epoch,
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

    def metrics(self, target, pred):
        # target :   ndarray (nsamples, ), int64
        # pred :  ndarry (nsamples, 2), int32
        max_pred = np.argmax(pred, axis=1)

        # ACCURACY
        accuracy_skl = accuracy_score(target, max_pred, normalize=True)
        # PRECISION
        precision_skl = precision_score(target, max_pred, average="binary")
        # RECALL
        recall_skl = recall_score(target, max_pred, average="binary")
        # F1_SCORE
        f1_score_skl = f1_score(target, max_pred)
        # Compute ROC curve and ROC area for each class
        # AUC & ROC CURVE
        # 1st class
        score_fpr1, score_tpr1, _ = roc_curve(target, pred[:, 0].astype(np.float64))
        auc_score1 = auc(score_fpr1, score_tpr1)
        # 2nd class
        score_fpr2, score_tpr2, _ = roc_curve(target, pred[:, 1].astype(np.float64))
        auc_score2 = auc(score_fpr2, score_tpr2)
        # another estimate of roc_curve : y_score = probalility estimate of positive label (argmax)
        # pred_roc_curve = pred[:, np.argmax(pred, axis=1)]
        pred_roc_curve = np.array(
            [pred[j, np.argmax(pred, axis=1)[j]] for j in range(pred.shape[0])]
        ).reshape(-1)
        score_fpr3, score_tpr3, _ = roc_curve(target, pred_roc_curve.astype(np.float64))
        auc_score3 = auc(score_fpr3, score_tpr3)
        # SCORE ROC AUC
        score_roc_auc = roc_auc_score(target, pred_roc_curve.astype(np.float64))

        return (
            accuracy_skl,
            precision_skl,
            recall_skl,
            f1_score_skl,
            score_fpr1,
            score_tpr1,
            auc_score1,
            score_fpr2,
            score_tpr2,
            auc_score2,
            score_fpr3,
            score_tpr3,
            auc_score3,
            score_roc_auc,
        )

    def train_val(self, n_epochs):
        train_losses = []
        val_losses = []
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        optimizer = self.optimizer

        metric_results = []

        _proba = torch.nn.Softmax(dim=1)

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

                    # pred_list.append(torch.argmax(outputs, dim=1))
                    pred_list.append(_proba(outputs))

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
                score_fpr1,
                score_tpr1,
                auc_score1,
                score_fpr2,
                score_tpr2,
                auc_score2,
                score_fpr3,
                score_tpr3,
                auc_score3,
                score_roc_auc,
            ) = self.metrics(target, pred)

            # print metrics
            print("epoch : ", epoch, " - accuracy :", accuracy_skl)
            print("epoch : ", epoch, "- precision : ", precision_skl)
            print("epoch : ", epoch, "- recall :", recall_skl)
            print("epoch : ", epoch, "- f1 score :", f1_score_skl)
            # print ('epoch : ', epoch, "- ROC : fpr1 :", score_fpr1)
            # print('epoch : ', epoch, '- ROC - tpr1 :', score_tpr1)
            print("epoch : ", epoch, "- auc_score on label 0 :", auc_score1)
            # print ('epoch : ', epoch, "- ROC : fpr1 :", score_fpr2)
            # print('epoch : ', epoch, '- ROC - tpr1 :', score_tpr2)
            print("epoch : ", epoch, "- auc_score on label 1 :", auc_score2)
            # print ('epoch : ', epoch, "- ROC : fpr1 :", score_fpr3)
            # print('epoch : ', epoch, '- ROC - tpr1 :', score_tpr3)
            print("epoch : ", epoch, "- auc_score :", auc_score3)
            print("epoch : ", epoch, "- roc_auc :", score_roc_auc)

            metric_results.append(
                (
                    epoch,
                    accuracy_skl,
                    precision_skl,
                    recall_skl,
                    f1_score_skl,
                    auc_score1,
                    auc_score2,
                    auc_score3,
                    score_roc_auc,
                )
            )

        # print list of loss on train and val data
        print("  * train losses  :", train_losses)
        print("  * val losses :", val_losses)

        return metric_results, train_losses, val_losses


"""
    SEARCH GRID FOR PARAMETERS :
    - nn parameters
    - decimation
    - filter parameters
"""


def print_metric_results(metric_results):
    print(
        "  * epoch, accuracy, precision, recall, f1 score, fpr1, tpr1, auc1, fpr2, tpr2, auc2, fpr3, tpr3, auc3, roc_auc"
    )
    for result in metric_results:
        print(result)


def print_losses(train_losses, val_losses):
    print("  * train losses  :", train_losses)
    print("  * val losses :", val_losses)


def filter_decim_searchgrid(
    nn_parameters_pipeline,
    sample_per_epoch_pipeline,
    decimator_pipeline,
    filter_pipeline,
    batch_size,
    validation_split,
    n_epochs,
    sampling_rate,
):

    results = []
    iteration = 0

    for _, nn_parameters in enumerate(nn_parameters_pipeline):

        for _, sample_per_epoch in enumerate(sample_per_epoch_pipeline):

            for _, decimation_factor in enumerate(decimator_pipeline):

                nn_parameters["n_samples"] = sample_per_epoch // decimation_factor

                model_to_train = EEGtraining(nn_parameters, sample_per_epoch)

                for _, (order, highpass, lowpass) in enumerate(filter_pipeline):

                    print(iteration)
                    iteration += 1

                    print("nn_parameters :", nn_parameters, "\n")
                    print("sample_per_epoch :", sample_per_epoch, "\n")
                    print("decimation_factor :", decimation_factor, "\n")
                    print("order :", order, "\n")
                    print("highpass :", highpass, "\n")
                    print("lowpass :", lowpass, "\n")

                    filter = (order, highpass, lowpass)
                    model_to_train.set_loaders(
                        decimation_factor,
                        sampling_rate,
                        filter,
                        batch_size,
                        validation_split,
                    )
                    (
                        metrics_param_model,
                        train_losses,
                        val_losses,
                    ) = model_to_train.train_val(n_epochs)
                    print_metric_results(metrics_param_model)
                    print_losses(train_losses, val_losses)
                    print("\n")

                    results.append(
                        (
                            nn_parameters,
                            sample_per_epoch,
                            decimation_factor,
                            filter,
                            metrics_param_model,
                            train_losses,
                            val_losses,
                        )
                    )

    return results
