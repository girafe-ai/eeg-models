from importlib import reload

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

from eeg_models import transforms
from eeg_models.datasets.braininvaders import BrainInvadersDataset
from eeg_models.eegnet import EegNet
from eeg_models.transforms import (
    ButterFilter,
    ChannellwiseScaler,
    Decimator,
    MarkersTransformer,
)


class EegTraining(object):
    def __init__(self, sampling_rate) -> None:
        self.loss_fn = None
        self.optimizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # window size : number of samples per epochs in output of dataset before decimation
        self.sampling_rate = sampling_rate  # 512
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

    def eeg_model(self, n_samples):
        model = EegNet(2, 16, n_samples)  # n_samples=45
        return model

    def set_loaders(
        self,
        decimation_factor,
        epoch_duration,
        sampling_rate,
        filter,
        batch_size,
        validation_split,
    ):
        order, highpass, lowpass = filter
        print(
            ">> Decimation factor : ",
            decimation_factor,
            "- epoch_duration : ",
            epoch_duration,
            "- order : ",
            order,
            "-highpass : ",
            highpass,
            " - lowpass : ",
            lowpass,
        )
        final_rate = self.sampling_rate // decimation_factor
        labels_mapping = {33285.0: 1, 33286.0: 0}
        reload(transforms)
        eeg_pipe = make_pipeline(
            Decimator(decimation_factor),
            ButterFilter(sampling_rate // decimation_factor, order, highpass, lowpass),
            ChannellwiseScaler(StandardScaler()),
        )
        markers_pipe = MarkersTransformer(labels_mapping, decimation_factor)
        epoch_count = int(epoch_duration * final_rate)
        # model = EegNet(2, 16, epoch_count)
        self.model = self.eeg_model(epoch_count)
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer.zero_grad()
        # criterion
        self.loss_fn = nn.CrossEntropyLoss()

        # available device :
        print(self.device)
        raw_dataset = BrainInvadersDataset()
        for i in range(1, 1 + len(raw_dataset)):
            eeg_pipe.fit(raw_dataset[i]["eegs"])

        dataset = []
        for i in range(1, 1 + raw_dataset.__len__()):
            epochs = []
            labels = []
            filtered = eeg_pipe.transform(raw_dataset[i]["eegs"])  # seconds
            markups = markers_pipe.transform(raw_dataset[i]["markers"])
            for signal, markup in zip(filtered, markups):
                epochs.extend(
                    [signal[:, start : (start + epoch_count)] for start in markup[:, 0]]
                )
                labels.extend(markup[:, 1])
            dataset.append((np.array(epochs), np.array(labels)))

        full_dataset = []
        for i in range(len(dataset)):
            for j in range(len(dataset[i][0])):
                full_dataset.append(
                    [
                        torch.from_numpy(dataset[i][0][j]).to(dtype=torch.float32),
                        np.int64(dataset[i][1][j]),
                    ]
                )

        # initialization
        shuffle_dataset = True
        random_seed = 42
        # Creating data indices for training and validation splits:
        dataset_size = len(full_dataset)
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
            full_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
        )
        self.val_loader = torch.utils.data.DataLoader(
            full_dataset,
            batch_size=batch_size,
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


# from importlib import reload

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# import transforms
# from eeg_models.datasets.braininvaders import BrainInvadersDataset
# from eegnet import EegNet


# def main():
#     """Prepare dataset and dataloaders"""
#     raw_dataset = BrainInvadersDataset()

#     sampling_rate = 512
#     decimation_factor = 10
#     final_rate = sampling_rate // decimation_factor
#     epoch_duration = 0.9
#     labels_mapping = {33285.0: 1, 33286.0: 0}
#     reload(transforms)
#     eeg_pipe = make_pipeline(
#         transforms.Decimator(decimation_factor),
#         transforms.ButterFilter(sampling_rate // decimation_factor, 4, 0.5, 20),
#         transforms.ChannellwiseScaler(StandardScaler()),
#     )
#     markers_pipe = transforms.MarkersTransformer(labels_mapping, decimation_factor)

#     for i in range(1, 1 + len(raw_dataset)):
#         eeg_pipe.fit(raw_dataset[i]["eegs"])

#     dataset = []
#     epoch_count = int(epoch_duration * final_rate)

#     for i in range(1, 1 + len(raw_dataset)):
#         epochs = []
#         labels = []
#         filtered = eeg_pipe.transform(raw_dataset[i]["eegs"])  # seconds
#         markups = markers_pipe.transform(raw_dataset[i]["markers"])
#         for signal, markup in zip(filtered, markups):
#             epochs.extend(
#                 [signal[:, start : (start + epoch_count)] for start in markup[:, 0]]
#             )
#             labels.extend(markup[:, 1])
#         dataset.append((np.array(epochs), np.array(labels)))

#     print(type(dataset[0][1][0]))
#     print(dataset[0][1][0].dtype)

#     full_dataset = []
#     for i in range(len(dataset)):
#         for j in range(len(dataset[i][0])):
#             full_dataset.append(
#                 [
#                     torch.from_numpy(dataset[i][0][j]).to(dtype=torch.float32),
#                     np.int64(dataset[i][1][j])
#                 ]
#             )


#     train_size = int(0.8 * len(full_dataset))
#     test_size = len(full_dataset) - train_size
#     train_dataset, test_dataset = torch.utils.data.random_split(
#         full_dataset, [train_size, test_size]
#     )

#     trainloader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=4, shuffle=True, num_workers=2
#     )
#     testloader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=4, shuffle=False, num_workers=2
#     )
#     return trainloader, testloader


# def train(epochs, model, criterion, dataloader, optimizer) -> None:
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for i, data in enumerate(dataloader, 0):
#             inputs, labels = data

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             if i % 2000 == 1999:
#                 print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
#                 running_loss = 0.0

#     print("Finished Training")


# def test(model, dataloader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in dataloader:
#             eeg, labels = data
#             outputs = model(eeg)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")


# if __name__ == "__main__":
#     trainloader, testloader = main()
#     model = EegNet(2, 16, 45)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters())
#     train(
#         epochs=8,
#         model=model,
#         criterion=criterion,
#         dataloader=trainloader,
#         optimizer=optimizer,
#     )

#     model_path = "./eeg.pth"
#     torch.save(model.state_dict(), model_path)

#     model = EegNet(2, 16, 45)
#     model.load_state_dict(torch.load(model_path))

#     test(model, testloader)
