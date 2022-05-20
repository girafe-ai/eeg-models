from importlib import reload

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import transforms
from datasets import BrainInvadersDataset
from eegnet import EegNet


def main():
    """Prepare dataset and dataloaders"""
    raw_dataset = BrainInvadersDataset()

    sampling_rate = 512
    decimation_factor = 10
    final_rate = sampling_rate // decimation_factor
    epoch_duration = 0.9
    labels_mapping = {33285.0: 1, 33286.0: 0}
    reload(transforms)
    eeg_pipe = make_pipeline(
        transforms.Decimator(decimation_factor),
        transforms.ButterFilter(sampling_rate // decimation_factor, 4, 0.5, 20),
        transforms.ChannellwiseScaler(StandardScaler()),
    )
    markers_pipe = transforms.MarkersTransformer(labels_mapping, decimation_factor)

    for i in range(1, 1 + len(raw_dataset)):
        eeg_pipe.fit(raw_dataset.__getitem__(i)["eegs"])

    dataset = []
    epoch_count = int(epoch_duration * final_rate)

    for i in range(1, 1 + raw_dataset.__len__()):
        epochs = []
        labels = []
        filtered = eeg_pipe.transform(raw_dataset.__getitem__(i)["eegs"])  # seconds
        markups = markers_pipe.transform(raw_dataset.__getitem__(i)["markers"])
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
                    dataset[i][1][j],
                ]
            )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=2
    )
    return trainloader, testloader


def train(epochs, model, criterion, dataloader, optimizer) -> None:
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")


def test(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            eeg, labels = data
            outputs = model(eeg)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")


if __name__ == "__main__":
    trainloader, testloader = main()
    model = EegNet(2, 16, 45)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train(
        epochs=8,
        model=model,
        criterion=criterion,
        dataloader=trainloader,
        optimizer=optimizer,
    )

    model_path = "./eeg.pth"
    torch.save(model.state_dict(), model_path)

    model = EegNet(2, 16, 45)
    model.load_state_dict(torch.load(model_path))

    test(model, testloader)
