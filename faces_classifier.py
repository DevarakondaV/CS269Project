import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
torch_dtype = torch.float32

class FACESDataset(Dataset):
    def __init__(self):
        self.path_asian = "data/faces_asian.npy"
        self.path_white = "data/faces_white.npy"
        self.path_black = "data/faces_black.npy"
        self.data_asian = np.load(self.path_asian)
        data_white = np.load(self.path_white)
        data_black = np.load(self.path_black)
        self.data_other = np.concatenate((data_black, data_white))
        ones = np.ones(self.data_asian.shape[0])
        zero = np.zeros(self.data_other.shape[0])
        self.target = np.expand_dims(np.concatenate((ones,zero)), 1)
        self.data = np.concatenate((self.data_asian, self.data_other))

    def __getitem__(self, index):
        X = self.data[index]
        y = self.target[index]
        X = (X - X.min())/(X.max() - X.min())
        X = torch.Tensor(X).to(dtype=torch_dtype)
        y = torch.Tensor(y).to(dtype=torch_dtype)
        return (X,y)

    def __len__(self):
        return len(self.data)


class Classifier(torch.nn.Module):


    def __init__(self):
        super(Classifier, self).__init__()
        def blockConv(k, in_feat, out_feat, sigmoid = False):
          layers = [torch.nn.Conv2d(in_feat, out_feat, k, padding='same')]
          if sigmoid:
            layers.append(torch.nn.Sigmoid())
          else:
            layers.append(torch.nn.ELU(0.2))
          return layers

        # Architecture
        self.model = torch.nn.Sequential(
            *blockConv(3, 3, 16),
            *blockConv(3, 16, 16),
            *blockConv(3, 16, 32),
            torch.nn.AvgPool2d(2, 2),
            *blockConv(3, 32, 32),
            *blockConv(3, 32, 64),
            torch.nn.AvgPool2d(2, 2),
            *blockConv(3, 64, 64),
            *blockConv(3, 64, 64),
            torch.nn.AvgPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(576, 25),
            torch.nn.ELU(0.2),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

def train_system(epochs, model, optimizer, criterion, data_train, writer):
    model.train()
    # Model Loss
    model_loss = 0
    # Keep track of losses
    n_samples = 0

    for epoch in range(epochs):
        total_samples = 0
        epoch_loss = 0
        for itr, data in enumerate(data_train):
            X,y = data
            n_samples = X.shape[0]
            X = X.reshape(n_samples, 3, 28, 28)
            total_samples += n_samples
            y_pred = model(X)
            loss = criterion(y_pred, y)
            print("Batch Loss: ", itr, loss)
            epoch_loss += loss.item() * n_samples
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_loss = epoch_loss / total_samples
        # Logging
        print("epoch [{}] loss {:.4f}".format(
            epoch + 1, model_loss))
        writer.add_scalar('loss', model_loss, epoch + 1)



dataset_train = FACESDataset()
data_train = DataLoader(
    dataset_train, batch_size = 32, shuffle = True)
loss_fn = torch.nn.BCELoss()
model = Classifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1E-3)
writer = SummaryWriter(log_dir = "logsFaces")
train_system(100, model, optimizer, loss_fn, data_train, writer)
checkpt_dir = "checkpointsFaces"
torch.save(model.state_dict(), checkpt_dir + "/faces.pth")



