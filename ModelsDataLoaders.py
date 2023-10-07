import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
import copy

class CustomDataLoader(Dataset):
    def __init__(self, file_dir, standardizer_dir):
        self.file_dir = file_dir
        self.standardizer_dir = standardizer_dir
        self.data = self.get_data()  # key=(subject_id, video_id) value=(feature, valence/arousal)\
        self.keys = list(self.data.keys())  # put keys to rows
        self.standardizers = self.get_standardizers()
        # feature = (num_electrodes, num_windows, num_of_features=8)

        # HxWxC

    def __len__(self):
        return len(self.keys)
    #
    def __getitem__(self, i):
        item_key = self.keys[i]  # (subject id, video id)
        feature = self.data[item_key][0]
        label = np.array(self.data[item_key][1]).astype(np.float32)
        # standardize features
        feature = self.standardize_data(feature)
        # we will flatten across 0th and 2nd dimension (across num_electrodes and num_features)
        feature = np.reshape(feature, (feature.shape[1], -1)).astype(np.float32)
        # now feature is (num_window, 14 x 8 shape)
        return torch.tensor(feature), torch.tensor(label)

    def get_data(self):
        with open(self.file_dir, "rb") as input_file:
            return pickle.load(input_file)

    def get_standardizers(self):
        with open(self.standardizer_dir, "rb") as input_file:
            return pickle.load(input_file)  # returns standardizer dictionary where key is electrode channel

    def standardize_data(self, current_data):
        # current_data = (14, num_window, 8)
        standardized_data = np.zeros_like(current_data)
        for j in range(current_data.shape[0]):
            standardized_data[j] = self.standardizers[j].transform(current_data[j])
        return standardized_data

class OurModel(nn.Module):
    def __init__(self, num_electrodes=14, num_features=8, output_size=3, layer_dim=2,
                 bottleneck = 16):
        # call the super constructor
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device name:", torch.cuda.get_device_name())
        self.feature_size = num_features * num_electrodes
        self.bottleneck = bottleneck
        # feature shape = (num_windows, self.feature_size)
        self.layer_dim = layer_dim # I do not want to have stacked LSTMs
        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.bottleneck,
                            num_layers=self.layer_dim, batch_first=True)
        # Input shape: (batch_dim, seq_dim, feature_dim)
        # add a fully connected for regressing the continuous estimation
        self.fc1 = nn.Linear(self.bottleneck, self.bottleneck // 2)
        self.fc2 = nn.Linear(self.bottleneck // 2, self.bottleneck//4)
        self.fc = nn.Linear(self.bottleneck//4, output_size)

        self.dropout = nn.Dropout(0.25)



    def forward(self, x):
        # x is in the shape (batch_dim, seq_dim, feature_dim)
        # initialize the hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.bottleneck).requires_grad_().to(self.device)
        # initialize the cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.bottleneck).requires_grad_().to(self.device)
        # detach just rips the node apart from the network
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # print("out just after: ", out)
        # out should be of shape: (batch_dim, seq_dim, hidden_dim)
        # just want to have the last one in the sequence
        out = out[:, -1, :]  # this corresponds to the hidden vectors at last time steps for all samples
        hidden_state = torch.clone(out)
        # now feed it into the fully connected layer
        # print("out before: ", out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc(out)
        # print("out after: ", out)
        return out, hidden_state


def train(model, train_loader, val_loader, lr=0.001, weight_decay=1e-5,
          num_epochs=1):

    criterion = nn.MSELoss()
    # num_epochs = model.config['train']['num_epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0003, momentum=0.99)

    history = dict(train=[], val=[], grad_norm=[])
    best_model_wts = None
    best_loss = 100000000.0
    best_epoch = -1

    print("Training is started!!!")
    for epoch in range(1, num_epochs + 1):
        # TRAIN
        model.train()  # enable train mode
        train_losses = []

        for x, y in train_loader:
            # x = (1, num_windows, 112)  y = (1, 3) 3 comes from valence arousal dominance
            # zero grad optimizer
            optimizer.zero_grad()

            x = x.to(model.device)
            y = y.to(model.device)
            pred, hidden = model(x)

            loss = criterion(pred, y)
            loss.backward()

            # clipping_value = 1  # arbitrary value of your choosing
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
            optimizer.step()
            train_losses.append(loss.detach().item())

        # VALIDATE
        val_losses = []
        model = model.eval()

        with torch.no_grad():
            for (x, y) in val_loader:
                x = x.to(model.device)
                y = y.to(model.device)
                pred, hidden = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())

        # mean rmse loss
        train_loss = np.mean(np.sqrt(np.array(train_losses)))
        val_loss = np.mean(np.sqrt(np.array(val_losses)))
        # Note that step should be called after validate()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss <= best_loss:
            print("here")
            best_loss = val_loss
            best_model_wts = model
            best_epoch = epoch
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        print(f'Epoch {epoch}: train median loss {train_loss} val median loss {val_loss}')
    return best_model_wts.eval(), history, best_epoch


def predict(trained_model, data_loader, bottleneck_size, output_size=3):
    predictions = np.zeros((len(data_loader.dataset), output_size))
    embeddings = np.zeros((len(data_loader.dataset), bottleneck_size))
    total_loss = 0
    with torch.no_grad():
        trained_model = trained_model.eval()
        counter = 0
        for (x, y) in data_loader:
            x = x.to(trained_model.device)
            pred, hidden = trained_model(x)

            x = np.squeeze(x.cpu().numpy())
            y = np.squeeze(y.numpy())
            pred = np.squeeze(pred.cpu().numpy())
            hidden = np.squeeze(hidden.cpu().numpy())

            loss = mean_squared_error(y, pred, squared=False)
            predictions[counter: counter + x.shape[0]] = pred
            embeddings[counter: counter + x.shape[0]] = hidden
            counter += x.shape[0]

            total_loss += loss.item()

    return np.array(predictions), np.array(embeddings), total_loss/len(data_loader)