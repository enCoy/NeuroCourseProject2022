from ModelsDataLoaders import CustomDataLoader, OurModel
import os
from torch.utils.data import DataLoader
from ModelsDataLoaders import train, predict
import torch
import random
import numpy as np

if __name__ == "__main__":
    np.random.seed(23)
    random.seed(23)
    torch.manual_seed(23)

    project_dir = r"C:\Users\Cem Okan\Dropbox (GaTech)\ECE 6790 PROJECT"
    feature_dir = os.path.join(project_dir, "Features")
    model_dir = os.path.join(project_dir, "Models")
    standardizer_dir = os.path.join(model_dir, "standardizer_dreamer.pickle")
    BOTTLENECK_SIZE = 32
    #initialize data loaders for train-val-test
    train_dataset = CustomDataLoader(os.path.join(feature_dir, 'train_dreamer.pickle'),
                                   standardizer_dir=standardizer_dir)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # shuffle is True for this one
    val_dataset = CustomDataLoader(os.path.join(feature_dir, 'val_dreamer.pickle'),
                                     standardizer_dir=standardizer_dir)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)  # shuffle is True for this one
    test_dataset = CustomDataLoader(os.path.join(feature_dir, 'test_dreamer.pickle'),
                                     standardizer_dir=standardizer_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)  # shuffle is True for this one

    model = OurModel(num_electrodes=14, num_features=8, output_size=3, layer_dim=2,
                 bottleneck = BOTTLENECK_SIZE)
    model = model.to(model.device)

    trained_model, _, best_epoch = train(model, train_loader, val_loader, lr=0.003, weight_decay=1e-5,
          num_epochs=11)
    print("best epoch: ", best_epoch)
    # save the model
    torch.save(trained_model.state_dict(), model_dir + f"/embedding_model.pth")
    # test it on test data
    preds, embeds, loss = predict(trained_model, test_loader, BOTTLENECK_SIZE, output_size=3)
    print("Test loss: ", loss)
    # test it on val data
    preds, embeds, loss = predict(trained_model, val_loader, BOTTLENECK_SIZE, output_size=3)
    print("Val loss: ", loss)
    # test it on train data
    preds, embeds, loss = predict(trained_model, train_loader, BOTTLENECK_SIZE, output_size=3)
    print("Train loss: ", loss)





