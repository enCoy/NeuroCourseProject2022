from ModelsDataLoaders import OurModel
import os
import pickle
import torch
from torch.utils.data import DataLoader
from ModelsDataLoaders import CustomDataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import umap
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

def draw_umap(data, labels, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=labels, s=100)
    plt.title(title, fontsize=18)

mode = 'classification'

models_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\ECE 6790 PROJECT\Models'
feature_dir = r'C:\Users\Cem Okan\Dropbox (GaTech)\ECE 6790 PROJECT\Features'
model_dir = os.path.join(models_dir, 'embedding_model.pth')
standardizer_dir = os.path.join(models_dir, 'standardizer_dreamer.pickle')
# seed_num = 666
seed_num = 666
np.random.seed(seed_num)
random.seed(seed_num)
torch.manual_seed(seed_num)
BOTTLENECK_SIZE = 32
model = OurModel(num_electrodes=14, num_features=8, output_size=3, layer_dim=2, bottleneck=BOTTLENECK_SIZE)
model = model.to(model.device)
model.eval()
model.load_state_dict(torch.load(model_dir))
standardizer = pickle.load(open(standardizer_dir, 'rb'))
dataset = CustomDataLoader(os.path.join(feature_dir, "train_seed.pickle"), standardizer_dir = standardizer_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
original_data = []
original_labels = []
latent_data = np.zeros((len(dataset), BOTTLENECK_SIZE))
labels = np.zeros(len(dataset))
with torch.no_grad():
    ctr = 0
    for x, y in loader:
        original_data += x[0,:].tolist()
        num_windows = x.shape[1]
        original_labels += num_windows*[y[0].tolist()]
        x = x.to(model.device)
        y = y.to(model.device)
        pred, hidden = model(x)
        pred = pred.cpu().numpy()
        hidden = hidden.cpu().numpy()
        latent_data[ctr,:] = hidden
#         v, a, d = y[0]
#         if v>2.5 and a>2.5:
#             labels[ctr] = 0
#         elif v<2.5 and a>2.5:
#             labels[ctr] = 1
#         elif v<2.5 and a<2.5:
#             labels[ctr] = 2
#         else:
#             labels[ctr] = 3
        labels[ctr] = y
        ctr+=1
original_data = np.array(original_data)
original_labels = np.array(original_labels)
classes =['neutral', 'sad', 'fear', 'happy']
if mode == 'visualizer':
    plt.close("all")
    # latent_data = StandardScaler().fit_transform(latent_data)

    draw_umap(latent_data, labels, n_components=3, title='n_components = 3')
    plt.show()
    # umap
    reducer = umap.UMAP(random_state=seed_num, metric='minkowski')
    embedding = reducer.fit_transform(latent_data)
    # classes = [0,1,2,3]
    plt.figure(figsize=(4.3, 3.3))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels, alpha=0.75, s=25)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the latent data')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.xlim((-10, 30))


    tsne = TSNE(n_components=2, perplexity=30, random_state=seed_num,
                method='exact', )
    embedding_tsne = tsne.fit_transform(latent_data)

    plt.figure()
    scatter = plt.scatter(
        embedding_tsne[:, 0],
        embedding_tsne[:, 1],
        c=labels)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('tSNE projection of the latent data')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.show()

if mode=='classification':
    train_dataset = CustomDataLoader(os.path.join(feature_dir, "train_seed.pickle"), standardizer_dir=standardizer_dir)
    val_dataset = CustomDataLoader(os.path.join(feature_dir, "val_seed.pickle"), standardizer_dir=standardizer_dir)
    test_dataset = CustomDataLoader(os.path.join(feature_dir, "test_seed.pickle"), standardizer_dir=standardizer_dir)

    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    latent_data = np.zeros((len(train_dataset), 32))
    labels = np.zeros(len(train_dataset))
    with torch.no_grad():
        for ctr, item in enumerate(loader):
            x, y = item
            x = x.to(model.device)
            y = y.to(model.device)
            pred, hidden = model(x)
            pred = pred.cpu().numpy()
            hidden = hidden.cpu().numpy()
            latent_data[ctr, :] = hidden
            labels[ctr] = y

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    latent_data_val = np.zeros((len(val_dataset), 32))
    labels_val = np.zeros(len(val_dataset))
    with torch.no_grad():
        for ctr, item in enumerate(val_loader):
            x, y = item
            x = x.to(model.device)
            y = y.to(model.device)
            pred, hidden = model(x)
            pred = pred.cpu().numpy()
            hidden = hidden.cpu().numpy()
            latent_data_val[ctr, :] = hidden
            labels_val[ctr] = y

    test_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    latent_data_test = np.zeros((len(test_dataset), 32))
    labels_test = np.zeros(len(test_dataset))
    with torch.no_grad():
        for ctr, item in enumerate(test_loader):
            x, y = item
            x = x.to(model.device)
            y = y.to(model.device)
            pred, hidden = model(x)
            pred = pred.cpu().numpy()
            hidden = hidden.cpu().numpy()
            latent_data_test[ctr, :] = hidden
            labels_test[ctr] = y

    best_model = None
    params = {'n_estimators': [100, 500, 1000], 'max_depth': [3, 5, 7, 9]}
    best_score = 0
    for n_est in params['n_estimators']:
        for depth in params['max_depth']:
            print(n_est, depth)
            clf = RandomForestClassifier(random_state=1, n_estimators=n_est, max_depth=depth)
            clf.fit(latent_data, labels)
            score = clf.score(latent_data_val, labels_val)
            print(score)
            if score > best_score:
                best_model = clf
                best_params = (n_est, depth)
                best_score = score
    print(best_params)

    print("val result: ", best_model.score(latent_data_val, labels_val))
    print("test result: ", best_model.score(latent_data_test, labels_test))
    test_predictions = best_model.predict(latent_data_test)
    f1 = f1_score(labels_test, test_predictions, average='macro')
    recall = recall_score(labels_test, test_predictions, average='macro')
    precision = precision_score(labels_test, test_predictions, average='macro')
    print("f1: ", f1)
    print("recall: ", recall)
    print("precision: ", precision)

    cm = confusion_matrix(labels_test, test_predictions, labels=best_model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = classes)
    disp.plot()
    plt.show()

