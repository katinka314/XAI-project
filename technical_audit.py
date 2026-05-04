from model_factory import model_dt, model_lr

import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader_vis = DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
test_correct = 0
for data, target in test_loader_vis:
    data = data.to(device)
    with torch.no_grad():
        output, features = model(data)


features_embedded_0 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(features[0].cpu())
features_embedded_1 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(features[1].cpu())
features_embedded_2 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(features[2].cpu())
features_embedded_3 = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(features[3].cpu())



plt.rcParams["figure.figsize"] = (20,5)

fig, ax = plt.subplots(1, 4)
ax[0].scatter(features_embedded_0[:,0], features_embedded_0[:,1], c=target)
ax[1].scatter(features_embedded_1[:,0], features_embedded_1[:,1], c=target)
ax[2].scatter(features_embedded_2[:,0], features_embedded_2[:,1], c=target)
im = ax[3].scatter(features_embedded_3[:,0], features_embedded_3[:,1], c=target)

plt.colorbar(im)
