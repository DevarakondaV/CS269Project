import pandas as pd
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
import random
from PIL import Image


torch.manual_seed(17)

target = [6]

MNIST = False
COLOR = False
FASHION = False
FACES = True
if MNIST:
    MNIST = torchvision.datasets.MNIST("data/", download=True)
    idx = [tar in target for tar in MNIST.targets]
    MNIST.targets = MNIST.targets[idx]
    MNIST.data = MNIST.data[idx]
    data_set = torch.Tensor(MNIST.data.numpy())
    save_path = "data/mnist_"
if FASHION:
    FMNIST = torchvision.datasets.FashionMNIST("data/", download = True)
    target = [0]
    idx = [tar in target for tar in FMNIST.targets]
    FMNIST.targets = FMNIST.targets[idx]
    FMNIST.data = FMNIST.data[idx]
    data_set = torch.Tensor(FMNIST.data.numpy())
    save_path = 'data/fmnist_'
if COLOR:
    # NOT WORKING
    CMNIST_DATA_DIR = "data/ColoredMNIST/train1.pt"
    CMNIST = torch.load(CMNIST_DATA_DIR)
    CMNIST = [item for item, label in CMNIST if label in target]
    data_set = torch.Tensor(CMNIST)
    print(data_set.shape)
    save_path = "data/cmnist_"
if FACES:
    DATA_DIR = "data/Faces/UTKFace/"
    save_path = "data/faces_"
    WHITE = "0"
    BLACK = "1"
    ASIAN = "2"
    OTHER = "4"
    # white - 0, 1 - black, 2 - asian, 3 - other
    files = os.listdir(DATA_DIR)
    randi = np.random.randint(100)
    files_white = [f for f in files if f.split("_")[2] == WHITE]
    files_black = [f for f in files if f.split("_")[2] == BLACK]
    files_asian = [f for f in files if f.split("_")[2] == ASIAN]
    files_white = [DATA_DIR + f for f in files_white]
    files_black = [DATA_DIR + f for f in files_black]
    files_asian = [DATA_DIR + f for f in files_asian]
    data_white = [np.expand_dims(np.asarray(Image.open(f).resize((28,28))),0) for f in files_white]
    data_black = [np.expand_dims(np.asarray(Image.open(f).resize((28,28))),0) for f in files_black]
    data_asian = [np.expand_dims(np.asarray(Image.open(f).resize((28,28))),0) for f in files_asian]
    
    data_white = np.concatenate(data_white)[:len(data_white) // 2]
    data_black = np.concatenate(data_black)
    data_asian = np.concatenate(data_asian)
    np.save("{}white".format(save_path), data_white)
    np.save("{}black".format(save_path), data_black)
    np.save("{}asian".format(save_path), data_asian)
    exit()

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super(AddGaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Transformations:
    def __init__(self):
        super(Transformations, self).__init__()
        self._translate1 = [0, 5]
        self._translate2 = [5, 0]

    def __call__(self, x):
        rand_int = np.random.randint(3)
        if rand_int == 0:
            return transforms.functional.affine(
                torch.Tensor(np.expand_dims(x, 0)),
                angle = 0,
                shear = 0,
                scale = 1,translate = self._translate1)
        if rand_int == 1:
            return transforms.functional.affine(
                torch.Tensor(np.expand_dims(x, 0)),
                angle = 0,
                shear = 0,
                scale = 1,translate = self._translate2)
        if rand_int == 2:
            x = 255 - torch.Tensor(np.expand_dims(x, 0))
            return x

# transforms = torch.nn.Sequential(
#     Transformations()
# )

# transforms = torch.nn.Sequential(
#     transforms.RandomApply(
#         torch.nn.ModuleList([
#             transforms.RandomAffine(0., translate = (0., 0.3)),
#             transforms.RandomRotation(5)
#         ])
#     )
# )



# scripted_transforms = torch.jit.script(transforms)
# output = scripted_transforms(data_set).numpy()
# print(output.shape)
transform = Transformations()
target = [str(i) for i in target]

np.save('{}before'.format(save_path) + '.'.join(target), data_set)
img1 = Image.fromarray(data_set.numpy()[0])
data_set = [transform(image) for image in data_set]
output = np.concatenate(data_set)

np.save('{}after'.format(save_path) + '.'.join(target), output)
# img1 = Image.fromarray(data_set.numpy()[10])
img2 = Image.fromarray(output[10])
img1.show()
img2.show()
exit()

MNIST_SIZE = 28

x_raw = np.array(raw_data)
x_raw = torch.FloatTensor(x_raw)


# create label.csv

label = x_raw.clone()
label = label.numpy()
x_df = pd.DataFrame(label[:, 0])
x_df.to_csv('label.csv')


# create mnist_raw.csv

x_before = x_raw.clone()
x_before = x_before.numpy()
x_df = pd.DataFrame(x_before[:, 1:])
x_df.to_csv('mnist_before.csv')


# create transfrom.csv
x_after = np.copy(x_before[:, 1:])
MNIST_SIZE = 28

# left Translation
curIndex = 0
batchSize = 6000
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j + randomShift < MNIST_SIZE - 1:
                x_after[index, pixel] = x_after[index, pixel + randomShift]
            else:
                x_after[index, pixel] = 0

curIndex += batchSize

# right Translation
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j < randomShift:
                x_after[index, pixel] = 0
            else:
                x_after[index, pixel] = x_after[index, pixel - randomShift]

curIndex += batchSize

# up Translation
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if i + randomShift < MNIST_SIZE - 1:
                newPixel = (i+randomShift)*MNIST_SIZE + j
                x_after[index, pixel] = x_after[index, newPixel]
            else:
                x_after[index, pixel] = 0

curIndex += batchSize

# down Translation
for index in range(curIndex, curIndex + batchSize):
    randomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if i < randomShift:
                x_after[index, pixel] = 0
            else:
                newPixel = (i-randomShift)*MNIST_SIZE + j
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# up left Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j + jRandomShift > MNIST_SIZE - 1 or i + iRandomShift > MNIST_SIZE - 1:
                x_after[index, pixel] = 0
            else:
                newPixel = (i+iRandomShift)*MNIST_SIZE + j + jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# down left Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j + jRandomShift > MNIST_SIZE - 1 or i < iRandomShift:
                x_after[index, pixel] = 0
            else:
                newPixel = (i-iRandomShift)*MNIST_SIZE + j + jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# up right Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j < jRandomShift or i + iRandomShift > MNIST_SIZE - 1:
                x_after[index, pixel] = 0
            else:
                newPixel = (i+iRandomShift)*MNIST_SIZE + j - jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# down right Translation
for index in range(curIndex, curIndex + batchSize):
    iRandomShift = random.randint(1, 7)
    jRandomShift = random.randint(1, 7)
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            if j < jRandomShift or i < iRandomShift:
                x_after[index, pixel] = 0
            else:
                newPixel = (i-iRandomShift)*MNIST_SIZE + j - jRandomShift
                x_after[index, pixel] = x_after[index, newPixel]

curIndex += batchSize

# noise added
mu, sigma = 0, 10  # mean and standard deviation
noise = np.random.normal(mu, sigma, 784)
for index in range(curIndex, curIndex + batchSize):
    for i in range(MNIST_SIZE * MNIST_SIZE):
        x_after[index, pixel] = x_after[index, pixel] + noise[i]

curIndex += batchSize

# invert contrast
MNIST_MAX = 255
for index in range(curIndex, curIndex + batchSize):
    for i in range(MNIST_SIZE):
        for j in range(MNIST_SIZE):
            pixel = i * MNIST_SIZE + j
            x_after[index, pixel] = -x_after[index, pixel] + MNIST_MAX

curIndex += batchSize


x_df = pd.DataFrame(x_after)
x_df.to_csv('mnist_after.csv')
