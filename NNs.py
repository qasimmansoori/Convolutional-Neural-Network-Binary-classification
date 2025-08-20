

# !pip install torchinfo
import os
import sys
from collections import Counter

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchinfo
import torchvision
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from tqdm.version import __version__ as tqdm__version__

# !gcloud storage cp gs://wqu-cv-course-datasets/project1.tar.gz . #First we download the data i am using worldquant university public dataset

# !tar --skip-old-files -xzf project1.tar.gz #Unzipping the downloaded file

train_path = os.path.join("data_p1","data_binary","train")
content = os.listdir(train_path)

dist_dict = {}
for con in content:
  dir = os.path.join(train_path,con)
  files = os.listdir(dir)
  num_files = len(files)
  dist_dict[con] = num_files
dist_dict

plt.bar(dist_dict.keys(),dist_dict.values())
plt.show()

show_image = os.path.join(train_path,"hog","ZJ000005.jpg")
show_img_h = Image.open(show_image)
show_img_h

show_image_b = Image.open("/content/data_p1/data_binary/train/blank/ZJ000013.jpg")
show_image_b

class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

transform = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])

    ]
)

dataset = datasets.ImageFolder(train_path, transform=transform)

g = torch.Generator()
g.manual_seed(42)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=g)

print(f"Length of training set: {len(train_dataset)}")
print(f"Length of validation set: {len(val_dataset)}")

def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    class_to_index = dataset.dataset.class_to_idx
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})
train_counts = class_counts(train_dataset)
train_counts

val_counts = class_counts(val_dataset)
val_counts

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True, generator=g)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle=False, generator=g)

data_iter = iter(train_loader)
images, labels = next(data_iter)
print(images.shape)
print(labels.shape)
labels

flatten = nn.Flatten()
tensor_flatten = flatten(images)

height = 224
width = 224

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*height*width, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
)
output_layer = nn.Linear(128,2)
model.append(output_layer)
print(model)

model.to("cuda")
model

summary(model, input_size=(32, 3, height, width))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, optimizer, loss_fn, data_loader, device="cuda"):
    # We'll report the loss function's average value at the end of the epoch.
    training_loss = 0.0

    # The train method simply sets the model in training mode. No training
    # has happened.
    model.train()

    # We iterate over all batches in the training set to complete one epoch
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        # Sets the gradients to zero. We need to do this every time.
        optimizer.zero_grad()

        # Unpack images (X) and labels (y) from the batch and add those
        # tensors to the specified device.
        inputs = inputs.to(device)
        targets = targets.to(device)

        # We make a forward pass through the network and obtain the logits.
        # With the logits, we can calculate our loss.
        output = model(inputs)
        loss = loss_fn(output, targets)

        # After calculating our loss, we calculate the numerical value of
        # the derivative of our loss function with respect to all the
        # trainable model weights. Once we have the gradients calculated,
        # we let the optimizer take a "step", in other words, update or
        # adjust the model weights.
        loss.backward()
        optimizer.step()

        # We increment the training loss for the current batch
        training_loss += loss.data.item() * inputs.size(0)

    # We calculate the training loss over the completed epoch
    return training_loss / len(data_loader.dataset)

loss_value = train_epoch(model, optimizer, loss_fn, train_loader, device="cuda")
print(f"The average loss during the training epoch was {loss_value:.2f}.")

loss_value = train_epoch(model, optimizer, loss_fn, train_loader, device="cuda")
print(f"The average loss during the training epoch was {loss_value:.2f}.")

def predict(model, data_loader, device="cuda"):
    # This tensor will store all of the predictions.
    all_probs = torch.tensor([]).to(device)

    # We set the model to evaluation mode. This mode is the opposite of
    # train mode we set in the train_epoch function.
    model.eval()

    # Since we're not training, we don't need any gradient calculations.
    # This tells PyTorch not to calculate any gradients, which speeds up
    # some calculations.
    with torch.no_grad():

        # Again, we iterate over the batches in the data loader and feed
        # them into the model for the forward pass.
        for inputs, targets in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)

            # The model produces the logits.  This softmax function turns the
            # logits into probabilities.  These probabilities are concatenated
            # into the `all_probs` tensor.
            probs = F.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs

probabilities_train = predict(model, train_loader, device="cuda")
print(probabilities_train.shape)

probabilities_val = predict(model, val_loader, device="cuda")
print(probabilities_val.shape)

print(probabilities_train[0])
probabilities_val[0]

predictions_train = torch.argmax(probabilities_train, dim=1)

print(f"Predictions shape: {predictions_train.shape}")
print(f"First 10 predictions: {predictions_train[:10]}")

predictions_val = torch.argmax(probabilities_val, dim=1)

print(f"Predictions shape: {predictions_val.shape}")
print(f"First 10 predictions: {predictions_val[:10]}")

device = "cuda"
targets_train = torch.cat([labels for _, labels in train_loader]).to(device)
is_correct_train = torch.eq(predictions_train, targets_train)
total_correct_train = torch.sum(is_correct_train).item()
accuracy_train = total_correct_train / len(train_loader.dataset)

print(f"Accuracy on the training data: {accuracy_train}")

targets_val = torch.cat([labels for _, labels in val_loader]).to(device)
is_correct_val = torch.eq(predictions_val, targets_val)
total_correct_val = torch.sum(is_correct_val).item()
accuracy_val = total_correct_val / len(val_loader.dataset)

print(f"Accuracy on the validation data: {accuracy_val}")

def score(model, data_loader, loss_fn, device="cuda"):
    # Initialize the total loss (cross entropy) and the number of correct
    # predictions. We'll increment these values as we loop through the
    # data.
    total_loss = 0
    total_correct = 0

    # We set the model to evaluation mode. This mode is the opposite of
    # train mode we set in the train_epoch function.
    model.eval()

    # Since we're not training, we don't need any gradient calculations.
    # This tells PyTorch not to calculate any gradients, which speeds up
    # some calculations.
    with torch.no_grad():
        # We iterate over the batches in the data loader and feed
        # them into the model for the forward pass.
        for inputs, targets in tqdm(data_loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)

            # Calculating the loss function for this batch
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            total_loss += loss.data.item() * inputs.size(0)

            # Calculating the correct predictions for this batch
            correct = torch.eq(torch.argmax(output, dim=1), targets)
            total_correct += torch.sum(correct).item()

    return total_loss / len(data_loader.dataset), total_correct / len(
        data_loader.dataset
    )

loss_train, accuracy_train = score(model, train_loader, loss_fn, device)
print(f"Training accuracy from score function: {accuracy_train}")

loss_val, accuracy_val = score(model, val_loader, loss_fn, device)
print(f"Validation accuracy from score function: {accuracy_val}")

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):

    for epoch in range(1, epochs + 1):
        # Run train_epoch once, and capture the training loss.
        training_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)

        # Score the model on the validation data.
        validation_loss, validation_accuracy = score(model, val_loader, loss_fn, device)

        print(
            f"Epoch: {epoch}, Training Loss: {training_loss:.2f}, "
            f"Validation Loss: {validation_loss:.2f}, Validation Accuracy: {validation_accuracy:.2f}"
        )

train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device=device)

cm = confusion_matrix(targets_val.cpu(), predictions_val.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["blank", "hog"])

disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical");

os.makedirs("model", exist_ok=True)

torch.save(model, os.path.join("model", "shallownet"))

