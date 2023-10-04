import numpy as np
import pandas as pd
import argparse
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import nibabel as nib
from data_split_3D_mlp import split_data
import time
import pickle
import sklearn.metrics

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, default="data/", help="data folder mounting point")
parser.add_argument("--model-type", type=str, default="mlp", help="Enter 3d or mlp")
parser.add_argument("--label-folder", type=str, default="label/", help="label folder mounting point")
parser.add_argument("--label-subfolder", type=str, default="", help="label subfolder mounting point")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default 1)")
parser.add_argument("--azure", type=bool, default=False, help="Running on azure")
parser.add_argument("--number-of-components", type=int, default=100, help="Number of group-ICA components")
args = parser.parse_args()

directory = Path(args.data_folder)
directory_label = Path(args.label_folder) / args.label_subfolder if args.label_subfolder else Path(args.label_folder)
batch_size = args.batch_size
device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')
print('Arguments:', args)

if args.azure:
    from azureml.core import Run
    run = Run.get_context()

print('Loading the data...')
if args.model_type == "3d":
    model_final = torch.load('model3D_final.pth', map_location='cpu')
else:
    model_final = torch.load('output_weights/model_mlp_case 1_model_0_epoch_999_final.pth', map_location='cpu')

subjects_train, subjects_valid, subjects_test, flat_labels, class_weights, label_names_unique = split_data(
    directory, directory_label, args.num_components)

# Determine dataset type based on model type
if args.model_type == "3d":
    dataset = {'test': NiiDataset(subjects_test)}
else:
    dataset = {'test': NiiDataset_MLP(subjects_test)}

loader = {'test': DataLoader(dataset['test'], num_workers=0, batch_size=args.batch_size, shuffle=True)}

# Model selection based on model type
if args.model_type == "3d":
    model = models.video.r3d_18(pretrained=True)
    model.fc = nn.Linear(512, 58)
else:
    model = MultilayerPerceptron2()

model.load_state_dict(model_final['model'])
model = model.to(device)
model.eval()

# Test phase
startTime = time.time()
epoch_accuracy = 0
counter = 0
correct = 0
all_output = []
all_predicted = []
all_target = []
misclassifications = []

# Evaluate the model on test data
for index, subjects_batch in enumerate(loader['test']):
    data, target = subjects_batch[0].to(device), subjects_batch[1].to(device)
    output = model(data)
    _, predicted = F.softmax(output, dim=1).max(1)
    
    counter += data.size(0)
    correct += predicted.eq(target).sum().item()
    all_predicted.extend(predicted.cpu().detach().numpy())
    all_output.extend(output.cpu().detach().numpy())
    all_target.extend(target.cpu().detach().numpy())


epoch_accuracy = 100. * correct / counter
print(f'Test Accuracy: {epoch_accuracy:1.0f}%')
print('Time Taken in seconds:', time.time() - startTime)
