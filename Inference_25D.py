import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torchvision.models
import sklearn.metrics
from data_split_25D import split_data

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--data-folder", type=str, default="data/", help="data folder mounting point")
parser.add_argument("--label-folder", type=str, default="label/", help="label folder mounting point")
parser.add_argument("--label-subfolder", type=str, default="", help="label subfolder mounting point")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default 8)")
parser.add_argument("--azure", type=bool, default=False, help="Running on azure")
args = parser.parse_args()

directory = Path(args.data_folder)
directory_label = Path(args.label_folder) / args.label_subfolder if args.label_subfolder else Path(args.label_folder)

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')
print('Arguments:', args)

if args.azure:
    from azureml.core import Run
    run = Run.get_context()  # Run context for Azure Machine Learning

print('Loading the data...')
model_final = torch.load('./Trained_Models/model_final25D.pth/Trained_Models/model_final25D.pth', map_location='cpu')

subjects_train, subjects_valid, subjects_test, labels_train, labels_val, labels_test, _, _, label_names_unique = split_data(directory, directory_label)

dataset = {'test': RGBDataset(subjects_test, labels_test)}
loader = {'test': DataLoader(dataset['test'], num_workers=0, batch_size=args.batch_size, shuffle=True)}

# Initialize the model and load the state
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 58)  # Modify the output layer
model.load_state_dict(model_final['model'])
model = model.to(device)
model.eval()

# Testing phase
start_time = time.time()
correct = 0
all_predicted, all_output, all_target = [], [], []
misclassifications = []

for _, subjects_batch in enumerate(loader['test']):
    data, target = subjects_batch[0].to(device), subjects_batch[1].to(device)
    output = model(data)
    _, predicted = F.softmax(output, dim=1).max(1)
    correct += predicted.eq(target).sum().item()

    all_predicted.extend(predicted.cpu().numpy())
    all_output.extend(output.cpu().numpy())
    all_target.extend(target.cpu().numpy())

accuracy = 100. * correct / len(all_target)
print(f'Test Accuracy: {accuracy:.1f}%')
print('Time Taken in seconds:', time.time() - start_time)

