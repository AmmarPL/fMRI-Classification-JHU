import numpy as np
from collections import Counter
import os, sys 
import csv
from sklearn.model_selection import train_test_split
import nibabel as nib
import glob
from sklearn.preprocessing import LabelEncoder
import itertools
import scipy.ndimage as nd
from pathlib import Path
# import torchio as tio
import torch
import pandas as pd

def split_data(data_dir, directory_label):
    """
    Load the dr_stage2_ic data and split on patient level.
    """

    nii_img = []
    for nii in sorted(data_dir.glob('dr_stage2_ic*nii.gz')):
        img_obj = nib.load(nii)
        image = img_obj.get_fdata()
        nii_img.append(image)

    print("Length of nii list is (should be 100 [clusters]): ",len(nii_img))
    print("Shape of each image is (last should be 176 [people]: ",nii_img[0].shape)

    #
    # Load in the Labels and conver the string labels to integers
    #
    
    label_df = pd.read_csv(directory_label)
    label_names = list(label_df['Cluster Name 100'])

    print(len(label_names), label_names)

    #
    # Get unique label names (but keep it ordered, therefore don't use "set")
    #
    lookup = set()  # a temporary lookup set
    label_names_unique = [x for x in label_names if x not in lookup and lookup.add(x) is None]
    print(label_names_unique)

    print(f'There are {len(label_names_unique)} unique labels')
    labels = [label_names_unique.index(x) for x in label_names]
    print(labels)

    class_weights = Counter(label_names)
    class_weights

    # Create a list repeating each of the 100 labels 84 times 
    #corresponding to the number of 3d volumes (aka subjects) for each component(label)

    flat_labels = [[x]*176 for x in labels]
    flat_labels = list(itertools.chain(*flat_labels))
    print(len(flat_labels), flat_labels[82:86])

    # 
    # Split the "patient" indices
    #
    all_indices = list(range(nii_img[0].shape[-1]))
    tt, test_indices = train_test_split(all_indices, test_size=0.2, shuffle = True, random_state=42) #Beijing
    train_indices, valid_indices = train_test_split(tt, test_size=0.1,shuffle = True, random_state= 42) #beijing
    # tt, test_indices = train_test_split(all_indices, test_size=nii_img[0].shape[-1]-2, shuffle = True, random_state=42) #Everyyting else
    train_indices, valid_indices = train_test_split(tt, test_size=1,shuffle = True, random_state= 42) #Everything else

    # subjects_train = create_subjects_list(nii_img, train_indices, labels)
    # subjects_valid = create_subjects_list(nii_img, valid_indices, labels)
    # subjects_test = create_subjects_list(nii_img, test_indices, labels)


    subjects_train, labels_train = create_subjects_dict(nii_img, train_indices, labels)
    subjects_valid, labels_val = create_subjects_dict(nii_img, valid_indices, labels)
    subjects_test, labels_test = create_subjects_dict(nii_img, test_indices, labels)
    # subjects_test,labels_test = create_subjects_dict(nii_img, list(range(nii_img[0].shape[-1])), labels)

    return subjects_train, subjects_valid, subjects_test,labels_train, labels_val, labels_test, flat_labels, class_weights, label_names_unique
    # return subjects_train, subjects_valid, subjects_test, flat_labels, class_weights, label_names_unique


def create_subjects_list(nii_img, patient_indices, labels):
    """

    """

    subjects = []
    for patient_index in patient_indices: 
        for cluster_index in range(100):
            subjects += [tio.Subject(
                volume = tio.ScalarImage(tensor=torch.Tensor(nii_img[cluster_index][:,:,:,patient_index][None,])),
                label = labels[cluster_index]
            )]

    return subjects


def create_subjects_dict(nii_img, patient_indices, labels):
    """

    """

    subjects = []
    list_of_labels = []
    for patient_index in patient_indices: 
        for cluster_index in range(100):
            subjects.append(nii_img[cluster_index][:,:,:,patient_index])
            list_of_labels.append(labels[cluster_index])


    return subjects, list_of_labels

