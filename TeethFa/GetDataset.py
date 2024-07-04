import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import random
from scipy.interpolate import interp1d

def load_data(file_path):
    data2 = pd.read_csv(file_path, header=None)
    data2 = np.array(data2)
    x = torch.from_numpy(data2).float().unsqueeze(0)
    return x


def getDataSetLinear(x):

    csv_folder_path = r''
    label_file_path = r''
    max_length = 100
    csv_file_names = os.listdir(csv_folder_path)
    labels = pd.read_csv(label_file_path)
    label_dict = dict(zip(labels['file'], labels['label']))
    data_list = []
    labels = []

    for file_name in csv_file_names:
        file_path = os.path.join(csv_folder_path, file_name)
        label = label_dict[file_name]
        data_list.append(load_data(file_path))
        labels.append(label)

    interpolated_data_list = []

    for data in data_list:
        n = data.shape[1]
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, max_length)
        f = interp1d(x_old, data, kind='linear', axis=1, fill_value="extrapolate")
        interpolated_data = f(x_new)
        interpolated_data_list.append(interpolated_data)

    data_array = np.array(interpolated_data_list)

    labels = np.array(labels)
    random_int = random.randint(0, 1000)
    data_array = np.squeeze(data_array, axis=1)
    data_array = data_array.transpose((0, 2, 1))
    train_data, test_data, train_labels, test_labels = train_test_split(data_array, labels, test_size=0.3,
                                                                        random_state=random_int)

    return train_data, test_data, train_labels, test_labels


def getDataSetLinearForMaml(x):
    csv_folder_path = '../mamltest/'+str(x)+'/segmented'
    label_file_path = '../mamltest/'+str(x)+'/labels.csv'

    max_length = 100
    csv_file_names = os.listdir(csv_folder_path)
    labels = pd.read_csv(label_file_path)

    label_dict = dict(zip(labels['file'], labels['label']))

    data_list = []
    labels = []
    for file_name in csv_file_names:
        file_path = os.path.join(csv_folder_path, file_name)
        label = label_dict[file_name]
        data_list.append(load_data(file_path))
        labels.append(label)

    interpolated_data_list = []

    for data in data_list:
        n = data.shape[1]
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, max_length)
        f = interp1d(x_old, data, kind='linear', axis=1, fill_value="extrapolate")
        interpolated_data = f(x_new)
        interpolated_data_list.append(interpolated_data)

    data_array = np.array(interpolated_data_list)

    labels = np.array(labels)

    random_int =random.randint(0, 1000)

    data_array = np.squeeze(data_array, axis=1)

    data_array = data_array.transpose((0, 2, 1))


    train_data, test_data, train_labels, test_labels = train_test_split(data_array, labels, test_size=0.3,
                                                                        random_state=random_int)

    return train_data, test_data, train_labels, test_labels




def getDataSetLinearForMamlTest(x,num_test):

    csv_folder_path = '../mamltest/'+str(x)+'/segmented'
    label_file_path = '../mamltest/'+str(x)+'/labels.csv'
    max_length = 100
    csv_file_names = os.listdir(csv_folder_path)
    labels = pd.read_csv(label_file_path)
    label_dict = dict(zip(labels['file'], labels['label']))
    data_list = []
    labels = []
    interpolated_data_list = []
    for file_name in csv_file_names:
        file_path = os.path.join(csv_folder_path, file_name)
        label = label_dict[file_name]
        data_list.append(load_data(file_path))
        labels.append(label)

    for data in data_list:
        n = data.shape[1]
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, max_length)
        f = interp1d(x_old, data, kind='linear', axis=1, fill_value="extrapolate")
        interpolated_data = f(x_new)
        interpolated_data_list.append(interpolated_data)

    data_array = np.array(interpolated_data_list)
    labels = np.array(labels)
    data_array = np.squeeze(data_array, axis=1)
    data_array = data_array.transpose((0, 2, 1))
    train_data, test_data, train_labels, test_labels = custom_train_test_split(data_array, labels, num_test)
    return train_data, test_data, train_labels, test_labels



def custom_train_test_split(data_array, labels, train_samples_per_label):
    unique_labels = np.unique(labels)
    train_data, train_labels, test_data, test_labels = [], [], [], []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        train_indices = np.random.choice(label_indices, size=train_samples_per_label, replace=False)
        test_indices = np.setdiff1d(label_indices, train_indices)

        train_data.append(data_array[train_indices])
        train_labels.extend([label] * train_samples_per_label)
        test_data.append(data_array[test_indices])
        test_labels.extend([label] * len(test_indices))
    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)
    return train_data, test_data, np.array(train_labels), np.array(test_labels)





