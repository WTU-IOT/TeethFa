import os
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from .OS_CNN_Structure_build import generate_layer_parameter_list
from .log_manager import eval_condition, eval_model, save_to_log
from .OS_CNN import OS_CNN,MetaOS_CNN
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class OS_CNN_easy_use():
    
    def __init__(self,Result_log_folder, 
                 dataset_name, 
                 device, 
                 start_kernel_size = 1,
                 Max_kernel_size = 89, 
                 paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128], 
                 max_epoch = 2000, 
                 batch_size=16,
                 print_result_every_x_epoch = 1,
                 lr = 0.001
                ):
        
        super(OS_CNN_easy_use, self).__init__()
        
        if not os.path.exists(Result_log_folder +dataset_name+'/'):
            os.makedirs(Result_log_folder +dataset_name+'/')
        Initial_model_path = Result_log_folder +'/'+'initial_model'
        model_save_path = Result_log_folder +'/'+'Best_model'
        

        self.Result_log_folder = Result_log_folder
        self.dataset_name = dataset_name        
        self.model_save_path = model_save_path
        self.Initial_model_path = Initial_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.start_kernel_size = start_kernel_size
        self.Max_kernel_size = Max_kernel_size
        self.paramenter_number_of_layer_list = paramenter_number_of_layer_list
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.lr = lr
        self.OS_CNN = None

    def fit(self, X_train, y_train, X_val, y_val):

        print('code is running on ',self.device)
        
        
        # covert numpy to pytorch tensor and put into gpu
        X_train = torch.from_numpy(X_train)
        X_train.requires_grad = False
        X_train = X_train.to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        
        print(X_train.shape)

        X_test = torch.from_numpy(X_val)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        y_test = torch.from_numpy(y_val).to(self.device)
        
        
        # add channel dimension to time series data
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze_(1)
            X_test = X_test.unsqueeze_(1)

        input_shape = X_train.shape[-1]
        n_class = max(y_train) + 1
        receptive_field_shape= min(int(X_train.shape[-1]/4),self.Max_kernel_size)
        
        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size,
                                                             receptive_field_shape,
                                                             self.paramenter_number_of_layer_list,
                                                             in_channel = int(X_train.shape[1]))
                                                             #in_channel=1)
        

        print("layer_parameter_list",layer_parameter_list)
        torch_OS_CNN = MetaOS_CNN(layer_parameter_list, n_class.item()).to(self.device)

        torch.save(torch_OS_CNN.state_dict(), self.Initial_model_path)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_OS_CNN.parameters(),lr= self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),2), shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),2), shuffle=False)


        torch_OS_CNN.train()
        best_accuracy = 0.0

        for epoch in range(self.max_epoch):
            for i, sample in enumerate(train_loader):
                optimizer.zero_grad()
                y_predict = torch_OS_CNN(sample[0])
                sample[1] = sample[1].long()
                output = criterion(y_predict, sample[1])
                output.backward()
                optimizer.step()

            scheduler.step(output)
            print("best_acc",best_accuracy)
            if eval_condition(epoch, self.print_result_every_x_epoch):
                for param_group in optimizer.param_groups:
                    print('epoch =', epoch, 'lr = ', param_group['lr'])
                torch_OS_CNN.eval()
                acc_train = eval_model(torch_OS_CNN, train_loader)
                acc_test = eval_model(torch_OS_CNN, test_loader)
                torch_OS_CNN.train()
                print('train_acc=\t', acc_train, '\t test_acc=\t', acc_test, '\t loss=\t', output.item())
                sentence = 'train_acc=\t' + str(acc_train) + '\t test_acc=\t' + str(acc_test)
                print('log saved at:')
                save_to_log(sentence, self.Result_log_folder, self.dataset_name)

                if acc_test > best_accuracy:
                    best_accuracy = acc_test
                    print('Saving best model...')
                    torch.save(torch_OS_CNN.state_dict(), self.model_save_path)

        self.OS_CNN = torch_OS_CNN

        
    def predict(self, X_test):
        
        X_test = torch.from_numpy(X_test)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        
        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze_(1)
        
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_test.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        self.OS_CNN.eval()
        
        predict_list = np.array([])
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0])
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)
            
        return predict_list

    def load(self, X_train, y_train, X_val, y_val):


        X_train = torch.from_numpy(X_train)
        X_train.requires_grad = False
        X_train = X_train.to(self.device)


        X_test = torch.from_numpy(X_val)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        y_test = torch.from_numpy(y_val).to(self.device)



        n_class = max(y_train) + 1
        receptive_field_shape = min(int(X_train.shape[-1] / 4), self.Max_kernel_size)

        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size,
                                                             receptive_field_shape,
                                                             self.paramenter_number_of_layer_list,
                                                             in_channel=int(X_train.shape[1]))
        # in_channel=1)
        print("layer_parameter_list", layer_parameter_list)
        # 多支结构
        # torch_OS_CNN = OS_CNN(layer_parameter_list, n_class.item(),2, False).to(self.device)
        torch_OS_CNN = OS_CNN(layer_parameter_list, n_class.item(), False).to(self.device)


        checkpoint = torch.load(self.Result_log_folder + '/' + 'Best_model')
        self.OS_CNN.load_state_dict(checkpoint)

        X_val = X_val.astype(np.float32)

        y_predict = self.predict(X_val)

        print('correct:', y_val)
        print('predict:', y_predict)
        acc = accuracy_score(y_predict, y_val)
        print("acc", acc)
        precision=precision_score(y_val,y_predict,average='macro')
        print("precision ",precision)
        f1 = f1_score(y_val, y_predict, average='macro')
        print("f1", f1)
        recall = recall_score(y_val, y_predict, average='macro')
        print("recall", recall)

        classes = ['front', 'left_front', 'left', 'right_front', 'right']
        self.plot_confusion_matrix(y_predict,y_val,classes)


    def plot_confusion_matrix(self,predicted, true_labels, class_names, normalize=True):

        confusion = confusion_matrix(true_labels, predicted)

        if normalize:
            confusion = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix')
        plt.show()

        
        
        
        