from Classifiers.OS_CNN.OS_CNN_easy_use import OS_CNN_easy_use
from sklearn.metrics import accuracy_score ,f1_score,recall_score,precision_score
import numpy as np
from GetDataset import getDataSetLinear


Result_log_folder = r''

train_data,test_data,train_labels,test_labels =getDataSetLinear("")


train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

model = OS_CNN_easy_use(
    Result_log_folder = Result_log_folder,
    dataset_name = "dataset_name",
    device = "cuda:0",
    max_epoch = 100,
    paramenter_number_of_layer_list = [8*128*1, 5*128*128,128*128*8]
    )

model.fit(train_data, train_labels, test_data, test_labels)

y_predict = model.predict(test_data)
acc = accuracy_score(test_labels, y_predict)
f1 = f1_score(test_labels, y_predict, average='macro')
recall=recall_score(test_labels, y_predict,average='macro')
precision=precision_score(test_labels, y_predict,average='macro')

model.load(train_data, train_labels, test_data, test_labels)

