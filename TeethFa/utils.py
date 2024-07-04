import torch
import torch.nn.functional as F
from collections import OrderedDict
from torchmeta.modules import MetaModule
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

def compute_accuracy(logits, targets):
    """Compute the accuracy"""
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()

def compute_percision(logits, targets):
    probabilities = F.softmax(logits, dim=1)
    _, predictions = torch.max(probabilities, dim=1)
    predictions_cpu = predictions.cpu().numpy()
    targets_cpu = targets.cpu().numpy()


    precision=precision_score(targets_cpu,predictions_cpu,average='macro')
    f1=f1_score(targets_cpu,predictions_cpu,average='macro')
    recall=recall_score(targets_cpu,predictions_cpu,average='macro')

    return precision,f1,recall

def tensors_to_device(tensors, device=torch.device('cpu')):
    """Place a collection of tensors in a specific device"""
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, (list, tuple)):
        return type(tensors)(tensors_to_device(tensor, device=device)
            for tensor in tensors)
    elif isinstance(tensors, (dict, OrderedDict)):
        return type(tensors)([(name, tensors_to_device(tensor, device=device))
            for (name, tensor) in tensors.items()])
    else:
        raise NotImplementedError()

class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """
    def __call__(self, array):
        return torch.from_numpy(array.astype('float32'))

    def __repr__(self):
        return self.__class__.__name__ + '()'
