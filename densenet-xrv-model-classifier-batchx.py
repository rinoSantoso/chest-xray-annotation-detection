#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import datasets, transforms, models # --> new
from torchmetrics.functional import accuracy
# from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader, random_split 
import torchxrayvision as xrv
from torchmetrics import ConfusionMatrix
import requests
from PIL import Image


# In[6]:


def tensor_to_imgnumpy(image: torch.Tensor, denormalize=False) -> np.ndarray:
    assert image.dim() == 3, f"expecting [3,256,256], the input size is {image.size()}" 
    
    imgnumpy = image.numpy().transpose(1,2,0)
    if denormalize:
        imgnumpy = imgnumpy*np.array((0.485, 0.456, 0.406)) + np.array((0.229, 0.224, 0.22))
    
    imgnumpy = imgnumpy.clip(0, 1)
    return imgnumpy


# In[1]:


def tensor_to_imgnumpy_simple(image):
    imgnumpy = image
    imgnumpy = imgnumpy.squeeze()
    return imgnumpy


# In[3]:


# In[7]:

class CustomNormalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
    """

    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return (2 * (img.astype(np.float32) / 255) - 1.) * 1024

#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class ToNumpy(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, img):
        return np.array(img)
    
class AddColorChannel(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, img):
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")
        return img[None, :, :]


# In[24]:


warning_log = {}

def fix_resolution(x, resolution: int, model: nn.Module):
        """Check resolution of input and resize to match requested."""

        # just skip it if upsample was removed somehow
        if not hasattr(model, 'upsample') or (model.upsample == None):
            return x

        if (x.shape[2] != resolution) | (x.shape[3] != resolution):
            if not hash(model) in warning_log:
                print("Warning: Input size ({}x{}) is not the native resolution ({}x{}) for this model. A resize will be performed but this could impact performance.".format(x.shape[2], x.shape[3], resolution, resolution))
                warning_log[hash(model)] = True
            return model.upsample(x)
        return x

def warn_normalization(x):
    """Check normalization of input and warn if possibly wrong. When 
    processing an image that may likely not have the correct 
    normalization we can issue a warning. But running min and max on 
    every image/batch is costly so we only do it on the first image/batch.
    """
    
    # Only run this check on the first image so we don't hurt performance.
    if not "norm_check" in warning_log:
        x_min = x.min()
        x_max = x.max()
        if torch.logical_or(-255 < x_min, x_max < 255) or torch.logical_or(x_min < -1024, 1024 < x_max):
            print(f'Warning: Input image does not appear to be normalized correctly. The input image has the range [{x_min:.2f},{x_max:.2f}] which doesn\'t seem to be in the [-1024,1024] range. This warning may be wrong though. Only the first image is tested and we are only using a heuristic in an attempt to save a user from using the wrong normalization.')
            warning_log["norm_correct"] = False
        else:
            warning_log["norm_correct"] = True
              
        warning_log["norm_check"] = True

        
test_inputs = []
test_targets = []
self_classifier_parameters = []
        
class FinetunedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # load pretrained model
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        
        self.features = model.features
        
        self.classifier = model.classifier
        
#         freeze the feature learning
        for param in self.features.parameters():
              param.requires_grad = False
        
        # change the number of output classes of the last layer
        # this is useless line as it the number of output classes is already set to be 10
        self.classifier = nn.Linear(
            in_features=self.classifier.in_features,
            out_features=2)
        
        # follow https://pytorch.org/hub/pytorch_vision_alexnet/
        tf_tonumpy = ToNumpy()
        tf_custom_normalize = CustomNormalize()
        tf_add_color_channel = AddColorChannel()
        tf_totensor = transforms.ToTensor()
        self.tf_compose = transforms.Compose([
            tf_tonumpy,
            tf_custom_normalize,
            tf_add_color_channel,
#             xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
#             tf_totensor
        ])
    
    def features2(self, x):
        x = fix_resolution(x, 224, self)
        warn_normalization(x)
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out
    
    def forward(self, x):
        x = fix_resolution(x, 224, self)
        
        features = self.features2(x)
        out = self.classifier(features)
        
        if hasattr(self, 'apply_sigmoid') and self.apply_sigmoid:
            out = torch.sigmoid(out)
        
        if hasattr(self,"op_threshs") and (self.op_threshs != None):
            out = torch.sigmoid(out)
            out = op_norm(out, self.op_threshs)
        return out

    
    def training_step(self, batch, batch_idx):
        # Copy paste from the previous article
        inputs, labels = batch
        
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs,labels) # --> NEW. Using nn.CrossEntropyLoss
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # This is new, but the structure is the same as training_step
        inputs, labels = batch
        
        outputs = self.forward(inputs)
#         import pdb; pdb.set_trace()
        loss = F.cross_entropy(outputs,labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy(preds, labels) # --> NEW
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # This is new, but the structure is the same as test_step
        # but I replace val_loss --> test_loss etc
        inputs, labels = batch
        
#         print(inputs.size())
#         print(inputs)
        
        split_inputs = torch.split(inputs, 1)
        
        for i in split_inputs:
#             print(i.size())
            test_inputs.append(i)
        
        for param in self.parameters():
            self_classifier_parameters.append(param.data)
#             print(param.data)
        
#         test_inputs.extend(inputs)
        test_targets.extend(labels.tolist())
        
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs,labels)
        
        preds = torch.argmax(outputs, dim=1)
        preds_split = torch.split(preds, 1)
#         print(preds_split[0].size())
        
        test_true_positive = 0
        test_false_positive = 0
        test_true_negative = 0
        test_false_negative = 0
        test_preds = []
        
        print(len(test_inputs))
        
        for img in test_inputs:
            try:
                test_pred = self.forward(img.cuda())
            except Exception as e:
                test_pred =  self.forward(img)

#             print(pred)
            test_preds.append(test_pred.argmax().item())
    
        for i in range(len(test_preds)):
            if test_preds[i] == 0:
                if test_targets[i] == 0:
                    test_true_positive+=1
                else:
                    test_false_positive+=1
            else:
                if test_targets[i] == 0:
                    test_false_negative+=1
                else:
                    test_true_negative+=1
        
        print("true positive: " + str(test_true_positive) + "\n" + "false positive: " + str(test_false_positive) + "\n" + "true negative: " + str(test_true_negative) + "\n"  + "false negative: " + str(test_false_negative))
        
        
#         confmat = ConfusionMatrix(num_classes=2)
#         print("Confusion Matrix: \nClean - Dirty")
#         print(confmat(preds, labels))
        acc = accuracy(preds, labels)
        
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        # split, transform, secretly move to GPU (if needed) by PL (not by us)
        if stage == 'fit' or stage is None:
            dataset_full = datasets.ImageFolder(root='./cars_buses/training/', transform=self.tf_compose)
            
            # split
            SIZE_TRAIN_DATA = int(len(dataset_full)*0.75)
            SIZE_VAL_DATA = len(dataset_full)-SIZE_TRAIN_DATA
            self.dataset_train, self.dataset_val = random_split(dataset_full, [SIZE_TRAIN_DATA,SIZE_VAL_DATA])
            
        if stage == 'test' or stage is None:
            self.dataset_test = datasets.ImageFolder(root='./cars_buses/test/', transform=self.tf_compose)
            
#         import pdb; pdb.set_trace()
            
    def train_dataloader(self): 
        return DataLoader(self.dataset_train, batch_size=50, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=50, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=50, num_workers=2)


# In[25]:


# pl.seed_everything(88) # --> for consistency, change the number with your favorite number :D

# model = FinetunedModel()

# # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
# try:
#     trainer = pl.Trainer(gpus=1,max_epochs=100,default_root_dir='./custom_logs')
# except Exception as e:
#     # most likely due to GPU, so fallback to non GPU
#     print(e)
#     trainer = pl.Trainer(max_epochs=100,default_root_dir='./custom_logs')

# trainer.fit(model)

# trainer.test()




pl.seed_everything(88)
# path = "./custom_logs/lightning_logs/version_11/checkpoints/epoch=99-step=2300.ckpt"
# model = FinetunedModel.load_from_checkpoint(checkpoint_path=path)

# trainer = pl.Trainer()
# trainer.test(model)

dataset_classes = ['Bus','Car']
    
loader = DataLoader(model.dataset_test, batch_size=1, shuffle=True)


targets = []
preds = []

true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0

for idx,(img,label) in enumerate(loader):
    targets.append(label.item())
    
#     print(img.size())
    
    try:
        pred = model.forward(img.cuda())
    except Exception as e:
        pred =  model.forward(img)
#         print(e)

    preds.append(pred.argmax().item())
    
    if pred.argmax().item() == 0:
        if label.item() == 0:
            true_positive+=1
        else:
            false_positive+=1
    else:
        if label.item() == 0:
            false_negative+=1
        else:
            true_negative+=1




# for img in test_inputs:
#     try:
#         pred = model.forward(img.cuda())
#     except Exception as e:
#         pred =  model.forward(img)
    
#     print(pred)
#     preds.append(pred.argmax().item())
    
from torchmetrics import AUC

targets_torch = torch.tensor(targets)
# targets_torch = torch.tensor(test_targets)
# targets = test_targets
preds_torch = torch.tensor(preds)


# print(preds)
# print(targets)

# confmat = ConfusionMatrix(num_classes=2)
# print("Confusion Matrix: \nClean - Dirty")
# print(confmat(preds_torch, targets_torch))

# true_positive = 0
# false_positive = 0
# true_negative = 0
# false_negative = 0
    
# for i in range(len(targets)):
#     if preds[i] == 0:
#         if targets[i] == 0:
#             true_positive+=1
#         else:
#             false_positive+=1
#     else:
#         if targets[i] == 0:
#             false_negative+=1
#         else:
#             true_negative+=1

print("true positive: " + str(true_positive) + "\n" + "false positive: " + str(false_positive) + "\n" + "true negative: " + str(true_negative) + "\n"  + "false negative: " + str(false_negative))

auc = AUC(reorder=True)
auc.update(preds_torch, targets_torch)
print("AUC score: ")
print(auc.compute())

model_classifier_parameters = []
for param in model.parameters():
    model_classifier_parameters.append(param.data)
#     print(param.data)

# for i in range(len(model_classifier_parameters)):
#     print(torch.equal(model_classifier_parameters[i], self_classifier_parameters[i]))


def imshow(imgnumpy: np.ndarray, label, denormalize=False):
    plt.imshow(tensor_to_imgnumpy_simple(imgnumpy))
    plt.title(dataset_classes[label])
    
loader = DataLoader(model.dataset_test, batch_size=1, shuffle=True)

plt.figure(figsize=(20, 8))
for idx,(img,label) in enumerate(loader):
    plt.subplot(4,10,idx+1)
    imshow(img[0],label,denormalize=True)
    
    
    # inference
    try:
        pred = model.forward(img.cuda())
    except Exception as e:
        pred =  model.forward(img)
#         print(e)
    
   
    
    title_dataset = dataset_classes[label]
    title_pred = dataset_classes[pred.argmax().item()]
    plt.title(f"{title_dataset}({title_pred})",color=("green" if title_dataset==title_pred else "red"))
    
    if idx == 40-1:
        break
        
plt.tight_layout()
plt.savefig('bar.png')


