import csv
import librosa
import numpy as np
from skimage.transform import resize
from PIL import Image
import os
import torch
import random
import torch.utils.data as torchdata
from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
from resnest.torch import resnest50_fast_1s1x64d as resnest50
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from sklearn.utils import shuffle


fft = 1024
hop = 256
sr = 32000
length = 10 * sr    # re les to 10ari na einai to kalo?
num_birds = 24
batch_size = 16
rng_seed = 1988
random.seed(rng_seed)
np.random.seed(rng_seed)
os.environ['PYTHONHASHSEED'] = str(rng_seed)
torch.manual_seed(rng_seed)
torch.cuda.manual_seed(rng_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class RainforestDataset(torchdata.Dataset):
    def __init__(self, filelist):
        self.specs = []
        self.labels = []
        for f in filelist:
            # Easier to pass species in filename at the start; worth changing later to more capable method
            label = int(str.split(f, '_')[1])
            label_array = np.zeros(num_birds, dtype=np.single)
            label_array[label] = 1.
            self.labels.append(label_array)
            
            # Open and save spectrogram to memory
            
            # If you use more spectrograms (add train_fp, for example), then they would not all fit to memory
            # In this case you should load them on the fly in __getitem__
            img = Image.open('mel_spectrums_flipped/' + f)
            stft_spec = np.array(img)
            img.close()
            
            # Transforming spectrogram from png to 0..1 array
            stft_spec = stft_spec / 255
            # Stacking for 3-channel image for resnet
            stft_spec = np.stack((stft_spec, stft_spec, stft_spec))
            
            self.specs.append(stft_spec)
    
    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, item):
        # Augment here if you want
        return self.specs[item], self.labels[item]


file_list = []
label_list = []

for f in os.listdir('mel_spectrums_flipped/'):
    if '.png' in f:
        file_list.append(f)
        label = str.split(f, '_')[1]
        label_list.append(label)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)

train_files = []
val_files = []

for fold_id, (train_index, val_index) in enumerate(skf.split(file_list, label_list)):
    # Picking only first fold to train/val on
    # This means loss of 20% training data
    # To avoid this, you can train 5 different models on 5 folds and average predictions
    if fold_id == 0:
        train_files = np.take(file_list, train_index)
        val_files = np.take(file_list, val_index)

print('Training on ' + str(len(train_files)) + ' examples')
print('Validating on ' + str(len(val_files)) + ' examples')



train_dataset = RainforestDataset(train_files)
val_dataset = RainforestDataset(val_files)

train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(train_dataset))
val_loader = torchdata.DataLoader(val_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(val_dataset))

# ResNeSt: Split-Attention Networks
# https://arxiv.org/abs/2004.08955
# Significantly outperforms standard Resnet
pretrained_weights = torch.load('resnest50_fast_1s1x64d-d8fbf808.pth')
model = resnest50(pretrained=False)
model.load_state_dict(pretrained_weights, strict=False)
# model = resnest50(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, num_birds)
)

# Picked for this notebook; pick new ones after major changes (such as adding train_fp to train data)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4)

# This loss function is not exactly suited for competition metric, which only cares about ranking of predictions
# Exploring different loss fuctions would be a good idea
pos_weights = torch.ones(num_birds)
pos_weights = pos_weights * num_birds
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

if torch.cuda.is_available():
    model = model.cuda()
    loss_function = loss_function.cuda()


best_corrects = 0

# Train loop
print('Starting training loop')
for e in range(0, 32):
    # Stats
    train_loss = []
    train_corr = []
    
    # Single epoch - train
    model.train()
    for batch, (data, target) in enumerate(train_loader):
        data = data.float()
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_function(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Stats
        vals, answers = torch.max(output, 1)
        vals, targets = torch.max(target, 1)
        corrects = 0
        for i in range(0, len(answers)):
            if answers[i] == targets[i]:
                corrects = corrects + 1
        train_corr.append(corrects)
        
        train_loss.append(loss.item())
    
    # Stats
    for g in optimizer.param_groups:
        lr = g['lr']
    print('Epoch ' + str(e) + ' training end. LR: ' + str(lr) + ', Loss: ' + str(sum(train_loss) / len(train_loss)) +
          ', Correct answers: ' + str(sum(train_corr)) + '/' + str(train_dataset.__len__()))
    
    # Single epoch - validation
    with torch.no_grad():
        # Stats
        val_loss = []
        val_corr = []
        
        model.eval()
        for batch, (data, target) in enumerate(val_loader):
            data = data.float()
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            
            output = model(data)
            loss = loss_function(output, target)
            
            # Stats
            vals, answers = torch.max(output, 1)
            vals, targets = torch.max(target, 1)
            corrects = 0
            for i in range(0, len(answers)):
                if answers[i] == targets[i]:
                    corrects = corrects + 1
            val_corr.append(corrects)
        
            val_loss.append(loss.item())
    
    # Stats
    print('Epoch ' + str(e) + ' validation end. LR: ' + str(lr) + ', Loss: ' + str(sum(val_loss) / len(val_loss)) +
          ', Correct answers: ' + str(sum(val_corr)) + '/' + str(val_dataset.__len__()))
    
    # If this epoch is better than previous on validation, save model
    # Validation loss is the more common metric, but in this case our loss is misaligned with competition metric, making accuracy a better metric
    if sum(val_corr) > best_corrects:
        print('Saving new best model at epoch ' + str(e) + ' (' + str(sum(val_corr)) + '/' + str(val_dataset.__len__()) + ')')
        torch.save(model, 'best_model_mel_pretrained_birdsong.pt')
        best_corrects = sum(val_corr)
        
    # Call every epoch
    scheduler.step()