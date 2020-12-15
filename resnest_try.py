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
from resnest.torch import resnest50
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
from torchaudio.transforms import FrequencyMasking

# 179/244 1-fold val --> me fft = 1024, hop = 256 --> LB 0.804
# 184/244 1-fold val --> me fft = 2048, hop = 512 --> LB 0.781

fft = 1024
hop = 256
# Less rounding errors this way
sr = 48000
length = 10 * sr

with open('train_tp.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

# Check minimum/maximum frequencies for bird calls
# Not neccesary, but there are usually plenty of noise in low frequencies, and removing it helps
fmin = 24000
fmax = 0

# Skip header row (recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max) and start from 1 instead of 0
for i in range(1, len(data)):
    if fmin > float(data[i][4]):
        fmin = float(data[i][4])
    if fmax < float(data[i][6]):
        fmax = float(data[i][6])
# Get some safety margin
fmin = int(fmin * 0.9)
fmax = int(fmax * 1.1)
print('Minimum frequency: ' + str(fmin) + ', maximum frequency: ' + str(fmax))


# print('Starting spectrogram generation')
# for i in range(1, len(data)):
#     # All sound files are 48000 bitrate, no need to slowly resample
#     wav, sr = librosa.load('train/' + data[i][0] + '.flac', sr=None)
    
#     t_min = float(data[i][3]) * sr
#     t_max = float(data[i][5]) * sr
    
#     # Positioning sound slice
#     center = np.round((t_min + t_max) / 2)
#     beginning = center - length / 2
#     if beginning < 0:
#         beginning = 0
    
#     ending = beginning + length
#     if ending > len(wav):
#         ending = len(wav)
#         beginning = ending - length
        
#     slice = wav[int(beginning):int(ending)]
    
#     # Mel spectrogram generation
#     # Default settings were bad, parameters are adjusted to generate somewhat reasonable quality images
#     # The better your images are, the better your neural net would perform
#     # You can also use librosa.stft + librosa.amplitude_to_db instead
#     mel_spec = librosa.feature.melspectrogram(slice, n_mels=128, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
#     mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
#     mel_spec = resize(mel_spec, (224, 400))
    
#     # Normalize to 0...1 - this is what goes into neural net
#     mel_spec = mel_spec - np.min(mel_spec)
#     mel_spec = mel_spec / np.max(mel_spec)

#     # And this 0...255 is for the saving in png format
#     mel_spec = mel_spec * 255
#     mel_spec = np.round(mel_spec)    
#     mel_spec = mel_spec.astype('uint8')
#     mel_spec = np.asarray(mel_spec)
    
#     png = Image.fromarray(mel_spec, 'L')
#     png.save('mel_spectrums/' + data[i][0] + '_' + data[i][1] + '_' + str(center) + '.png')
    
#     if i % 100 == 0:
#         print('Processed ' + str(i) + ' train examples from ' + str(len(data)))


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
    def __init__(self, specs, labels, transform=None):
        self.specs = specs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, item):

        image = Image.open(self.specs[item])
        image = self.transform(image)
        image = torch.cat(3*[image])

        label = self.labels[item]
        label = torch.from_numpy(np.array(label))

        return image, label

x = []
y = []
data_path = 'stft_spectrums/'

for path, subdirs, files in os.walk(os.path.join(data_path)):
        for name in files:
            # x.append(os.path.join('./data/train/', name.split('_')[0] + '.flac'))
            x.append(os.path.join(path, name))
            labels = name.split('.png')[0].split('_')[1:2]
            # labels = name.split('.npy')[0].split('_')[1:]
            for l in range(len(labels)):
                labels[l] = int(labels[l])
            y.append(labels)

x, y = shuffle(x, y, random_state=rng_seed)
# x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.20, random_state=rng_seed)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
for fold_id, (train_index, val_index) in enumerate(skf.split(x, y)):
    if fold_id == 0:
        x_train, x_valid = np.take(x, train_index), np.take(x, val_index)
        y_train, y_valid = np.take(y, train_index), np.take(y, val_index)


def extractLabels(lst): 
    return list(map(lambda el:[el], lst)) 

y_train_new = extractLabels(y_train)
y_valid_new = extractLabels(y_valid)

# needs after making even train and valid
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(y_train_new)
y_valid = mlb.fit_transform(y_valid_new)

print('Training on ' + str(len(x_train)) + ' examples')
print('Validating on ' + str(len(x_valid)) + ' examples')


# transformations = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor()])
# train_dataset = RainforestDataset(train_files, transformations)
# val_dataset = RainforestDataset(val_files)

# train_loader = torchdata.DataLoader(train_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(train_dataset))
# val_loader = torchdata.DataLoader(val_dataset, batch_size=batch_size, sampler=torchdata.RandomSampler(val_dataset))

def train_dataloader(
    x_train=None,
    y_train=None,
    batch_size=None,
    num_workers=None
):
    train_loader = DataLoader(
        RainforestDataset(
            x_train,
            y_train,
            transform=transforms.Compose(
                [
                    # torchaudio.transforms.MelSpectrogram(n_mels=128, 
                    #                          n_fft=2048, 
                    #                          sample_rate=16000, 
                    #                          hop_length=512), 
                    # torchaudio.transforms.AmplitudeToDB(),
                    # transforms.Resize((input_shape[1], input_shape[2]), interpolation=2),
                    # transforms.Resize((64, 2004), interpolation=2),
                    # transforms.Resize((224, 224), interpolation=2),
                    transforms.RandomCrop([200, 400]),
                    transforms.ToTensor(),
                    # tensor_random_crop(input_shape=input_shape),
                    # tensor_random_roll(input_shape=input_shape),
                    # FrequencyMasking(freq_mask_param=10),
                    # TimeMask(max_width=656, use_mean=False)
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #              std=[0.229, 0.224, 0.225])
                    # transforms.Lambda(lambda img: img * 2.0 - 1.0),  # effi advprop
                ]
            ),
            # mode='train'
        ),
        pin_memory=False,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader


def valid_dataloader(
    x_valid=None,
    y_valid=None,
    batch_size=None,
    num_workers=None):
    valid_loader = DataLoader(
        RainforestDataset(
            x_valid,
            y_valid,
            transform=transforms.Compose(
                [
                    # torchaudio.transforms.MelSpectrogram(n_mels=128, 
                    #                          n_fft=2048, 
                    #                          sample_rate=16000, 
                    #                          hop_length=512), 
                    # torchaudio.transforms.AmplitudeToDB(),
                    # transforms.Resize((input_shape[1], input_shape[2]), interpolation=2),
                    # transforms.Resize((64, 2004), interpolation=2),
                    # transforms.Resize((224, 2400), interpolation=2), # 2400 coz we need 400 per segment resize
                    # torchvision.transforms.RandomVerticalFlip(p=1.0),                    
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #              std=[0.229, 0.224, 0.225])
                    # transforms.Lambda(lambda img: img * 2.0 - 1.0),  # effi advprop
                ]
            ),
            # mode='valid'
        ),
        pin_memory=False,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return valid_loader

num_workers = 0
train_loader = train_dataloader(
        x_train=x_train,
        y_train=y_train,
        batch_size=batch_size,
        num_workers=num_workers
    )
valid_loader = valid_dataloader(
        x_valid=x_valid,
        y_valid=y_valid,
        batch_size=batch_size, # need 1 for now
        num_workers=num_workers
    )

# ResNeSt: Split-Attention Networks
# https://arxiv.org/abs/2004.08955
# Significantly outperforms standard Resnet
model = resnest50(pretrained=True)

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
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
            device = 'cuda'
            data, target = data.to(device), target.to(device, dtype=torch.float32)
            
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
          ', Correct answers: ' + str(sum(train_corr)) + '/' + str(x_train.__len__()))
    
    # Single epoch - validation
    with torch.no_grad():
        # Stats
        val_loss = []
        val_corr = []
        
        model.eval()
        for batch, (data, target) in enumerate(valid_loader):
            data = data.float()
            if torch.cuda.is_available():
                device = 'cuda'
                data, target = data.to(device), target.to(device, dtype=torch.float32)
            
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
          ', Correct answers: ' + str(sum(val_corr)) + '/' + str(x_valid.__len__()))
    
    # If this epoch is better than previous on validation, save model
    # Validation loss is the more common metric, but in this case our loss is misaligned with competition metric, making accuracy a better metric
    if sum(val_corr) > best_corrects:
        print('Saving new best model at epoch ' + str(e) + ' (' + str(sum(val_corr)) + '/' + str(x_valid.__len__()) + ')')
        torch.save(model, 'best_model_stft_2.pt')
        best_corrects = sum(val_corr)
        
    # Call every epoch
    scheduler.step()

# Free memory
del model


def load_test_file(f):
    wav, sr = librosa.load('test/' + f, sr=None)

    # Split for enough segments to not miss anything
    segments = len(wav) / length
    segments = int(np.ceil(segments))
    
    mel_array = []
    
    for i in range(0, segments):
        # Last segment going from the end
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]
        
        # Same mel spectrogram as before
        mel_spec = librosa.stft(slice, n_fft=fft, hop_length=hop, window='hann')
        mel_spec = librosa.amplitude_to_db(np.abs(mel_spec))
        mel_spec = resize(mel_spec, (224, 400))
    
        mel_spec = mel_spec - np.min(mel_spec)
        mel_spec = mel_spec / np.max(mel_spec)
        
        mel_spec = np.stack((mel_spec, mel_spec, mel_spec))

        mel_array.append(mel_spec)
    
    return mel_array


# Loading model back
model = resnest50(pretrained=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(1024, num_birds)
)

model = torch.load('best_model_stft_2.pt')
model.eval()

if torch.cuda.is_available():
    model.cuda()
    
# Prediction loop
print('Starting prediction loop')
with open('submission.csv', 'w', newline='') as csvfile:
    submission_writer = csv.writer(csvfile, delimiter=',')
    submission_writer.writerow(['recording_id','s0','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11',
                               's12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23'])
    
    test_files = os.listdir('test/')
    print(len(test_files))
    
    # Every test file is split on several chunks and prediction is made for each chunk
    for i in range(0, len(test_files)):
        data = load_test_file(test_files[i])
        data = torch.tensor(data)
        data = data.float()
        if torch.cuda.is_available():
            data = data.cuda()

        output = model(data)

        # Taking max prediction from all slices per bird species
        # Usually you want Sigmoid layer here to convert output to probabilities
        # In this competition only relative ranking matters, and not the exact value of prediction, so we can use it directly
        maxed_output = torch.max(output, dim=0)[0]
        maxed_output = maxed_output.cpu().detach()
        
        file_id = str.split(test_files[i], '.')[0]
        write_array = [file_id]
        
        for out in maxed_output:
            write_array.append(out.item())
    
        submission_writer.writerow(write_array)
        
        if i % 100 == 0 and i > 0:
            print('Predicted for ' + str(i) + ' of ' + str(len(test_files) + 1) + ' files')

print('Submission generated')