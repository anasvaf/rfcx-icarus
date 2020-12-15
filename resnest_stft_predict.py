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


fft = 1024
hop = 256
sr = 28000
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


def load_test_file(f):
    wav, sr = librosa.load('test/' + f, sr=None)

    # Split for enough segments to not miss anything
    segments = len(wav) / length
    segments = int(np.ceil(segments))
    
    spec_array = []
    
    for i in range(0, segments):
        # Last segment going from the end
        if (i + 1) * length > len(wav):
            slice = wav[len(wav) - length:len(wav)]
        else:
            slice = wav[i * length:(i + 1) * length]
        
        # Same mel spectrogram as before
        stft_spec = librosa.stft(slice, n_fft=fft, hop_length=hop, window='hann')
        stft_spec = librosa.amplitude_to_db(np.abs(stft_spec))
        stft_spec = resize(stft_spec, (224, 400))
    
        stft_spec = stft_spec - np.min(stft_spec)
        stft_spec = stft_spec / np.max(stft_spec)
        
        stft_spec = np.stack((stft_spec, stft_spec, stft_spec))

        spec_array.append(stft_spec)
    
    return spec_array


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

model = torch.load('best_model_stft_28k - Copy.pt')
model.eval()

if torch.cuda.is_available():
    model.cuda()
    
# Prediction loop
print('Starting prediction loop')
with open('submission_best_loss.csv', 'w', newline='') as csvfile:
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