import csv
import librosa
import numpy as np
from skimage.transform import resize
from PIL import Image


fft = 1024
hop = 256
sr = 32000
length = 10 * sr    # re les to 10ari na einai to kalo?

with open('train_tp.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

# Apo Lefterh
fmin = 0
fmax = 16000

print('Starting spectrogram generation')
for i in range(1, len(data)):
    # Keep original SR
    wav, sr = librosa.load('train/' + data[i][0] + '.flac', sr=None)
    
    t_min = float(data[i][3]) * sr
    t_max = float(data[i][5]) * sr
    
    # Positioning sound slice
    center = np.round((t_min + t_max) / 2)
    beginning = center - length / 2
    if beginning < 0:
        beginning = 0
    
    ending = beginning + length
    if ending > len(wav):
        ending = len(wav)
        beginning = ending - length
        
    slice = wav[int(beginning):int(ending)]
    
    # Mel-spectrogram generation
    mel_spec = librosa.feature.melspectrogram(slice, n_mels=128, n_fft=fft, hop_length=hop, sr=sr, fmin=fmin, fmax=fmax, power=1.5)
    mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
    mel_spec = np.flipud(mel_spec)
    # Resize 224 width for ResNet
    mel_spec = resize(mel_spec, (224, 547))
    
    # Normalize to 0...1
    mel_spec = mel_spec - np.min(mel_spec)
    mel_spec = mel_spec / np.max(mel_spec)

    # Save to PNG
    mel_spec = mel_spec * 255
    mel_spec = np.round(mel_spec)    
    mel_spec = mel_spec.astype('uint8')
    mel_spec = np.asarray(mel_spec)
    
    png = Image.fromarray(mel_spec, 'L')
    png.save('mel_spectrums_flipped/' + data[i][0] + '_' + data[i][1] + '_' + str(center) + '.png')
    
    if i % 100 == 0:
        print('Processed ' + str(i) + ' train examples from ' + str(len(data)))