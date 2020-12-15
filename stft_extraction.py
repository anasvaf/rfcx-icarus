import csv
import librosa
import numpy as np
from skimage.transform import resize
from PIL import Image


fft = 1024
hop = 256
sr = 28000
length = 10 * sr    # re les to 10ari na einai to kalo?

with open('train_tp.csv') as f:
    reader = csv.reader(f)
    data = list(reader)

# Apo Lefterh
fmin = 24000
fmax = 0

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
    
    # STFT-spectrogram generation
    stft_spec = librosa.stft(slice, n_fft=fft, hop_length=hop, window='hann')
    stft_spec = librosa.amplitude_to_db(np.abs(stft_spec))
    # Resize 224 width for ResNest
    stft_spec = resize(stft_spec, (224, 400))
    
    # Normalize to 0...1
    stft_spec = stft_spec - np.min(stft_spec)
    stft_spec = stft_spec / np.max(stft_spec)

    # Save to PNG
    stft_spec = stft_spec * 255
    stft_spec = np.round(stft_spec)    
    stft_spec = stft_spec.astype('uint8')
    stft_spec = np.asarray(stft_spec)
    
    png = Image.fromarray(stft_spec, 'L')
    png.save('stft_spectrums_28k/' + data[i][0] + '_' + data[i][1] + '_' + str(center) + '.png')
    
    if i % 100 == 0:
        print('Processed ' + str(i) + ' train examples from ' + str(len(data)))