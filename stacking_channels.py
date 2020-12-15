import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

from joblib import Parallel, delayed

class spectrograms:
  
    def __init__(self, spec_folder = './spectrograms', images = './images', dim = 256, n_mfcc = 60, 
                 n_fft = 1024, sample_rate = 22050, kill_temp = False, stack = True, size = 256):
        self.dim = dim
        self.spec_folder = spec_folder
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.kill_temp = kill_temp
        self.stack = stack
        self.size = size
        self.images = images
     
        
    def makdirs(self):
        try:           
            os.mkdir(self.images)
            os.mkdir(os.path.join(self.images, 'train_tp'))
            os.mkdir(os.path.join(self.images, 'train_fp'))
            os.mkdir(self.spec_folder)
            os.mkdir(os.path.join(self.spec_folder, 'mel'))
            os.mkdir(os.path.join(self.spec_folder, 'mel', 'train_tp'))
            os.mkdir(os.path.join(self.spec_folder, 'mel', 'train_fp'))
            os.mkdir(os.path.join(self.spec_folder, 'stft'))
            os.mkdir(os.path.join(self.spec_folder, 'stft', 'train_tp'))
            os.mkdir(os.path.join(self.spec_folder, 'stft', 'train_fp'))
            os.mkdir(os.path.join(self.spec_folder, 'mfcc'))
            os.mkdir(os.path.join(self.spec_folder, 'mfcc', 'train_tp'))
            os.mkdir(os.path.join(self.spec_folder, 'mfcc', 'train_fp'))
        except:
            pass
        
    def labels(self, csv_in): 
        if os.path.exists(os.path.splitext(csv_in)[0] + '_labels.csv'):
            return
        else:
            train = pd.read_csv(csv_in)
            ids = []
            lab = []    
            for i in range(len(train)):
                y =[]
                if train['recording_id'][i] not in ids:
                    ids.append(train['recording_id'][i])
                    for row in range(i, len(train)):
                        if train['recording_id'][row] == train['recording_id'][i]:
                            y.append(train['species_id'][row])
                    lab.append(np.unique(y))
            ids = pd.Series(ids, name = 'IDs')
            lab = pd.Series(lab, name = 'labels')

            labdf = pd.DataFrame()
            for row in range(len(lab)):
                for i in range(len(list(lab[row]))):
                    name = 's' + str(list(lab[row])[i])
                    if name not in labdf:
                        labdf[name] = 0
                    labdf.at[row,name] = 1
            labdf = labdf.reindex(sorted(labdf.columns, key=lambda x: int(x[1:])), axis=1)
            df = pd.merge(ids, labdf, right_index = True, left_index = True) 
            df.fillna(0, inplace=True)
            df.to_csv(os.path.splitext(csv_in)[0] + '_labels.csv')#, index=False)

    def create_spectrograms(self):
        self.labels('train_tp.csv')
        self.labels('train_fp.csv')
        
        def produce(row, fold):
            samples, rate = librosa.load(os.path.join('train', row + '.flac'), sr=None)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            filename = (os.path.join(self.spec_folder, 'mel', fold, row + '.png'))
            Mel = librosa.feature.melspectrogram(y=samples, sr=self.sample_rate, n_fft = self.n_fft, hop_length = self.n_fft//2)
            librosa.display.specshow(librosa.power_to_db(Mel, ref=np.max))
            plt.savefig(os.path.join(filename), bbox_inches='tight',pad_inches=0)
            plt.close('all')
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            filename = (os.path.join(self.spec_folder, 'stft', fold, row + '.png'))
            S = librosa.stft(y=samples, n_fft = self.n_fft, hop_length = self.n_fft//2)
            Sdb = librosa.amplitude_to_db(abs(S))
            librosa.display.specshow(Sdb, sr = self.sample_rate)
            plt.savefig(os.path.join(filename), bbox_inches='tight',pad_inches=0)
            plt.close('all')
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            filename = (os.path.join(self.spec_folder, 'mfcc', fold, row + '.png'))
            S = librosa.feature.mfcc(y=samples, sr = self.sample_rate, n_mfcc = self.n_mfcc, n_fft = self.n_fft, hop_length = self.n_fft//2)
            librosa.display.specshow(S, sr = self.sample_rate)
            plt.savefig(os.path.join(filename), bbox_inches='tight',pad_inches=0)
            plt.close('all')
            
        train_tp = pd.read_csv('train_tp_labels.csv')
        train_tp = list(train_tp['IDs'])
        train_fp = pd.read_csv('train_fp_labels.csv')       
        train_fp = list(train_fp['IDs'])
            
        Parallel(n_jobs=4)(delayed(produce)(row, 'train_tp') 
                                            for row in tqdm(train_tp, total = len(train_tp), position=0, leave=True))
        Parallel(n_jobs=4)(delayed(produce)(row, 'train_fp') 
                                            for row in tqdm(train_fp, total = len(train_fp), position=0, leave=True))
        
    def spectrogram_image(self, p):
  
        if self.stack:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(self.size, self.size),interpolation = cv2.INTER_AREA)
        else:
            img = cv2.imread(p)
        X = np.array(img)
        return X
    
    def read_resize(self, file, fold):
        
        mel = os.path.join(self.spec_folder, 'mel', fold, file + '.png')        
        Xmel = self.spectrogram_image(mel)

        stft = os.path.join(self.spec_folder, 'stft', fold, file + '.png')  
        Xstft = self.spectrogram_image(stft)

        mfcc = os.path.join(self.spec_folder, 'mfcc', fold, file + '.png')  
        Xmfcc = self.spectrogram_image(mfcc) 
        
        if self.stack:
            Xmel = np.reshape(Xmel, (Xmel.shape[0], Xmel.shape[1], 1))
            Xstft = np.reshape(Xstf, (Xstft.shape[0], Xstft.shape[1], 1))
            Xmfcc = np.reshape(Xmfc, (Xmfcc.shape[0], Xmfcc.shape[1], 1))
            
            X = np.concatenate((Xmel, Xstft, Xmfcc), axis=2)
        else:
            X = np.vstack((Xmel, Xstft, Xmfcc))

        im = Image.fromarray(X)
        im.save(os.path.join(self.images, str(fold), file + '.png'))
        
    def merge(self):
        
        train_tp = pd.read_csv('train_tp_labels.csv')
        train_tp = list(train_tp['IDs'])
        train_fp = pd.read_csv('train_fp_labels.csv')       
        train_fp = list(train_fp['IDs'])
            
        Parallel(n_jobs=4, backend = 'threading')(delayed(self.read_resize)(file, 'train_tp')
                                                               for file in tqdm(train_tp, total = len(train_tp), position=0, leave=True))
        Parallel(n_jobs=4, backend = 'threading')(delayed(self.read_resize)(file, 'train_fp')
                                                               for file in tqdm(train_fp, total = len(train_fp), position=0, leave=True))
        
        if self.kill_temp:
            os.rmdir('./spectrograms')
         
if __name__ == "__main__":
    spectrograms().makdirs()
    spectrograms().create_spectrograms()
    spectrograms().merge()
