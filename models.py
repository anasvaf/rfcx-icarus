from functools import partial
from torchlibrosa.stft import Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
import torch.nn as nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from pytorch_utils import do_mixup
import timm


encoder_params = {
    "resnest50d" : {
        "features" : 2048,
        "init_op"  : partial(timm.models.resnest50d, 
                            pretrained=True,
                            in_chans=1)
    }
}


class AudioClassifier(nn.Module):
    def __init__(self, encoder, sample_rate, window_size, 
                hop_size, classes_num):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, 
                                                hop_length=hop_size, 
                                                win_length=window_size, 
                                                window=window, 
                                                center=center, 
                                                pad_mode=pad_mode, 
                                                freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, 
                                                time_stripes_num=2, 
                                                freq_drop_width=8, 
                                                freq_stripes_num=2)

        self.encoder = encoder_params[encoder]["init_op"]()
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.dropout = Dropout(0.3)
        self.fc = Linear(encoder_params[encoder]['features'], classes_num)
    
    def forward(self, input, spec_aug=False, mixup_lambda=None):
        #print(input.type())
        x = self.spectrogram_extractor(input.float()) 
        # (batch_size, 1, time_steps, freq_bins)

        #if spec_aug:
        #    x = self.spec_augmenter(x)
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
            #pass
        
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x