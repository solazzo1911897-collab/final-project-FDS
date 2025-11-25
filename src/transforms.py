import torch
import torch.nn as nn
from nnAudio.Spectrogram import CQT1992v2
import torchaudio.transforms as T

class GWTransform(nn.Module):
    def __init__(self, sr=2048, fmin=20, fmax=500, hop_length=64):
        super().__init__()
        self.cqt = CQT1992v2(
            sr=sr,
            fmin=fmin,
            fmax=fmax,
            hop_length=hop_length,
            output_format="Magnitude",
            verbose=False
        )
        
        self.time_masking = T.TimeMasking(time_mask_param=5) 
        self.freq_masking = T.FrequencyMasking(freq_mask_param=2)

    def forward(self, waveform, training=False):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        spec = self.cqt(waveform)
        spec = torch.log1p(spec)
        
        if training:
            spec = self.time_masking(spec)
            spec = self.freq_masking(spec)
        
        return spec