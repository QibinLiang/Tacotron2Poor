from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch as tr
import json
from torchaudio import transforms as T
import torchaudio


class CustomDataset(Dataset):
    def __init__(self, json_path, n_mels=80) -> None:
        super().__init__()
        f = open(json_path)
        self.data = json.load(f)
        self.data = [(key, self.data[key]) for key in self.data.keys()]
        self.fbank = T.MelSpectrogram(f_min=125, f_max=7600, n_mels=n_mels)
        self.n_mels = n_mels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, val = self.data[index]
        # get mel_spec
        wav_path = val['wav']
        # get transcript
        wrd = val['wrd']
        # get transcript length
        wrd_len = len(wrd)
        # load wav
        wav, sr = torchaudio.load(wav_path)
        padded_mel, target_stop = self.compute_padded_mel(wav, wrd_len, self.n_mels)
        return padded_mel, target_stop , wrd, wrd_len

    def compute_padded_mel(self, wav, wrd_len, n_mels, frames_per_char=6):
        frames = wrd_len * frames_per_char
        padded_mel = tr.zeros(n_mels, frames)
        target_stop = tr.ones(n_mels, frames)
        mel_spec = self.fbank(wav)
        target_mel_spec_len = mel_spec.shape[1]
        padded_mel[:, :target_mel_spec_len] = mel_spec
        target_stop[:, target_mel_spec_len:] = 0.0
        return padded_mel, target_stop
