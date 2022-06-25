from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch as tr
import json
from torchaudio import transforms as T
import torchaudio
import config
import torch


class CustomDataset(Dataset):
    def __init__(self, json_path, n_mels=80) -> None:
        super().__init__()
        f = open(json_path)
        self.data = json.load(f)
        self.data = [(key, self.data[key]) for key in self.data.keys()]
        self.fbank = T.MelSpectrogram(sample_rate=16000, win_length=800, hop_length=200, n_fft=1024, n_mels=n_mels, f_min=125,
                                   f_max=7600, normalized='slaney')
        #self.fbank = T.MelSpectrogram(f_min=125, f_max=7600, n_mels=n_mels, normalized=True)
        self.n_mels = n_mels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, val = self.data[index]

        wav_path = val['wav']
        # get transcript
        wrd = val['wrd']
        wrd = tr.LongTensor(character_tokenizer(wrd, config.token2id))
        # get transcript length
        wrd_len = len(wrd)
        # load wav
        wav, sr = torchaudio.load(wav_path)
        # get mel_spec
        padded_mel, target_mel_spec_len, target_stop = self.compute_mel(wav, wrd_len, self.n_mels)
        padded_mel = padded_mel.squeeze(0).permute(1, 0)
        return padded_mel, target_mel_spec_len, target_stop, wrd, wrd_len

    def compute_mel(self, wav, wrd_len, n_mels):
        mel_spec = self.fbank(wav)
        target_stop = tr.ones(mel_spec.shape[-1])
        target_mel_spec_len = mel_spec.shape[2]
        return mel_spec, target_mel_spec_len, target_stop


def character_tokenizer(wrd, token2id):
    ids = []
    for token in wrd:
        ids.append(token2id[token])
    return ids


def convert_token_char(tokens, token2id):
    id2token = {token2id[key]: key for key in token2id.keys()}
    wrds = [id2token[token] for token in tokens]
    return wrds


def collate_fn_pad(batch):
    padded_mel, frames, target_stop, wrd, wrd_len = zip(*batch)
    # get the target length
    target_lens = tr.tensor([f for f in frames])
    # compat all mels
    mel_batch = [tr.Tensor(t) for t in padded_mel]
    # pad mels again
    mel_batch = tr.nn.utils.rnn.pad_sequence(mel_batch)

    # pad target_stop
    target_stop = tr.nn.utils.rnn.pad_sequence(target_stop,batch_first=True)
    # pad wrd
    wrd = tr.nn.utils.rnn.pad_sequence(wrd, batch_first=True)
    wrd_len = tr.tensor([l for l in wrd_len])
    batch = mel_batch, target_lens, target_stop, wrd, wrd_len
    return batch


#dataset = CustomDataset('dev.json')
#dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn_pad)
#mel_outputs, mel_outputs_res, stop_tokens = batch = next(iter(dataloader))

#print(mel_outputs.shape, mel_outputs_res.shape, stop_tokens.shape)
