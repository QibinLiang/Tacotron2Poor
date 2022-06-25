import models
import utils
from torch.utils.data import DataLoader
import torch as tr

def TacoLoss(mel_out, mel_out_res, target_mel, stop_token, target_stop):
    # the target_mel is shape of (batch_size, T, feat_size)
    target_mel = target_mel.permute(0, 2, 1)
    stop_token = stop_token.squeeze(2)
    stop_token.requires_gard = False
    mel_out.requires_gard = False
    stop_token_loss = tr.nn.BCEWithLogitsLoss()(stop_token, target_stop)
    mel_out_loss = tr.nn.MSELoss()(target_mel, mel_out)
    mel_out_res_loss = tr.nn.MSELoss()(target_mel, mel_out_res)
    print(mel_out_loss , mel_out_res_loss , stop_token_loss)
    loss = mel_out_loss + mel_out_res_loss + stop_token_loss
    return loss


def data_to_device(data_list, device):
    data_list_device = []
    for data in data_list:
        data_list_device.append = data.to(device)
    return data_list_device


def train(train_json, log, epochs=1):
    dataset = utils.CustomDataset(train_json)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=utils.collate_fn_pad)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

    taco = models.Tacotron()
    taco = taco.to(device)
    optimizer = tr.optim.Adam(taco.parameters, lr=0.001, eps=1e-6, weight_decay=1e-6)

    loss_history = []
    loss_cum = 0
    for e in range(epochs):
        for _, item in enumerate(dataloader):
            #item = data_to_device(item, device)
            mel_batch, target_lens, target_stop, wrd, wrd_len = item
            mel_batch = mel_batch.to(device)
            wrd = wrd.to(device)
            target_stop = target_stop.to(device)

            mel_batch = mel_batch.permute(1, 0, 2)
            mel_out, mel_out_res, stop_token = taco(wrd, wrd_len, mel_batch)
            loss = TacoLoss(mel_out, mel_out_res, mel_batch, stop_token, target_stop)
            taco.zero_grad()
            loss.backward()
            grad_norm = tr.nn.utils.clip_grad_norm_(taco.parameters(), 1.0)
            optimizer.step()
            loss_cum += loss
            if _ % 99 == 0:
                f = open(log, 'a')
                avg_loss = loss_cum / 100.0
                f.write(avg_loss)
                f.write('\n')
                f.flush()
                f.close()

train('json/train.json', 'train.log')
