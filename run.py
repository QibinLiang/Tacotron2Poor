import models
import utils
from torch.utils.data import DataLoader
import torch as tr
import logging
import speechbrain

logger = logging.getLogger(__name__)

def TacoLoss(mel_out, mel_out_res, target_mel, stop_token, target_stop):
    # the target_mel is shape of (batch_size, T, feat_size)
    target_mel = target_mel.permute(0, 2, 1)
    stop_token = stop_token.squeeze(2)
    target_stop.requires_gard = False
    target_mel.requires_gard = False
    stop_token_loss = tr.nn.BCEWithLogitsLoss()(stop_token, target_stop)
    mel_out_loss = tr.nn.MSELoss()(target_mel, mel_out)
    mel_out_res_loss = tr.nn.MSELoss()(target_mel, mel_out_res)
#    print(mel_out_loss, mel_out_res_loss, stop_token_loss, mel_out.shape)
    print(mel_out_loss, mel_out_res_loss,stop_token_loss)
    loss = mel_out_loss + mel_out_res_loss + stop_token_loss
    return loss


def data_to_device(data_list, device):
    data_list_device = []
    for data in data_list:
        data_list_device.append = data.to(device)
    return data_list_device

def get_mask_from_lengths(lengths, max_len=None):
    """Creates a mask from a tensor of lengths
    Arguments
    ---------
    lengths: torch.Tensor
        a tensor of sequence lengths
    Returns
    -------
    mask: torch.Tensor
        the mask
    max_len: int
        The maximum length, i.e. the last dimension of
        the mask tensor. If not provided, it will be
        calculated automatically
    """
    if max_len is None:
        max_len = tr.max(lengths).item()
    ids = tr.arange(0, max_len, device=lengths.device, dtype=lengths.dtype)
    mask = (ids < lengths.unsqueeze(1)).byte()
    mask = tr.le(mask, 0)
    return mask

def train(train_json, log, train_logger, epochs=120):
    dataset = utils.CustomDataset(train_json)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=utils.collate_fn_pad)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
    
    taco = models.Tacotron()
    taco = taco.to(device)
    optimizer = tr.optim.Adam(taco.parameters(), lr=0.001, eps=1e-6, weight_decay=1e-6)

    loss_history = []
    loss_cum = 0
    for e in range(epochs):
        print("------------  " + str(e) +" -----------",file=open("output.txt", "a") )
        for _, item in enumerate(dataloader):
            #item = data_to_device(item, device)
            mel_batch, target_lens, target_stop, wrd, wrd_len = item
            mel_batch = mel_batch.to(device)
            target_stop = target_stop.to(device)
            wrd = wrd.to(device)
            mel_batch = mel_batch.permute(1, 0, 2)
            wrd_len_mask = get_mask_from_lengths(wrd_len).to(device)
            mel_out, mel_out_res, stop_token = taco(wrd, wrd_len, mel_batch,wrd_len_mask)
            loss = TacoLoss(mel_out, mel_out_res, mel_batch, stop_token, target_stop)
            taco.zero_grad()
            loss.backward()
            grad_norm = tr.nn.utils.clip_grad_norm_(taco.parameters(), 1.0)
            optimizer.step()
            loss_cum += loss
            train_logger.log_stats(
                stats_meta={"epoch": e, "grad":grad_norm},
                train_stats={"loss": loss},
            )
            if _ == 20:
                print("20")
            print(loss, grad_norm,file=open("output.txt", "a"))
        tr.save({
            'epoch': e,
            'model_state_dict': taco.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, 'model.pt')
train_logger = speechbrain.utils.train_logger.FileTrainLogger("train_log.txt" )
train('json/train.json', 'train.log', train_logger)
