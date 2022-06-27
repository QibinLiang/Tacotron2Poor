import models
import utils
from torch.utils.data import DataLoader
import torch as tr
import logging
import speechbrain
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def tacoloss(mel_out, mel_out_res, target_mel, stop_token, target_stop):
    """The tacoloss combines the BCE loss for predicting the frames of mel-spectrogram,
    the MSE loss for the predicted mel-spectrogram and the MSE loss for the residual mel-spectrogram.

    Parameters
    ----------
    mel_out :  (B, T_max, mel_dim)
        predicted mel-spectrogram from decoder.
    mel_out_res :  (B, T_max, mel_dim)
        sum of predicted mel and residual mel.
    target_mel :  (B, mel_dim, T_max)
        the padded target mel-spectrogram
    stop_token :  (B, T_max, 1)
        padded prediction for the length of mel-spectrogram
    target_stop : (B, T_max)
        padded ground truth length of mel-spectrogram
    Returns
    -------
    loss :
        the loss of Tacotron model
    """
    # the target_mel is shape of (B, T_max, mel_dim)
    target_mel = target_mel.permute(0, 2, 1)
    stop_token = stop_token.squeeze(2)
    target_stop.requires_gard = False
    target_mel.requires_gard = False
    # compute BCE loss for stop frame prediction
    stop_token_loss = tr.nn.BCEWithLogitsLoss()(stop_token, target_stop)
    # compute MSE loss for mel-spectrogram prediction
    mel_out_loss = tr.nn.MSELoss()(target_mel, mel_out)
    # compute MSE loss for the residual of mel-spectrogram prediction
    mel_out_res_loss = tr.nn.MSELoss()(target_mel, mel_out_res)
    # combine all losses
    loss = mel_out_loss + mel_out_res_loss + stop_token_loss
    return loss


def data_to_device(data_list, device):
    """ put the data to the device

    Parameters
    ----------
    data_list :
        ( mel_batch,
        target_lens,
        target_stop,
        wrd, wrd_len )

    device :
        device used to train the model
    Returns
    -------
        mel_batch, target_lens, target_stop, wrd, wrd_len
    """
    mel_batch, target_lens, target_stop, wrd, wrd_len = data_list
    mel_batch = mel_batch.to(device)
    target_stop = target_stop.to(device)
    wrd = wrd.to(device)
    return mel_batch, target_lens, target_stop, wrd, wrd_len


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


def train(train_json, train_logger, epochs=120, load=False):
    dataset = utils.CustomDataset(train_json)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=utils.collate_fn_pad)
    device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

    taco = models.Tacotron()
    taco = taco.to(device)
    optimizer = tr.optim.Adam(taco.parameters(), lr=0.001, eps=1e-6, weight_decay=1e-6)

    # if load==True, it will load the model and continue the training.
    if load:
        checkpoint = tr.load('model.pt')
        taco.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        begin_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        begin_epoch = 0

    for e in range(begin_epoch, epochs):
        for _, item in enumerate(dataloader):
            item = data_to_device(item, device)
            mel_batch, target_lens, target_stop, wrd, wrd_len = item
            mel_batch = mel_batch.permute(1, 0, 2)
            # create the mask for the attention location weights
            wrd_len_mask = get_mask_from_lengths(wrd_len).to(device)
            # forward
            mel_out, mel_out_res, stop_token = taco(wrd, wrd_len, mel_batch, wrd_len_mask)
            loss = tacoloss(mel_out, mel_out_res, mel_batch, stop_token, target_stop)
            # clear gradient
            taco.zero_grad()
            # backward
            loss.backward()
            # clip the gradient to avoid gradient explosion
            grad_norm = tr.nn.utils.clip_grad_norm_(taco.parameters(), 1.0)
            # update learnable parameters
            optimizer.step()
            train_logger.log_stats(
                stats_meta={"epoch": e, "grad": grad_norm},
                train_stats={"loss": loss},
            )
        tr.save({
            'epoch': e,
            'model_state_dict': taco.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, 'model.pt')
        return


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='Tacoton2 Poor')
    parser.add_argument('--train', type=bool, default=True, help='whether the mode is training or not')
    parser.add_argument('--json', type=str, default=None, help='the path of json file')
    parser.add_argument('--data', type=str, default=None, help='the path of dataset')
    parser.add_argument('--text', type=str, default=None, help='the path of text for inference')
    parser.add_argument('--log', type=str, help='the path of logger file',
                        default='train_log.txt')
    parser.add_argument('--load', type=bool, default=None, help='continue the training')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    is_train = args.train
    data_path = args.data
    json_path = args.json
    text_path = args.text
    log_path = args.log

    if is_train:
        assert data_path or json_path, "either data path or json path should be given"
    else:
        assert text_path, "text path should be given if the mode is inference"

    train_logger = speechbrain.utils.train_logger.FileTrainLogger(log_path)
    train(json_path, train_logger)
