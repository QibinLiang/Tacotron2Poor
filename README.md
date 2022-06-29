# Tacotron2 Poor
[**中文**](https://github.com/PhyseChan/Tacotron2Poor/blob/master/README.zh.md)

This is the implementation for [**Tacotron2**](https://arxiv.org/pdf/1712.05884.pdf) but implemented in an inefficient way that is lack of tricks and details. This project doesn't include the WaveNet module, which means this project just predicts the mel-spectrogram from the text.

This project uses the TIMIT dataset to train Tacotron2 model and the data will be compressed to a json file by SpeechBrain. More details and functionality will be added in the future.

----

### Basic modules
- [x] Tacotron2 Model
- [x] DataLoader
- [x] LossFunction
- [x] Training step
- [ ] Eval step
- [ ] Test step
- [ ] Inference
- [ ] WaveNet
- [x] logger

### Additional functionality
- [ ] Dynamic sampling (partly shuffle the sorted dataset)
- [ ] Parallel training
- [ ] Hyper parameters loader
- [x] Checkpoint saving and loading
- [ ] Data Normalization (Currently using the json file created by Speechbrain)
- [ ] Griffin-Lim

## train loss

### LJspeech
(https://github.com/PhyseChan/Tacotron2Poor/blob/master/train_loss.jpg)

[![loss on LJspeech]([图片地址](https://github.com/PhyseChan/Tacotron2Poor/train_loss.jpg))](https://github.com/PhyseChan/Tacotron2Poor/train_loss.jpg)
