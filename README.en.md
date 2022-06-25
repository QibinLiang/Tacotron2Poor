# Tacotron2 Poor 
This is the implementation for Tacotron2 but implemented in an inefficient way that is lack of tricks and details. This project doesn't include the WaveNet module, which means this project just predicts the mel-spectrogram from the text.

This project uses the TIMIT dataset to train Tacotron2 model and the data will be compressed to a json file by SpeechBrain. More details and functionality will be added in the future.

----

### Basic modules
- [x] Tacotron2 Model
- [x] DataLoader
- [x] LossFunction
- [x] Training Step
- [ ] Inference
- [ ] WaveNet

### Additional functionality
- [ ] Dynamic sampling (partly shuffle the sorted dataset)
- [ ] Distributed training
- [ ] Hyper parameters loader
- [ ] Checkpoint saving and loading
- [ ] Data Normalization (Currently using the json file created by Speechbrain)