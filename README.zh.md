# Tacotron2 Poor 
这个Tacotron2 poor是灰烬山寨破损版[**Tacotron2**](https://arxiv.org/pdf/1712.05884.pdf)的复现项目，项目暂时仅实现了Tacotron 2的基本功能与模型，WaveNet并未包含其中，日后会添加。模型使用了TIMIT作为训练集，其中训练使用的json文件是由SpeechBrain所生成的。

----

### 基本模块
- [x] Tacotron2模型
- [x] DataLoader
- [x] 损失函数
- [x] 训练
- [ ] 验证
- [ ] 测试
- [ ] 推断
- [ ] WaveNet
- [x] 日志

### 额外功能
- [ ] 动态采样（局部随机采样已排序的文本）
- [ ] 并行计算
- [ ] 自定义超参，加载超参
- [x] 检查点保存与加载
- [ ] 数据预处理
- [ ] Griffin-Lim 算法

## 训练

### 配置

使用了E5-2678 cpu以及 Tesla k80显卡。 由于显存限制，batch size设置为32。

### LJspeech
![LJspeech train loss](https://github.com/PhyseChan/Tacotron2Poor/blob/master/train_loss.jpg)


`重要⚠️` 如上图所示，损失值在第1800个迭代附近突然地升高，这种现象在论文[paper](https://arxiv.org/pdf/2204.13437.pdf)也被提及
。

