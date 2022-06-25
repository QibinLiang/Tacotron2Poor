# Tacotron2 Poor 
这个Tacotron2 poor是灰烬山寨破损版的Tacotron 2复现项目，项目暂时仅实现了Tacotron 2的基本功能与模型，WaveNet并未包含其中，日后会添加。模型使用了TIMIT作为训练集，其中训练使用的json文件是由SpeechBrain所生成的。

----

### 基本模块
- [x] Tacotron2模型
- [x] DataLoader
- [x] 损失函数
- [x] 训练
- [ ] 推断
- [ ] WaveNet

### Additional functionality
- [ ] 动态采样（局部随机采样已排序的文本）
- [ ] 并行计算
- [ ] 自定义超参，加载超参
- [ ] 检查点保存与加载
- [ ] 数据预处理