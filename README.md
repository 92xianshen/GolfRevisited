Golf Revisited
==============

高尔夫数据、CNN 样例模型、训练/预测代码、预训练模型

在 TensorFlow 2.3.2 下 `inference.py` 通过测试，`train.py` 未经测试，仅供参考！

如需帮助，请联系 `jiaolibin@mail.bnu.edu.cn`

如果对研究工作有任何帮助，欢迎引用：

```
[1] Jiao L , Bie R , Wu H , et al. Golf swing classification with multiple deep convolutional neural networks[J]. International Journal of Distributed Sensor Networks, 2018, 14(10).
[2] Wu H , Bie R , Jiao L , et al. Towards Real-Time Multi-Sensor Golf Swing Classification Using Deep CNNs[J]. Journal of database management, 2018, 29(3):17-42.
```

`dataset/`
----------

数据格式：`NPZ` 格式，X*.npz 为高尔夫信号，y*.npz 为人工标记数据，`.shape` 为 `[<n_size>, <n_sensor>, <n_signal_sample>]`

数据预处理：数据经过标准化处理，均值为 0，标准差为 1

数据划分：请参考 ppt

另外，`original/` 内为原始数据，`balanced/` 内为平衡数据

`model/`
--------

`GolfResNet.py`：CNN 模型定义

`GolfResNetFeats.py`：CNN 特征提取部分

`ppt/`

2021 年 6 月 22 日汇报 ppt

`pretrained/`
-------------

预训练模型

`inference.py`
--------------

加载预训练模型，逐条数据预测的样例代码

注意：`tensorflow.keras.Model.predict` 输出 logits，`argmax` 输出分类结果

`train.py`
----------

之前使用的训练代码，未经测试，仅供参考！