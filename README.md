# NutNet
The implementation of CCS 2024 "I Don't Know You, But I Can Catch You: Real-Time Defense against Diverse Adversarial Patches for Object Detectors"

Demos can be found here: [https://sites.google.com/view/nutnet](https://sites.google.com/view/nutnet).

The full version of the paper with an appendix can be found here: [https://arxiv.org/abs/2406.10285](https://arxiv.org/abs/2406.10285).

## Files Description

- [model_weights/](model_weights/) - pretrained weights of the models （目标检测器的权重）
- [ae_train_process/](ae_train_process/) - some original and reconstructed images when training the autoencoder （观察自编码器训练过程中的输入输出结果）
- [ae_weights/](ae_weights/) - the trained autoencoder's weight （存训练的自编码器权重）
- [ae_weights_/](ae_weights_/) - our pretrained autoencoder's weight （我们已经训练好的自编码器权重）
- [Dataset/](Dataset/) - the dataset we use for evaluation （我们用来评估的数据集） [https://pan.quark.cn/s/77dec05e3e1e](https://pan.quark.cn/s/77dec05e3e1e)，提取码：hk5V
- [test_map/](test_map/) - the detection results on the validation dataset （存数据集的检测结果）

- [ae_trainer.ipynb](ae_trainer.ipynb) - train the autoencoder （训练自编码器）
- [test_map_defense.ipynb](test_map_defense.ipynb) - evaluate the AP (average precision) of the detection model with defense （测试有防御状态下的目标检测平均精度）
- [test_map_vanilia.ipynb](test_map_vanilia.ipynb) - evaluate the AP (average precision) of the detection model without defense （测试无防御状态下的目标检测平均精度）
- [patchFilter.py](patchFilter.py) - combine the defense and the detection model for convenience （封装自编码器和目标检测器）

## Some configurations of the NutNet

In [patchFilter.py](patchFilter.py), `box_num` controls the block size. 

检测器的 wrapper 类的初始化中通过控制 box_num 参数来决定 PatchFilter 分块的尺度，例如：

```
class YOLOv4_wrapper(object):

    def __init__(self, yolo_detector, inp_dim, device, box_num=32):
        self.yolo_detector = yolo_detector
        self.device = device
        self.inp_dim = inp_dim
        self.box_num = box_num
        self.box_length = inp_dim//self.box_num
        if box_num == 8:
            self.ae = AutoEncoder8().to(device)
            self.ae.load_state_dict(torch.load("./ae_weights/n_52.pth"))
        elif box_num == 16:
            self.ae = AutoEncoder16().to(device)
            self.ae.load_state_dict(torch.load("./ae_weights/n_26.pth"))
        elif box_num == 32:
            self.ae = AutoEncoder32().to(device)
            self.ae.load_state_dict(torch.load("./ae_weights/n_13.pth"))

        self.loss = nn.MSELoss(reduction='none')
```
The input image is 416\*416 pixels. When we set box_num=32, the image will be divided into 32\*32 blocks, and each one is 13\*13 pixels. 

此处设置了 box_num=32，即将输入的 416×416 图像分为 32×32 个小块，每块大小为 13×13。相应的也可以在下面设置需要载入的预训练权重。

---

```
mask1 = ((loss>0.2).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((self.box_num*self.box_num, 3, self.box_length, self.box_length)))
delta = torch.abs(output-img)
mask2 = (delta.sum(dim=[1])>.25).float().unsqueeze(1).expand(-1, 3, -1, -1)
mask = mask1*mask2
img2 = torch.where(mask==0, img, 0)

```

In the `__call__()` function, there are two thresholds for mask1 (coarse-grained) and mask2 (fine-grained). For example, we use 0.2 and 0.25 here. They can be modified according to the actual situation.

在 wrapper 类的 `__call__()` 函数中，使用了两个阈值，即 mask1 （粗粒度）和 mask2 （细粒度）的阈值，例如此处为 0.2 和 0.25，可以根据模型自行修改。
