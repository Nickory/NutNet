import torch
import torch.nn as nn
import numpy as np

from darknet_v2 import Darknet
from darknet_v3 import Darknet as Darknet53
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
from util_model_img import predict_transform_v2, tensor_to_image, write_result, write_results



class AutoEncoder8(nn.Module):

    def __init__(self):
        super(AutoEncoder8, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=2, padding=1),    # batch, 8, 24, 24
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),    # batch, 16, 12, 12
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),    # batch, 32, 6, 6
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # batch, 16, 12, 12
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),    # batch, 16, 24, 24
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=8, stride=2, padding=1),    # batch, 3, 52, 52
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


class AutoEncoder16(nn.Module):

    def __init__(self):
        super(AutoEncoder16, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=2, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),    # batch, 32, 5, 5
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2, padding=1),    # batch, 3, 26, 26
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        
        return x


class AutoEncoder32(nn.Module):

    def __init__(self):
        super(AutoEncoder32, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=2, stride=1, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=1),    # batch, 32, 5, 5
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=1),    # batch, 16, 8, 8
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=1),    # batch, 8, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=1, padding=1),    # batch, 3, 13, 13
            nn.Tanh(),
        )

    def forward(self, x):

        x = self.encoder(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        
        return x



class YOLOv2_wrapper(object):

    def __init__(self, yolo_detector, inp_dim, num_classes, stride, confidence, nms_thresh, device, box_num=8):
        self.yolo_detector = yolo_detector
        #self.yolo_detector.eval()
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.stride = stride
        self.conf_thres = confidence
        self.nms_iou_thres = nms_thresh
        self.device = device
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


    def __call__(self, img0):

        img = img0*2-1

        img = img.unfold(2, self.box_length, self.box_length).unfold(3, self.box_length, self.box_length)
        img = img.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.box_length, self.box_length, 3)
        img = img.permute(0, 3, 1, 2)

        output = self.ae(img)

        loss = torch.mean(self.loss(output, img), dim=(1,2,3))
        mask1 = ((loss>0.125).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((self.box_num*self.box_num, 3, self.box_length, self.box_length)))
        delta = torch.abs(output-img)
        mask2 = (delta.sum(dim=[1])>.2).float().unsqueeze(1).expand(-1, 3, -1, -1)
        mask = mask1*mask2
        img2 = torch.where(mask==0, img, 0)

        img2 = img2.unsqueeze(0).view(1, self.box_num, self.box_num, 3, self.box_length, self.box_length)#
        img2 = img2.permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 3, self.inp_dim, self.inp_dim)
        img2 = torch.clamp(0.5*(img2+1), 0, 1)

        prediction = self.yolo_detector(img2).data
        prediction = predict_transform_v2(prediction, self.inp_dim, self.yolo_detector.anchors, self.num_classes, self.stride, self.conf_thres, True)
        output = write_results(prediction, self.conf_thres, self.num_classes, nms=True, nms_conf=self.nms_iou_thres)

        if type(output) is int:
            pass
        elif output.size(-1) == 86:
            output = 0
        else:
            output[:, 5] = output[:, 5]*output[:, 6]
            output[:, 6] = output[:, 7]
            output = output[:, 1:-1]

        return output


class YOLOv3_wrapper(object):

    def __init__(self, yolo_detector, inp_dim, num_classes, confidence, nms_thresh, device, box_num=32):
        self.yolo_detector = yolo_detector
        #self.yolo_detector.eval()
        self.num_classes = num_classes
        self.conf_thres = confidence
        self.nms_iou_thres = nms_thresh
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

    def __call__(self, img0):
        
        img = img0*2-1

        img = img.unfold(2, self.box_length, self.box_length).unfold(3, self.box_length, self.box_length)
        img = img.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.box_length, self.box_length, 3)
        img = img.permute(0, 3, 1, 2)

        output = self.ae(img)
        loss = torch.mean(self.loss(output, img), dim=(1,2,3))
        mask1 = ((loss>0.125).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((self.box_num*self.box_num, 3, self.box_length, self.box_length)))

        delta = torch.abs(output-img)
        mask2 = (delta.sum(dim=[1])>0.3).float().unsqueeze(1).expand(-1, 3, -1, -1)
        mask = mask1*mask2
        img2 = torch.where(mask==0, img, 0)

        img2 = img2.unsqueeze(0).view(1, self.box_num, self.box_num, 3, self.box_length, self.box_length)#
        img2 = img2.permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 3, self.inp_dim, self.inp_dim)
        img2 = torch.clamp(0.5*(img2+1), 0, 1)

        prediction = self.yolo_detector(img2).data
        output = write_result(prediction, self.conf_thres, self.num_classes, nms=True, nms_conf=self.nms_iou_thres)

        if type(output) is int:
            pass
        elif output.size(-1) == 86:
            output = 0
        else:
            output[:, 5] = output[:, 5]*output[:, 6]
            output[:, 6] = output[:, 7]
            output = output[:, 1:-1]

        return output


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

    def __call__(self, img0):
        
        img = img0*2-1

        img = img.unfold(2, self.box_length, self.box_length).unfold(3, self.box_length, self.box_length)
        img = img.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, self.box_length, self.box_length, 3)
        img = img.permute(0, 3, 1, 2)

        output = self.ae(img)
        loss = torch.mean(self.loss(output, img), dim=(1,2,3))

        mask1 = ((loss>0.2).float().unsqueeze(1).unsqueeze(2).unsqueeze(3).expand((self.box_num*self.box_num, 3, self.box_length, self.box_length)))
        delta = torch.abs(output-img)
        mask2 = (delta.sum(dim=[1])>.25).float().unsqueeze(1).expand(-1, 3, -1, -1)
        mask = mask1*mask2
        img2 = torch.where(mask==0, img, 0)

        img2 = img2.unsqueeze(0).view(1, self.box_num, self.box_num, 3, self.box_length, self.box_length)#
        img2 = img2.permute(0, 3, 1, 4, 2, 5).contiguous().view(1, 3, self.inp_dim, self.inp_dim)
        img2 = torch.clamp(0.5*(img2+1), 0, 1)

        img2 = tensor_to_image(img2)

        image, boxes, scores, classes = self.yolo_detector.detect_image(img2, draw_image=False)

        if boxes is None:
            return None
        output = np.concatenate((boxes, np.expand_dims(scores,axis=-1), np.expand_dims(classes,axis=-1)), axis=1)

        return output


def defense_model_v2(device="cuda", confidence=0.5, nms_thresh=0.4, stride=32, num_classes=80, inp_dim=416, box_num=32):

    device = torch.device(device)
    weightsfile = 'model_weights/yolo.weights'
    cfgfile = "cfg/yolo.cfg"

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model = model.to(device)
    model.eval()

    model = YOLOv2_wrapper(model, inp_dim, num_classes, stride, confidence, nms_thresh, device, box_num=box_num)

    return model


def defense_model_v3(device="cuda", confidence=0.5, nms_thresh=0.4, num_classes=80, inp_dim=416, box_num=32):

    device = torch.device(device)
    weightsfile = "model_weights/yolov3.weights"
    cfgfile = "cfg/yolov3.cfg"

    model = Darknet53(device, cfgfile)
    model.load_weights(weightsfile)
    model.net_info["height"] = inp_dim
    model = model.to(device)
    model.eval()

    model = YOLOv3_wrapper(model, inp_dim, num_classes, confidence, nms_thresh, device, box_num=box_num)

    return model


def defense_model_v4(device="cuda", confidence=0.5, nms_thresh=0.4, num_anchors=3, num_classes=80, inp_dim=416, box_num=32):

    CUDA = True
    all_classes = get_classes("data/coco.names")
    num_classes = len(all_classes)
    yolo = YOLOv4(num_classes, num_anchors)
    yolo = yolo.to(device)
    yolo.load_state_dict(torch.load("model_weights/pytorch_yolov4_1.pt"))
    yolo.eval()
    model = Decode(confidence, nms_thresh, (inp_dim, inp_dim), yolo, all_classes, CUDA)

    model = YOLOv4_wrapper(model, inp_dim, device, box_num=box_num)

    return model

