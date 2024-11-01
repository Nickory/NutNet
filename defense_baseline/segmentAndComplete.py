import torch
import numpy as np
import torchvision.transforms as T

from darknet_v2 import Darknet
from darknet_v3 import Darknet as Darknet53
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
from util_model_img import load_classes, predict_transform_v2, tensor_to_image, write_results, write_result

from defense_baseline.patch_detector import PatchDetector




class YOLOv2_wrapper(object):
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector, inp_dim, num_classes, stride, confidence, nms_thresh, device):
        self.yolo_detector = yolo_detector
        #self.yolo_detector.eval()
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.stride = stride
        self.conf_thres = confidence
        self.nms_iou_thres = nms_thresh
        self.device = device

        self.SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25], n_patch=1)
        self.SAC_processor.unet.load_state_dict(torch.load("defense_baseline/ckpt/coco_at.pth", map_location='cpu'))
        self.SAC_processor = self.SAC_processor.to(device)


    def __call__(self, img):
        
        x_processed, _, _ = self.SAC_processor([img.squeeze()], bpda=True, shape_completion=True)
        img = x_processed[0].unsqueeze(0)

        prediction = self.yolo_detector(img).data
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
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector, num_classes, confidence, nms_thresh, device):
        self.yolo_detector = yolo_detector
        #self.yolo_detector.eval()
        self.num_classes = num_classes
        self.conf_thres = confidence
        self.nms_iou_thres = nms_thresh
        self.device = device

        self.SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25], n_patch=1)
        self.SAC_processor.unet.load_state_dict(torch.load("defense_baseline/ckpt/coco_at.pth", map_location='cpu'))
        self.SAC_processor = self.SAC_processor.to(device)


    def __call__(self, img):
        
        x_processed, _, _ = self.SAC_processor([img.squeeze()], bpda=True, shape_completion=True)
        img = x_processed[0].unsqueeze(0)

        prediction = self.yolo_detector(img).data
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
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector, device):
        self.yolo_detector = yolo_detector
        self.device = device

        self.SAC_processor = PatchDetector(3, 1, base_filter=16, square_sizes=[125, 100, 75, 50, 25], n_patch=1)
        self.SAC_processor.unet.load_state_dict(torch.load("defense_baseline/ckpt/coco_at.pth", map_location='cpu'))
        self.SAC_processor = self.SAC_processor.to(device)


    def __call__(self, img):
        
        x_processed, _, _ = self.SAC_processor([img.squeeze()], bpda=True, shape_completion=True)
        img = x_processed[0]
        img = tensor_to_image(img)

        #prediction = self.yolo_detector.detect_image(img)
        image, boxes, scores, classes = self.yolo_detector.detect_image(img, draw_image=False)

        if boxes is None:
            return None
        output = np.concatenate((boxes, np.expand_dims(scores,axis=-1), np.expand_dims(classes,axis=-1)), axis=1)

        return output


def defense_model_v2(device="cuda", confidence=0.5, nms_thresh=0.4, stride=32, num_classes=80, inp_dim=416):

    device = torch.device(device)
    weightsfile = 'weights/yolo.weights'
    cfgfile = "cfg/yolo.cfg"

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model = model.to(device)
    model.eval()

    model = YOLOv2_wrapper(model, inp_dim, num_classes, stride, confidence, nms_thresh, device)

    return model


def defense_model_v3(device="cuda", confidence=0.5, nms_thresh=0.4, num_classes=80, inp_dim=416):

    device = torch.device(device)
    weightsfile = "weights/yolov3.weights"
    cfgfile = "cfg/yolov3.cfg"

    model = Darknet53(device, cfgfile)
    model.load_weights(weightsfile)
    model.net_info["height"] = inp_dim
    model = model.to(device)
    model.eval()

    model = YOLOv3_wrapper(model, num_classes, confidence, nms_thresh, device)

    return model


def defense_model_v4(device="cuda", confidence=0.5, nms_thresh=0.4, num_anchors=3, num_classes=80, inp_dim=416):

    CUDA = True
    all_classes = get_classes("data/coco.names")
    num_classes = len(all_classes)
    yolo = YOLOv4(num_classes, num_anchors)
    yolo = yolo.to(device)
    yolo.load_state_dict(torch.load("weights/pytorch_yolov4_1.pt"))
    yolo.eval()
    model = Decode(confidence, nms_thresh, (inp_dim, inp_dim), yolo, all_classes, CUDA)

    model = YOLOv4_wrapper(model, device)

    return model

