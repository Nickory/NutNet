import torch
import numpy as np
import torchvision.transforms as T

from darknet_v2 import Darknet
from darknet_v3 import Darknet as Darknet53
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
from util_model_img import predict_transform_v2, tensor_to_image, write_results, write_result


class Gradient(torch.nn.Module):
    r'''
    Compute the first-order local gradient
    '''

    def __init__(self) -> None:
        super().__init__()

        self.d_x = torch.nn.Conv2d(
            1, 1, kernel_size=(1, 2), bias=False)
        self.d_y = torch.nn.Conv2d(
            1, 1, kernel_size=(2, 1), bias=False)
        self.zero_pad_x = torch.nn.ZeroPad2d((0, 1, 0, 0))
        self.zero_pad_y = torch.nn.ZeroPad2d((0, 0, 0, 1))
        self.update_weight()

    def update_weight(self):
        first_order_diff = torch.FloatTensor([
            [1, -1],
        ])
        kernel_dx = first_order_diff.unsqueeze(0).unsqueeze(0)
        kernel_dy = first_order_diff.transpose(1, 0).unsqueeze(0).unsqueeze(0)

        self.d_x.weight = torch.nn.Parameter(kernel_dx).requires_grad_(False)
        self.d_y.weight = torch.nn.Parameter(kernel_dy).requires_grad_(False)

    def forward(self, img):
        batch_size = img.shape[0]
        img_aux = img.reshape(-1, img.shape[-2], img.shape[-1])
        img_aux.unsqueeze_(1)
        grad_x = self.d_x(img_aux)
        grad = self.zero_pad_x(grad_x).pow(2)
        grad_y = self.d_y(img_aux)
        grad += self.zero_pad_y(grad_y).pow(2)
        grad.sqrt_()
        grad.squeeze_(1)
        grad = grad.reshape(
            batch_size, -1, img_aux.shape[-2], img_aux.shape[-1])
        return grad


def get_mask(grad, window_size=15, stride=10, threshold=0.1, smoothing_factor=2.3) -> torch.Tensor:
        
    #stride = window_size - overlap

    grad_unfolded = torch.nn.functional.unfold(
        grad, window_size, stride=stride)
    mask_unfolded = torch.mean(
        grad_unfolded, dim=1, keepdim=True) > threshold
    mask_unfolded = mask_unfolded.repeat(1, grad_unfolded.shape[1], 1)
    mask_unfolded = mask_unfolded.float()
    mask_folded = torch.nn.functional.fold(
        mask_unfolded, grad.shape[2:], kernel_size=window_size, stride=stride)
    mask_folded = (mask_folded >= 1).float()
    grad *= mask_folded
    grad = torch.clamp(smoothing_factor * grad, 0, 1)

    return grad



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

        self.get_gradient = Gradient().to(self.device)


    def __call__(self, img):
        
        img_grad = self.get_gradient(img)
        max_grad = torch.amax(img_grad, dim=(2, 3), keepdim=True)
        min_grad = torch.amin(img_grad, dim=(2, 3), keepdim=True)
        img_grad = (img_grad - min_grad) / (max_grad - min_grad + 1e-7)

        grad_mask = get_mask(img_grad)
        img = img * (1 - grad_mask)

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

        self.get_gradient = Gradient().to(self.device)


    def __call__(self, img):
        
        img_grad = self.get_gradient(img)
        max_grad = torch.amax(img_grad, dim=(2, 3), keepdim=True)
        min_grad = torch.amin(img_grad, dim=(2, 3), keepdim=True)
        img_grad = (img_grad - min_grad) / (max_grad - min_grad + 1e-7)

        grad_mask = get_mask(img_grad)
        img = img * (1 - grad_mask)

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

        self.get_gradient = Gradient().to(self.device)


    def __call__(self, img):
        
        img_grad = self.get_gradient(img)
        max_grad = torch.amax(img_grad, dim=(2, 3), keepdim=True)
        min_grad = torch.amin(img_grad, dim=(2, 3), keepdim=True)
        img_grad = (img_grad - min_grad) / (max_grad - min_grad + 1e-7)

        grad_mask = get_mask(img_grad)
        img = img * (1 - grad_mask)
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

