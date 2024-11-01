import torch
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from math import ceil
import torchvision.transforms as T

from darknet_v2 import Darknet
from darknet_v3 import Darknet as Darknet53
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
from util_model_img import predict_transform_v2, tensor_to_image, write_result, write_results


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


class ObjSeekerModel(object):

    def __init__(self, base_model, device):
        super(ObjSeekerModel, self).__init__()

        # base detector setup (unrelated to the defense)
        self.base_model = base_model # base model
        self.base_conf_thres = 0.6 #the confidence threshold for base prediction boxes that will be used for box pruning 

        # mask parameters
        self.num_line = 30 #number of lines in one dimension
        self.masked_conf_thres = 0.8 #only retained masked prediction boes whose confidence exceeds this threshold
      
        # box pruning parameters
        self.ioa_prune_thres = 0.6 # discard masked prediction boxes whose ioa with base prediction box exceeds this threshold
        self.dbscan = DBSCAN(eps=0.1, min_samples=1,metric='precomputed') #dbscan instance for overlap boxes pruning
        self.match_class = True # whether consider class labels

        #misc
        self.device = device


    def __call__(self, img, base_output_precomputed=None, raw_masked_output_precomputed=None):
        # defense inference forward pass
        # img: input image [B,3,H,W] for yolo; or a dict for img and its meta information for mmdet
        # base_output_precomputed. precomputed base detector detection results (on the original imaeges). If it is None, compute the results on the fly
        # raw_masked_output_precomputed. precomputed detection results on masked image. If it is None, compute the results on the fly

        # return a list of detection results (the same format as the output of YOLO_Wrapper). one list element corresponds to one image
        return self.ioa_inference(img, base_output_precomputed=base_output_precomputed, raw_masked_output_precomputed=raw_masked_output_precomputed)


    def ioa_inference(self, img,base_output_precomputed=None, raw_masked_output_precomputed=None,save_dir=None,paths=None,names=None):
        # the same usage as __call__()

        # setup
        if isinstance(img,torch.Tensor): #yolo input is tensor
            num_img,_,height,width = img.shape
        else: # the input for mmdet is not torch.Tensor
            num_img,_,height,width = img['img'][0].shape

        # get coordinates of partition points
        idx_list = self.get_mask_idx(height,width)
        # for each image, get filtered masked predictions
        defense_output = [torch.zeros((0,6)).to(self.device) for i in range(num_img)] # to hold the final output 
        masked_output = [[] for i in range(num_img)] # to hold masked boxes from different masked images
        
        if base_output_precomputed is None:
            base_output_precomputed = self.base_model(img, conf_thres=self.base_conf_thres)
        #print(base_output_precomputed)
        #print([pred for pred in base_output_precomputed])
        base_output = [pred[pred[:,4]>self.base_conf_thres].to(self.device) for pred in base_output_precomputed] # each is xyxy conf cls # used as part of the output
        if self.num_line <=0: #vanilla inference
            return base_output

        for img_i in range(num_img):
            clip_coords(base_output[img_i], (height, width)) # clip negative values
            if self.match_class: # add class offset so that boxes of different classes can never have intersection
                base_output[img_i][:,:4] = base_output[img_i][:,:4]+ base_output[img_i][:, 5:6] * 4096            

        # next, we are going to gather masked box and perform box pruning 
        for ii,jj in idx_list: # for each partition point
            mask_list = self.gen_mask(ii,jj,height,width) if raw_masked_output_precomputed is None else list(range(4))
            
            for mask_i,mask in enumerate(mask_list): # for each mask (4 masks in total)
                if raw_masked_output_precomputed is None: # no precomputed detections. do detection now!
                    masked_output_ = self.base_model(img, mask=mask, conf_thres=self.masked_conf_thres)  # each is xyxy conf cls 
                
                for img_i in range(num_img): # for each image
                    if raw_masked_output_precomputed is None: # get masked boxes for this image, depending on if precomputed detection is available
                        masked_boxes = masked_output_[img_i].to(self.device)
                    else:
                        masked_boxes = raw_masked_output_precomputed[img_i][(ii,jj)][mask_i].to(self.device)
                        masked_boxes = masked_boxes[masked_boxes[:,4]>self.masked_conf_thres]

                    base_boxes = base_output[img_i]
                    clip_coords(masked_boxes, (height, width)) # clip negative values
                    if len(masked_boxes) >0:
                        if self.match_class: # class offset
                            masked_boxes[:,:4] = masked_boxes[:,:4] + masked_boxes[:, 5:6] * 4096
                        # calculate ioa, and filter boxes
                        if len(base_boxes)>0: ###False:
                            ioa = self.box_ioa(masked_boxes[:,:4], base_boxes[:,:4]) 
                            ioa = torch.max(ioa,dim=1)[0]
                            fil = ioa < self.ioa_prune_thres
                            masked_boxes = masked_boxes[fil] 
                        masked_output[img_i].append(masked_boxes)

        try:
            for img_i in range(num_img):
                masked_boxes = masked_output[img_i]
                masked_boxes = torch.cat(masked_boxes) if len(masked_boxes) > 0 else torch.zeros((0,6)).to(self.device)
                if len(masked_boxes)>1: ###False:
                    masked_boxes = self.unionize_cluster(masked_boxes) # box unionizing
                base_boxes = base_output[img_i]
                if self.match_class: # remove class offset
                    masked_boxes[:,:4] = masked_boxes[:,:4] - masked_boxes[:, 5:6] * 4096
                    base_boxes[:,:4] = base_boxes[:,:4] - base_boxes[:, 5:6] * 4096
                defense_output[img_i] = torch.cat([base_boxes,masked_boxes])
                #defense_output[img_i][:, :4] = defense_output[img_i][:, :4] / height
        except ValueError:
            defense_output = [0]
            

        return defense_output[0]


    def get_raw_masked_boxes(self, img):
        # generate masked detection boxes (i.e., generate precomputed masked prediction). 
        # then we can dump/load the masked boxes to save computation in our experiment
        if isinstance(img,torch.Tensor):
            num_img,_,height,width = img.shape
        else:
            num_img,_,height,width = img['img'][0].shape

        raw_masked_output = [defaultdict(list) for i in range(num_img)] 
        idx_list = self.get_mask_idx(height,width)
        for ii,jj in idx_list:
            mask_list = self.gen_mask(ii,jj,height,width) 
            for mask in mask_list:
                masked_output = self.base_model(img,mask=mask,conf_thres=self.masked_conf_thres)  # each is xyxy conf cls # 
                for i,pred in enumerate(masked_output):
                    if pred is None:
                        raw_masked_output[i][(ii,jj)].append(None)
                    else:
                        raw_masked_output[i][(ii,jj)].append(pred.detach().cpu())

        return raw_masked_output


    def unionize_cluster(self,overlap_boxes):
        # box unionizing
        # cluster overlap_boxes, and merge each cluster into one box

        # calculate "distances" between boxes; the distances is based on ioa; the distance matrix will be used for cluster
        ioa = self.box_ioa(overlap_boxes[:,:4],overlap_boxes[:,:4])
        # calculate pair-wise distance
        distance = 1-torch.maximum(ioa,ioa.T)
        distance = distance.cpu().numpy()

        # dbscan clustering
        cluster = self.dbscan.fit_predict(distance)
        num_cluster = np.max(cluster)+1
        cluster = torch.from_numpy(cluster).to(self.device)

        unionized_boxes = torch.zeros((num_cluster,6)).to(self.device) #xyxy conf cls

        for cluster_i in range(num_cluster):
            boxes = overlap_boxes[cluster==cluster_i]
            unionized_boxes[cluster_i,:2] = torch.min(boxes[:,:2],dim=0)[0]
            unionized_boxes[cluster_i,2:4] = torch.max(boxes[:,2:4],dim=0)[0]
            unionized_boxes[cluster_i,4] = torch.max(boxes[:,4]) # take the highest confidence
            unionized_boxes[cluster_i,5] = boxes[0,5] # take a "random" class

        return unionized_boxes


    def gen_mask(self,ii,jj,height,width):

        #generate 4 mask tensors for location (ii,jj) on a (height,width) image

        mask_list = []
        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,:ii,:] = 0
        mask_list.append(mask)
        
        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,ii:,:] = 0
        mask_list.append(mask)

        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,:,:jj] = 0
        mask_list.append(mask)
        
        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,:,jj:] = 0
        mask_list.append(mask)

        return mask_list


    def get_mask_idx(self,height,width):
        # generate a list of (ii,jj) coordinates
        if self.num_line<=0:
            return []
        h_stride = ceil(height / (self.num_line+1))
        w_stride = ceil(width / (self.num_line+1))
        ii_idx = list(range(h_stride,height,h_stride))
        jj_idx = list(range(w_stride,width,w_stride))

        idx_list = zip(ii_idx,jj_idx)
        return idx_list


    def box_ioa(self, box1, box2):
        # slighlt modify the code for box_iou to calculate box_ioa
        # the output is inter / area1
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None])


class YOLOv2_wrapper(object):
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector, inp_dim, num_classes, stride):
        self.yolo_detector = yolo_detector
        #self.yolo_detector.eval()
        self.inp_dim = inp_dim
        self.num_classes = num_classes
        self.stride = stride
    def __call__(self, img, conf_thres, mask=None, nms_iou_thres=0.65):
        # do vanilla object detection.
        if mask is None:
            prediction = self.yolo_detector(img).detach()
        else:
            prediction = self.yolo_detector(img*mask).detach()
        # the detection result is a list of tensor; one tensor for one image
        # the format is `xyxy conf cls` # the coordinates is for the resize images, not for the original image size

        prediction = predict_transform_v2(prediction, self.inp_dim, self.yolo_detector.anchors, self.num_classes, self.stride, conf_thres, True)
        output = write_results(prediction, conf_thres, self.num_classes, nms=True, nms_conf=nms_iou_thres)
        
        if type(output) is int or output.size(-1) == 86:
            output = torch.zeros(1, 6)
        else:
            output[:, 5] = output[:, 5]*output[:, 6]
            output[:, 6] = output[:, 7]
            output = output[:, 1:-1]

        return [output]


class YOLOv3_wrapper(object):
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector, num_classes):
        self.yolo_detector = yolo_detector
        #self.yolo_detector.eval()
        self.num_classes = num_classes

    def __call__(self, img, conf_thres, mask=None, nms_iou_thres=0.65):
        # do vanilla object detection.
        if mask is None:
            prediction = self.yolo_detector(img).detach()
        else:
            prediction = self.yolo_detector(img*mask).detach()

        output = write_result(prediction, conf_thres, self.num_classes, nms=True, nms_conf=nms_iou_thres)

        if type(output) is int or output.size(-1) == 86:
            output = torch.zeros(1, 6)
        else:
            output[:, 5] = output[:, 5]*output[:, 6]
            output[:, 6] = output[:, 7]
            output = output[:, 1:-1]

        return [output]


class YOLOv4_wrapper(object):
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector):
        self.yolo_detector = yolo_detector
    
    def __call__(self, img, conf_thres, mask=None, nms_iou_thres=0.65):
        # do vanilla object detection.
        if mask is None:
            image, boxes, scores, classes = self.yolo_detector.detect_image(tensor_to_image(img), draw_image=False)
        else:
            image, boxes, scores, classes = self.yolo_detector.detect_image(tensor_to_image(img*mask), draw_image=False)

        if boxes is None:
            output = torch.zeros(1, 6)
        else:
            output = torch.from_numpy(np.concatenate((boxes, np.expand_dims(scores,axis=-1), np.expand_dims(classes,axis=-1)), axis=1))

        return [output]


def defense_model_v2(device="cuda"):

    stride = 32
    #CUDA = torch.cuda.is_available()
    device = torch.device(device)

    inp_dim = 416
    num_classes = 80
    weightsfile = 'weights/yolo.weights'
    cfgfile = "cfg/yolo.cfg"

    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model = model.to(device)
    model.eval()

    model_ = YOLOv2_wrapper(model, inp_dim, num_classes, stride)
    objseeker = ObjSeekerModel(model_, device)


    return objseeker


def defense_model_v3(device="cuda", confidence=0.5, nms_thresh=0.4, num_classes=80, inp_dim=416):

    device = torch.device(device)
    weightsfile = "weights/yolov3.weights"
    cfgfile = "cfg/yolov3.cfg"

    model = Darknet53(device, cfgfile)
    model.load_weights(weightsfile)
    model.net_info["height"] = inp_dim
    model = model.to(device)
    model.eval()

    model_ = YOLOv3_wrapper(model, num_classes)
    objseeker = ObjSeekerModel(model_, device)

    return objseeker


def defense_model_v4(device="cuda", confidence=0.5, nms_thresh=0.4, num_anchors=3, num_classes=80, inp_dim=416):

    CUDA = True
    all_classes = get_classes("data/coco.names")
    num_classes = len(all_classes)
    yolo = YOLOv4(num_classes, num_anchors)
    yolo = yolo.to(device)
    yolo.load_state_dict(torch.load("weights/pytorch_yolov4_1.pt"))
    yolo.eval()
    model = Decode(confidence, nms_thresh, (inp_dim, inp_dim), yolo, all_classes, CUDA)

    model_ = YOLOv4_wrapper(model)
    objseeker = ObjSeekerModel(model_, device)

    return objseeker
