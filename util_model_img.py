

import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np
import random
import pickle as pkl


############################################################################
##                       模型输入输出处理的一些函数                         ##
############################################################################

def letterbox_image(img, input_dim):
    """
    
    Description
    -----------
    将输入的图像缩放（长宽比不变）并填充到指定大小。
    
    Parameters
    ----------
    @img : (np.array)
        the input image with size [h, w, c].
    @input_dim : (int)
        the size of model input (416*416 for Yolo3).
    
    Returns
    -------
    @canvas : (np.array)
        the image with size [416, 416, c]
    """

    img_h, img_w = img.shape[0], img.shape[1]
    h, w = input_dim
    new_h = int(img_h * min(w / img_w, h / img_h))
    new_w = int(img_w * min(w / img_w, h / img_h))

    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((input_dim[1], input_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    
    return canvas



def load_classes(names_file):
    """
    Description
    -----------
    输入一个包含了分类名称的文件，输出分类组成的列表
    
    """
    
    file = open(names_file, 'r')
    names = file.read().split('\n')[:-1]
    
    return names



def get_ind(prediction, ori_index, conf_thresh=0.5, num_classes=80):
    """
    Description
    -----------
    从经过处理的输出中找到符合检测阈值的某一个分类的索引值。
    
    Parameters
    ----------
    @prediction : (Tensor)
        raw output from the model.
    @ori_index : (int)
        index of the class we want to hide.
    @conf_thresh : (float)
        threshold of object confidence.
    @num_classes : (int)
        the number of total classes.
    
    Returns
    -------
    @ind_nz : (list/np.array)
        index of prediction records whose obj confidence is greater than 0.5 and class is person.
    
    """
    
    
    # 1 * 10647 * 85
    prediction = prediction.cpu()
    conf_mask = (prediction[:, :, 4] > conf_thresh).float().unsqueeze(2)    
    
    max_v, max_i = torch.max(prediction[:, :, 5:5 + num_classes], 2)
    class_mask = (max_i == ori_index).float().unsqueeze(2)
    
    prediction = prediction * class_mask
    prediction = prediction * conf_mask
    
    ind_nz = torch.nonzero(prediction[0, :, 4]) # k * 1
    
    if ind_nz.shape[0] == 0:
        return [0]
    else:

        ind_out = np.zeros(shape=[3, ind_nz.shape[0]])
        ind_out[0, :] = ind_nz[:, 0]
        ind_out[1, :] = max_i[:, ind_nz[:, 0]]
        ind_out[2, :] = prediction.data[0, ind_nz[:, 0], 4]

        ind_out = ind_out[:, (-ind_out[2, :]).argsort()]
        ind_nz = ind_out[0, :].astype(np.int32)
        return ind_nz


def bbox_iou(box1, box2):
    """
    Description
    -----------
    计算两个 bounding box 的 IoU。
    
    Parameters
    ----------
    @box1 : (Tensor)
        the left-top and right-bottom coordinates of the box with size [1, 4].
    @box2 : (Tensor)
        the left-top and right-bottom coordinates of the box with size [1, 4].
        
    Returns
    -------
    @iou : (Tensor)
        the union over intersection of two boxes with size [1].
    """
    
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b2_y2, b2_y2)
    
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def unique(tensor):
    """
    Description
    -----------
    借助 np.unique()，输入一个张量，返回张量中的唯一值。

    """
    
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)

    return tensor_res


def predict_transform_v2(prediction, inp_dim, anchors, num_classes, stride=32, confidence=0.5, CUDA=True):
    batch_size = prediction.size(0)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    #print(prediction.shape)
    prediction = prediction.transpose(1,2).contiguous()
    #print(prediction.shape)
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    #print(prediction.shape)
    
    
    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    
    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)
    
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    #Softmax the class scores
    prediction[:, :, 5:5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:, :, 5:5 + num_classes])).data

    prediction[:, :, :4] *= stride
    '''
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    
    
    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    
    box = prediction[ind_nz[0], ind_nz[1]]
    
    box_a = box.new(box.shape)
    box_a[:, 0] = (box[:, 0] - box[:, 2] / 2)
    box_a[:, 1] = (box[:, 1] - box[:, 3] / 2)
    box_a[:, 2] = (box[:, 0] + box[:, 2] / 2) 
    box_a[:, 3] = (box[:, 1] + box[:, 3] / 2)
    box[:, :4] = box_a[:, :4]
    
    prediction[ind_nz[0], ind_nz[1]] = box
    
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask
    ''' 
    
    return prediction


def write_result(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    """
    Description
    -----------
    
    
    Parameters
    ----------
    @prediction : (Tensor)
    @confidence : (float)
    @num_classes : (int)
    @nms : (bool)
    @nms_conf : (float)
    
    Returns
    -------
    @output : (Tensor)
    
    """
    
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_a[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_a[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_a[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]

        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)

        seq = (image_pred[:, :5], max_conf, max_conf_score)

        image_pred = torch.cat(seq, 1)
        non_zero_ind = torch.nonzero(image_pred[:, 4])
        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        
        try:
            img_classes = unique(image_pred_[:, -1])
        except:
            continue

        for clss in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == clss).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            if nms:
                for i in range(idx):
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
                    except IndexError:
                        break

                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] = image_pred_class[i+1:] * iou_mask

                    non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
            
    try:
        return output
    except:
        return 0


def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):

    #print(prediction.shape)
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    try:
        ind_nz = torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_a[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_a[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_a[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    prediction[:, :, :4] = box_a[:, :, :4]
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]

        
        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores 
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        
        # Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        except:
            continue

        try:
            image_pred_[:,-1]
        except IndexError:
            image_pred_ = image_pred_.unsqueeze(0)
        
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])        
        
                
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind]

        
            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            try:
                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]
            except IndexError:
                image_pred_class = image_pred_class.unsqueeze(0)
                # conf_sort_index = torch.sort(image_pred_class[4], descending=True)[1]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)
                    
                    
            
            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    
    try:
        return output
    except:
        return 0


############################################################################
##                       图像和张量转换的一些函数                           ##
############################################################################


def tensor_to_image(tensor):
    """
    Description
    -----------
    将用于输入模型的张量转换为 numpy 的数组（给 cv2 用的）
    
    Parameters
    ----------
    @tensor : (Tensor)
        the model input tensor with size [b, c, h, w]
    
    Returns
    -------
    @image : (np.array)
        the image with size [h, w, c(BGR)] from tensor
    """
        
    tensor = tensor.cpu().squeeze()
    tensor = tensor*255
    try:
        image = tensor.data.numpy()
    except RuntimeError:
        image = tensor.numpy()
    image = image.transpose(1,2,0)

    image = image[:,:,::-1].astype(np.uint8).copy()

    return image


def image_to_tensor(image):

    image = image[:, :, ::-1].transpose((2, 0, 1)).copy()
    tensor = torch.from_numpy(image).float().div(255.0).unsqueeze(0)

    return tensor

