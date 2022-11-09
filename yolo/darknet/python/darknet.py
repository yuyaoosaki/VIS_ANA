from ctypes import *
import math
import random
import os
import numpy as np
import pandas as pd
from cmath import inf

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL("libdarknet.so", RTLD_GLOBAL)
# lib = CDLL(os.path.join(os.getcwd(),"yolo_9000/darknet/libdarknet.so"), RTLD_GLOBAL)
lib = CDLL(os.path.join(os.getcwd(),"../libdarknet.so"), RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

# 矩形aと、複数の矩形bのIoUを計算
def iou_np(a, b, a_area, b_area):
    # aは1つの矩形を表すshape=(4,)のnumpy配列
    # array([xmin, ymin, xmax, ymax])
    # bは任意のN個の矩形を表すshape=(N, 4)のnumpy配列
    # 2次元目の4は、array([xmin, ymin, xmax, ymax])
    
    # a_areaは矩形aの面積
    # b_areaはbに含まれる矩形のそれぞれの面積
    # shape=(N,)のnumpy配列。Nは矩形の数
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou_np = intersect / (a_area + b_area - intersect)
    return iou_np

# 自分用
def myNMS(bboxes, scores, classes, iou_threshold=0.5):
    # bboxesの矩形の面積を一気に計算
    areas = (bboxes[:,2] - bboxes[:,0] + 1) \
            * (bboxes[:,3] - bboxes[:,1] + 1)
    
    N = len(classes)
    df = pd.DataFrame(np.concatenate([classes.reshape(N,1), scores.reshape(N,1), areas.reshape(N,1), bboxes], 1), columns=['class', 'score', 'area', 'x_min', 'y_min', 'x_max', 'y_max'])
    df = df.astype({'class':str, 'score':float, 'area':float, 'x_min':float, 'y_min':float, 'x_max':float, 'y_max':float})
    
    df_classes = df.groupby('class')
    bboxes = None
    scores = np.array([])
    classes = np.array([])
    result = []
    for cls, df_cls in df_classes:
        flag = True
        while flag:
            flag = False
            i = 0
            while len(df_cls) > 1 and len(df_cls) > i:
                ara = df_cls['area'].values
                bb = df_cls[['x_min', 'y_min', 'x_max', 'y_max']].values
                iou = iou_np(bb[i], np.delete(bb,i,0), ara[i], np.delete(ara,i))
                overlap_idx = np.where(np.insert(iou, i, -inf) >= iou_threshold)[0]
                overlap_idxPULSi = np.append(overlap_idx, i)
                
                if len(overlap_idx) > 0:
                    flag = True
                    xmin_new, ymin_new = df_cls.iloc[overlap_idxPULSi][['x_min', 'y_min']].min()
                    xmax_new, ymax_new = df_cls.iloc[overlap_idxPULSi][['x_max', 'y_max']].max()
                    ara_new = (xmax_new - xmin_new + 1) * (ymax_new - ymin_new + 1)
                    score_new = df_cls.iloc[overlap_idxPULSi]['score'].max()
                    df_cls = df_cls.drop(df_cls.index[overlap_idxPULSi])
                    df_cls = df_cls.append({'class':cls, 'score':score_new, 'area':ara_new, 'x_min':xmin_new, 'y_min':ymin_new, 'x_max':xmax_new, 'y_max':ymax_new}, ignore_index=True)
                    i = 0
                else:
                    i += 1
                    continue

        N = len(df_cls)
        x = ((df_cls['x_min'].values + df_cls['x_max'].values) / 2)
        y = ((df_cls['y_min'].values + df_cls['y_max'].values) / 2)
        w = (df_cls['x_max'].values - df_cls['x_min'].values)
        h = (df_cls['y_max'].values - df_cls['y_min'].values)
        df_cls = df_cls.assign(x=x, y=y, w=w, h=h)

        for row in df_cls.itertuples():
            result.append((row[1], row[2], (row[7], row[8], row[9], row[10])))
    return result

def detect(net, meta, image, object_idxes, thresh=.1, hier_thresh=.5, nms=.45, key='confidence'):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    # ②-1 objectList(検出結果)をclasses(N,), scores(N,), bboxes(N,4)に分割
    classes = np.array([])
    scores = np.array([])
    bboxes = None
    for j in range(num):
        for i in range(meta.classes):
        # for i in object_idxes:
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i].decode(), dets[j].prob[i], (b.x, b.y, b.w, b.h)))
                classes = np.append(classes, meta.names[i].decode())
                scores = np.append(scores, dets[j].prob[i])
                bboxes = np.array([(b.x, b.y, b.w, b.h)]) if bboxes is None else np.append(bboxes, np.array([(b.x, b.y, b.w, b.h)]), axis=0)

    # ②-2 bboxesの中身を(x, y, w, h)から(xmin, ymin, xmax, ymax)に変更
    N = len(classes)
    if N > 1:
        x_min = (bboxes[:, 0] - bboxes[:, 2] / 2).reshape(N,1)
        y_min = (bboxes[:, 1] - bboxes[:, 3] / 2).reshape(N,1)
        x_max = (bboxes[:, 0] + bboxes[:, 2] / 2).reshape(N,1)
        y_max = (bboxes[:, 1] + bboxes[:, 3] / 2).reshape(N,1)
        bboxes = np.concatenate([x_min, y_min, x_max, y_max], 1)
        res = myNMS(bboxes, scores, classes, iou_threshold=0.01)
    if key == 'confidence':
        res = sorted(res, key=lambda x: -x[1])
    elif key == 'area':
        res = sorted(res, key=lambda x: x[2][2]*x[2][3])
    free_image(im)
    free_detections(dets, num)
    return res
    
class yoloVIDVIP():
    def __init__(self):
        self.net = load_net(b"yolo/yolov2-tiny-vidvip.weights/yolov2-tiny-vidvip.cfg", b"yolo/yolov2-tiny-vidvip.weights/yolov2-tiny-vidvip.weights", 0)
        self.meta = load_meta(b"yolo/darknet/cfg/combine9k.data")
        self.names = [self.meta.names[i].decode() for i in range(self.meta.classes)]
        
    def detect(self, image, object_names, max_num=-1, thresh=.1, key='confidence'):
        object_idxes = [self.names.index(object_name) for object_name in object_names]
        result = detect(self.net, self.meta, image, object_idxes, thresh=thresh, key=key)
        max_num = len(result) if max_num<0 else max_num
        result = result[:max_num]
        return result