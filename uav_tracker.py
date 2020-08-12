"""
  author: Sierkinhane
  since: 2019-2-17 15:03:22
  description: integerate pose estimation, face&human detection and indentity identification.
"""

import argparse
import os
import time

import cv2

import numpy as np

import torch

from utils.utils import *
from utils.datasets import *
from yolo_models import *
from face_models import Resnet50FaceModel, Resnet18FaceModel

from tools.commute import send_msg, init_commu


# tracker
class Tracker(object):
    def __init__(self, args):
        self.args = args

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        """
        human&face detection
        """
        self.boxSize = 384
        self.yolov3 = "./cfg/yolov3orihf.cfg"
        self.dataConfigPath = "cfg/coco.data"
        self.weightsPath_hf = "weights/latest_h_f.pt"
        self.confThres = 0.5
        self.nmsThres = 0.45
        self.dataConfig = parse_data_config(self.dataConfigPath)
        self.classes = load_classes(self.dataConfig['names'])
        """
        indentification
        """
        self.weightsPath_c = "./weights/res18_aug_market_cuhk.pth.tar"
        self.suspected_bbx = []
        self.infer_shape = (96, 128)
        # replay embedded vector buffer: store 10 timestep of embedded vector of target
        self.target_vector_buffer = np.zeros((10, 512))
        self.target_bbx = np.array([])
        self.bufferSize = 10
        self.bufferPointer = 0
        self.counter = 0
        self.way2 = True
        """ other params"""
        self.model_d = self.getHFDModel()
        self.model_c = self.getCenterModel()
        self.initialize = True
        self.video_init()
        self.tcp_init()
        self.clear_target()

    def update_target(self, bbox, shape):
        self.cur_target = (bbox, shape)
        self.have_target = True

    def clear_target(self):
        self.have_target = False

    def check_target(self):
        if self.have_target:
            return True
        return False

    def send_target(self):
        if self.check_target():
            width = self.cur_target[1][1]
            height = self.cur_target[1][0]
            send_msg(self.sock, True, self.cur_target[0], (width, height))
        else:
            send_msg(self.sock, False, [0, 0, 0, 0], (0, 0))

    def tcp_init(self):
        if self.args.tcp_enable:
            self.sock = init_commu(self.args.host_ip, self.args.port)

    def video_init(self):
        self.frame_rate_ratio = args.frame_interval
        self.video_path = args.video_path
        if self.args.cam != -1:
            print("Using webcam " + str(self.args.cam))
            self.cap = cv2.VideoCapture(self.args.cam)
        else:
            self.cap = cv2.VideoCapture(self.video_path)

        if self.args.cam != -1:
            ret, frame = self.cap.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[1]
            self.im_height = frame.shape[0]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.cap.open(self.video_path)
            self.im_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.cap.isOpened()
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_result_path = os.path.join(self.args.save_path,
                                                 "results.avi")
            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.result_writer = cv2.VideoWriter(
                self.save_result_path, fourcc, 20,
                (self.im_width, self.im_height))

            if self.args.save_origin:
                self.save_source_path = os.path.join(self.args.save_path,
                                                     "source_video.avi")
                self.source_writer = cv2.VideoWriter(
                    self.save_source_path, fourcc, 20,
                    (self.im_width, self.im_height))

            print("Save results to {}".format(self.args.save_path))

    def getCenterModel(self):

        # model = Resnet50FaceModel
        model = Resnet18FaceModel
        model = model(False).to(self.device)
        checkpoint = torch.load(self.weightsPath_c)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

        return model

    def getHFDModel(self):

        model = Darknet(self.yolov3, self.boxSize)
        model.load_state_dict(torch.load(self.weightsPath_hf)['model'])
        model.to(self.device).eval()

        return model

    def getPoseModel(self):

        model = cascaded_pose_net_dev.PoseModel(cfg_path=self.yoloBase)
        model.load_state_dict(torch.load(self.weightsPath))
        # model = torch.nn.DataParallel(model)
        model.to(self.device).eval()

        return model

    def normalization(self, img, resize=False):
        if resize:
            # print(img.shape)
            h, w = img.shape[:2]
            img = cv2.resize(img, (0, 0),
                             fx=self.infer_shape[0] / w,
                             fy=self.infer_shape[1] / h,
                             interpolation=cv2.INTER_CUBIC)

        return img.astype(np.float32) / 255.

    def resizeRequested(self, img, height=96, width=96):

        height_, width_ = img.shape[:2]
        return cv2.resize(img, (0, 0),
                          fx=width / width_,
                          fy=height / height_,
                          interpolation=cv2.INTER_CUBIC)

    def iou_fillter(self):
        """Compute IoU between detect box and gt boxes

            Parameters:
            ----------
            box: numpy array , shape (4, ): x1, y1, x2, y2
                input box
            boxes: numpy array, shape (n, 4): x1, y1, x2, y2
                input ground truth boxes
        """
        # box = (x1, y1, x2, y2)
        box = self.target_bbx[:]
        # print(box)
        boxes = np.array(self.suspected_bbx)
        if len(boxes) == 0 or len(box) == 0:
            return
        # print(boxes)
        box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] +
                                                  1)

        # abtain the offset of the interception of union between crop_box and gt_box
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (box_area + area - inter)
        # select ovr > 0.4
        thre_ovr_idex = np.where(ovr > 0.4)
        # update boxes
        u_boxes = boxes[thre_ovr_idex]
        # update ovr
        ovr = ovr[thre_ovr_idex]

        if len(u_boxes) > 3:
            # return the top3 ovr index
            top3_index = np.argsort(ovr)[-3:]
            self.suspected_bbx = u_boxes[top3_index]
        elif len(u_boxes) == 1:
            self.suspected_bbx = u_boxes
        elif len(u_boxes) == 0:
            # 镜头突然切换，iou为0，对所有预测框筛选，得出目标
            # 目标原先的bbx失去跟踪意义，清空
            self.way2 = True
            self.target_bbx = np.array([])
            self.suspected_bbx = boxes
        # print(self.suspected_bbx)

    def indentification(self, img, canvas, model):

        imgs = []
        ori = img.copy()

        if self.counter != 0:
            self.iou_fillter()

        if self.counter == 0:
            query_img = self.query_img
            query_img = self.normalization(query_img, resize=True)

            query_img = torch.from_numpy(query_img.transpose(2, 0,
                                                             1)).unsqueeze(0)
            query_img = query_img.to(self.device)
            _, embeddings = model(query_img)
            embeddings = embeddings.cpu().detach().numpy()
            self.target_vector_buffer[self.bufferPointer, :] = embeddings
            self.bufferPointer += 1

            # self.target_bbx = np.append(self.target_bbx, self.suspected_bbx[0])
            self.counter = 1
        else:

            for bbx in self.suspected_bbx:
                img = ori[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2]), :]
                img = self.normalization(img, resize=True)

                img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
                imgs.append(img)
                # img = self.transform_for_infer(self.infer_shape)(img)
                # imgs.append(img.unsqueeze(0))

            if len(imgs) != 0:
                imgs = torch.cat(imgs, 0)
                imgs = imgs.to(self.device)
                # print(imgs.shape)
                # tic = time.time()
                _, embeddings = model(imgs)
                # toc = time.time()
                # print(toc-tic)
                embeddings = embeddings.cpu().detach().numpy()  # (3, 512)

                distance = np.zeros((1, len(
                    self.suspected_bbx)))  # (1, 3) 3--bbox 10--vector buffer
                if self.bufferPointer < 19:
                    for i in range(self.bufferPointer):
                        distance += np.sum((embeddings - np.expand_dims(
                            self.target_vector_buffer[i, :], axis=0))**2,
                                           axis=1)
                    distance /= self.bufferPointer
                else:
                    for i in range(self.bufferSize):
                        distance += np.sum((embeddings - np.expand_dims(
                            self.target_vector_buffer[i, :], axis=0))**2,
                                           axis=1)
                    distance /= self.bufferSize

                # distance = np.squeeze(distance)
                # print(distance)

                # 1. 设定阈值 < 0.4
                # index = np.where(distance < 0.4)
                # 2. 找到空间距离最小的bbox
                index = np.argmin(distance[0])
                if self.way2:
                    if distance[0][index] < 0.6:
                        if self.bufferPointer > 9:
                            self.bufferPointer = 0

                        self.target_vector_buffer[
                            self.bufferPointer, :] = embeddings[index, :]
                        self.bufferPointer += 1

                        x1, y1, x2, y2 = self.suspected_bbx[index]
                        # 更新target的bbx
                        # print(self.target_bbx)
                        # print(self.suspected_bbx[index])
                        self.target_bbx = self.suspected_bbx[index]
                        label = 'Target %.2f' % distance[0][index]
                        plot_one_box([x1, y1, x2, y2],
                                     canvas,
                                     label=label,
                                     color=(0, 255, 170))
                        self.way2 = False
                        self.update_target([x1, y1, x2, y2], canvas.shape)
                else:
                    if distance[0][index] < 0.4:
                        if self.bufferPointer > 9:
                            self.bufferPointer = 0
                        self.target_vector_buffer[
                            self.bufferPointer, :] = embeddings[index, :]
                        self.bufferPointer += 1

                        x1, y1, x2, y2 = self.suspected_bbx[index]
                        # 更新target的bbx
                        # print(self.target_bbx)
                        # print(self.suspected_bbx[index])
                        self.target_bbx = self.suspected_bbx[index]
                        label = 'Target %.2f' % distance[0][index]
                        plot_one_box([x1, y1, x2, y2],
                                     canvas,
                                     label=label,
                                     color=(0, 255, 170))
                        self.update_target([x1, y1, x2, y2], canvas.shape)

        return canvas

    def humanDetector(self, img, model):

        ori = img.copy()
        img, _, _, _ = resize_square(img,
                                     height=self.boxSize,
                                     color=(127.5, 127.5, 127.5))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img = self.normalization(img)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        img_detections = []
        with torch.no_grad():
            pred = model(img)
            pred = pred[pred[:, :, 4] > self.confThres]

            if len(pred) > 0:
                detections = non_max_suppression(pred.unsqueeze(0),
                                                 self.confThres, self.nmsThres)
                img_detections.extend(detections)
            else:
                detections = np.array([])

        if len(detections) != 0:

            # The amount of padding that was added
            pad_x = max(ori.shape[0] - ori.shape[1],
                        0) * (self.boxSize / max(ori.shape))
            pad_y = max(ori.shape[1] - ori.shape[0],
                        0) * (self.boxSize / max(ori.shape))
            # Image height and width after padding is removed
            unpad_h = self.boxSize - pad_y
            unpad_w = self.boxSize - pad_x

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * ori.shape[0]
                box_w = ((x2 - x1) / unpad_w) * ori.shape[1]
                y1 = (((y1 - pad_y // 2) / unpad_h) *
                      ori.shape[0]).round().item()
                x1 = (((x1 - pad_x // 2) / unpad_w) *
                      ori.shape[1]).round().item()
                x2 = (x1 + box_w).round().item()
                y2 = (y1 + box_h).round().item()
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
                label = '%s %.2f' % (self.classes[int(cls_pred)], conf)
                color = [(255, 85, 0), (0, 255, 170)]
                if int(cls_pred) == 0:
                    self.suspected_bbx.append([x1, y1, x2, y2])
                    if self.initialize:
                        instance = ori[int(y1):int(y2), int(x1):int(x2), :]
                        cv2.namedWindow('detected instance', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('detected instance',
                                         instance.shape[1], instance.shape[0])
                        cv2.imshow('detected instance', instance)
                        cv2.waitKey(1)
                        skip = False
                        while True:
                            ipt = input(
                                'is this the target(y for yes and n for no and s for skip frame):'
                            )
                            if ipt == 'y':
                                self.query_img = instance
                                print('target saved!')
                                self.initialize = False
                                break
                            elif ipt == 'n':
                                print('go on initialization!')
                                break
                            elif ipt == 's':
                                print(
                                    'skip this frame and go on initialization!'
                                )
                                skip = True
                                break
                            else:
                                print('unknown format!')
                        if skip:
                            break
                    else:
                        plot_one_box([x1, y1, x2, y2],
                                     ori,
                                     label=label,
                                     color=color[int(cls_pred)])

        return ori

    def tracker_init(self):
        cv2.namedWindow("tracker init", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracker init", args.display_width,
                         args.display_height)
        idx_frame = 0
        cap = self.cap
        while cap.grab():
            idx_frame += 1
            if idx_frame % self.frame_rate_ratio == 0:
                _, frame = cap.retrieve()
                cv2.imshow("tracker init", frame)
                cv2.waitKey(1)
                self.humanDetector(frame, self.model_d)
            if not self.initialize:
                print('initialization done!')
                break

        cv2.destroyWindow("tracker init")

    def run(self):
        cap = self.cap
        idx_frame = 0
        cv2.namedWindow("tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("tracking", self.args.display_width,
                         self.args.display_height)
        while cap.grab():
            idx_frame += 1
            if idx_frame % self.frame_rate_ratio == 0:
                _, frame = cap.retrieve()
                if self.args.save_origin:
                    self.source_writer.write(frame)
                tic = time.time()
                canvas = frame
                canvas = self.humanDetector(frame, self.model_d)
                self.clear_target()
                canvas = self.indentification(frame, canvas, self.model_c)
                if self.args.tcp_enable:
                    self.send_target()
                self.suspected_bbx = []  # clear the cache of human
                toc = time.time()
                cv2.putText(canvas, "FPS:%f" % (1. / (toc - tic)), (10, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.result_writer.write(canvas)
                cv2.imshow('tracking', canvas)
                cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",
                        type=str,
                        default="workdirs/samples/cam1_uav1.mp4")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path",
                        type=str,
                        default="./workdirs/outputs/test")
    parser.add_argument("--save_origin", action="store_true")
    parser.add_argument("--camera",
                        action="store",
                        dest="cam",
                        type=int,
                        default="-1")

    parser.add_argument("--tcp_enable", action="store_true")
    parser.add_argument("--host_ip", type=str, default='192.168.43.217')
    parser.add_argument("--port", type=int, default=9999)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracker = Tracker(args)
    tracker.tracker_init()
    tracker.run()
