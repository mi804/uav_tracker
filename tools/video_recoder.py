import argparse
import os

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./videos/samples")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--camera",
                        action="store",
                        dest="cam",
                        type=int,
                        default="0")
    parser.add_argument("--frame_interval", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Using webcam " + str(args.cam))
    vdo = cv2.VideoCapture(args.cam)
    ret, frame = vdo.read()
    assert ret, "Error: Camera error"
    im_width = frame.shape[0]
    im_height = frame.shape[1]
    os.makedirs(args.save_path, exist_ok=True)
    save_video_path = os.path.join(args.save_path, f"video_cam{args.cam}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(save_video_path, fourcc, 20,
                             (im_height, im_width))
    idx_frame = 0
    while vdo.grab():
        idx_frame += 1
        print(f'frame:{idx_frame}')
        if idx_frame % args.frame_interval:
            continue
        _, ori_im = vdo.retrieve()
        writer.write(ori_im)
        if args.show:
            cv2.imshow(f'cam{args.cam}', ori_im)
            cv2.waitKey(1)
    writer.release()
