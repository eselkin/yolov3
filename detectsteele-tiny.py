import argparse
import time
import csv
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *


def detect(
        cfg,
        data_cfg,
        weights,
        images='data/samples',  # input folder
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False,
        cage_w=29.21,
        cage_h=12.7,
        split_x=2,
        split_y=2,
        empty_r = 1,
        empty_c = 1,
        repeat = 1
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(
            weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    model.fuse()

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    if not webcam:
        dataloader = LoadImages(images, img_size=img_size)
    else:
        dataloader = LoadWebcam(img_size=img_size)
    
    for i in range(repeat):
        # Get classes and colors
        classes = load_classes(parse_data_cfg(data_cfg)['names'])
        colors = [[random.randint(0, 255) for _ in range(3)]
                for _ in range(len(classes))]
        out_video_path = os.path.join(output, 'out2.avi')
        out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 30, (720, 480))
        cum_time = 0
        with open('distance.csv', 'w', newline='') as csvdistance, open('position.csv', 'w', newline='') as csvposition:
            for i, (path, img, im0, vid_cap) in enumerate(dataloader):
                distance_csv = csv.writer(csvdistance)
                position_csv = csv.writer(csvposition, delimiter=' ')
                last_positions_x = None
                last_positions_y = None
                mouse_bodies_positions_x = np.zeros(
                    (split_x, split_y), dtype=np.float)
                mouse_bodies_positions_y = np.zeros(
                    (split_x, split_y), dtype=np.float)
                t = time.time()
                save_path = str(Path(output) / Path(path).name)

                # Get detections
                img = torch.from_numpy(img).unsqueeze(0).to(device)
                pred, _ = model(img)
                det = non_max_suppression(pred, conf_thres, nms_thres)[0]

                if det is not None and len(det) > 0:
                    # Rescale boxes from 416 to true image size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results to screen
                    # print('%gx%g ' % img.shape[2:], end='')  # print image size
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        # print('%g %ss' % (n, classes[int(c)]), end=', ')

                    # Draw bounding boxes and labels of detections
                    for *xyxy, conf, cls_conf, cls in det:
                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        # Add bbox to the image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label,
                                    color=colors[int(cls)])
                        x1, y1, x2, y2 = xyxy
                        if classes[int(cls)] == "mouse_body":
                            x1n = min(x1.cpu().numpy(), img_size-1)
                            y1n = min(y1.cpu().numpy(), img_size-1)
                            x2n = min(x2.cpu().numpy(), img_size-1)
                            y2n = min(y2.cpu().numpy(), img_size-1)
                            c = int((x1n*split_x)/img_size)
                            r = int((y1n*split_y)/img_size)
                            if mouse_bodies_positions_x[r, c] == 0 and not (r == empty_r and c == empty_c):
                                xmn = (x1n + x2n) / 2
                                ymn = (y1n + y2n) / 2
                                mouse_bodies_positions_x[r, c] = xmn
                                mouse_bodies_positions_y[r, c] = ymn
                    if last_positions_x is None:
                        # first run
                        last_positions_x = mouse_bodies_positions_x
                        last_positions_y = mouse_bodies_positions_y
                    where_x = np.where(mouse_bodies_positions_x == 0)
                    locations = np.dstack(where_x)[0]
                    for location in locations:
                        r = location[0]
                        c = location[1]
                        mouse_bodies_positions_x[r, c] = last_positions_x[r, c]
                        mouse_bodies_positions_y[r, c] = last_positions_y[r, c]
                    diff_x = mouse_bodies_positions_x-last_positions_x
                    diff_y = mouse_bodies_positions_y-last_positions_y
                    difference_x_cm = np.abs(
                        diff_x) * (cage_w/img_size)  # px * cm/px = cm
                    difference_y_cm = np.abs(
                        diff_y) * (cage_h/img_size)  # px * cm/px = cm
                    difference_hyp_cm = np.sqrt(
                        difference_x_cm+difference_y_cm)  # in cm
                    flat_hyps = difference_hyp_cm.flatten().tolist()
                    xs = (mouse_bodies_positions_x.flatten()
                        * (cage_w/img_size)).tolist()
                    ys = (mouse_bodies_positions_y.flatten()
                        * (cage_h/img_size)).tolist()
                    xys = zip(xs, ys)
                    distance_csv.writerow(flat_hyps)
                    position_csv.writerow(xys)
                    last_positions_x = mouse_bodies_positions_x
                    last_positions_y = mouse_bodies_positions_y
                dt = time.time() - t
                cum_time += dt
                out.write(im0)
                if i == 1000:
                    out.release()
                    break
        print(f'Cum time {cum_time}s for 1000 frames so {cum_time/1000} per frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str,
                        default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str,
                        default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--images', type=str,
                        default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5,
                        help='iou threshold for non-maximum suppression')
    parser.add_argument('--repeat', type=int, default=1, help='for testing purposes, repeat detections. Can only be used with video')

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            images=opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            repeat=opt.repeat
        )
