import argparse
import os.path

import slpt
from slpt.config import cfg
from slpt.config import update_config

from slpt.utils import create_logger, crop_v2
from slpt.SLPT import Sparse_alignment_network
from slpt.dataloader import WFLW_test_Dataset

import torch, cv2, math
import numpy as np
import pprint

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from slpt import facedetector
from slpt import utils
from tqdm import trange
from pathlib import Path
import libfacedetection


def parse_args():
    parser = argparse.ArgumentParser(description='Process a single image')

    # face detector
    parser.add_argument('-m', '--trained_model',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--input', type=str, help='the image file to be detected')
    parser.add_argument('--output', required=True, type=str, help='path to save the img to')
    parser.add_argument('--output-landmarks', type=str, help='path to save the landmarks to')
    parser.add_argument('--last-frame',  type=int, help='only process this many frames')
    parser.add_argument('--confidence_threshold', default=0.7, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
    parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
    parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')

    # landmark detector
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str)
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)

    args = parser.parse_args()

    return args


def draw_landmark(landmark, image):

    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 6, (0, 255, 0), -1)

    return image


def crop_img(img, bbox, transform):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans

def face_detection(img, model, im_width, im_height, device, confidence_threshold, top_k, keep_top_k, nms_threshold):
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
    img = np.float32(img)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    scale = torch.Tensor([im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height, im_width, im_height,
                          im_width, im_height])
    scale = scale.to(device)

    # feed forward
    loc, conf, iou = model(img)

    # post processing
    priorbox = facedetector.PriorBox(facedetector.cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = facedetector.decode(loc.data.squeeze(0), prior_data, facedetector.cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    cls_scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    iou_scores = iou.squeeze(0).data.cpu().numpy()[:, 0]
    # clamp here for the compatibility for ONNX
    _idx = np.where(iou_scores < 0.)
    iou_scores[_idx] = 0.
    _idx = np.where(iou_scores > 1.)
    iou_scores[_idx] = 1.
    scores = np.sqrt(cls_scores * iou_scores)

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    selected_idx = np.array([0, 1, 2, 3, 14])
    keep = facedetector.nms(dets[:, selected_idx], nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]

    return dets

def find_max_box(dets, vis_thres):
    potential_box = []
    for b in dets:
        if b[14] < vis_thres:
            continue
        potential_box.append(np.array([b[0], b[1], b[2], b[3], b[14]], dtype=np.int))

    if len(potential_box) > 0:
        x1, y1, x2, y2 = (potential_box[0][:4]).astype(np.int32)
        Max_box = (x2 - x1) * (y2 - y1)
        Max_index = 0
        for index in range(1, len(potential_box)):
            x1, y1, x2, y2 = (potential_box[index][:4]).astype(np.int32)
            temp_box = (x2 - x1) * (y2 - y1)
            if temp_box >= Max_box:
                Max_box = temp_box
                Max_index = index
        return dets[Max_index]
    else:
        return None


def get_or_download_model():
    import gdown
    try:
        from torch.hub import get_dir
    except BaseException:
        from torch.hub import _get_torch_home as get_dir

    hub_dir = Path(get_dir())
    model_path = hub_dir / 'checkpoints/slpt_12.pth'
    if not model_path.exists():
        model_path.parent.mkdir(exist_ok=True)
        logger.warning('Downloading SLPT model (GPL)')
        gdown.download(id='1fBBHqVSW4XQ_eB3ClYv4mS2pYmRWPuFJ', output=str(model_path))
    return model_path

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    device = torch.device(args.device)

    torch.set_grad_enabled(False)

    # Cuda
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # load face detector
    initial_landmarks_path = Path(slpt.__file__).parent / 'config/init_98.npz'
    net = facedetector.YuFaceDetectNet(phase='test', size=None)  # initialize detector
    face_detector_model = Path(libfacedetection.__file__).parent / 'tasks/task1/weights/yunet_final.pth'
    net = facedetector.load_model(net, face_detector_model, True)


    net.eval()
    net = net.to(device)
    print('Finished loading Face Detector!')

    model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                     cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                     cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                     cfg.TRANSFORMER.FEED_DIM, initial_landmarks_path, cfg)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()


    model_path = get_or_download_model()
    checkpoint = torch.load(model_path)
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.module.state_dict().keys()}
    model.module.load_state_dict(pretrained_dict)
    model.eval()

    print('Finished loading face landmark detector')

    # Camera Begin
    frame = cv2.imread(args.input)
    im_height, im_width, *_ = frame.shape

    # Video writer
    # out = cv2.VideoWriter('out4.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 20, (im_width, im_height))

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    all_landmarks = []

    dets = face_detection(frame.copy(), net, 320, 240, device, args.confidence_threshold, args.top_k, args.keep_top_k, args.nms_threshold)
    bbox = find_max_box(dets, args.vis_thres)

    if bbox is not None:
        bbox[0] = int(bbox[0] / 320.0 * im_width + 0.5)
        bbox[2] = int(bbox[2] / 320.0 * im_width + 0.5)
        bbox[1] = int(bbox[1] / 240.0 * im_height + 0.5)
        bbox[3] = int(bbox[3] / 240.0 * im_height + 0.5)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)

        outputs_initial = model(alignment_input.cuda())
        output = outputs_initial[2][0, -1, :, :].cpu().numpy()

        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)
        all_landmarks.append(landmark)
        # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        frame = draw_landmark(landmark, frame)
        # out.write(frame)
        # cv2.imshow('res', frame)
        cv2.imwrite(args.output, frame)

    all_landmarks = np.stack(all_landmarks)
    if args.output_landmarks is not None:
        np.save(args.output_landmarks, all_landmarks)

