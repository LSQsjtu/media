import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(BlazeBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (1, 2, 1, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)
    def forward(self, x):
        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.act(self.convs(h) + x)

class lastBlock(nn.modules):
    def __init__(self, in_channels, out_channels):
        super (lastBlock, self).__init__()

        self.convs =  nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):

        return self.convs(x)

class BlazePalm(nn.Module):
    """The BlazeFace face detection model from MediaPipe.

    The version from MediaPipe is simpler than the one in the paper;
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/google/mediapipe/
    """
    def __init__(self):
        super(BlazePalm, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 2944
        self.num_coords = 18
        self.score_clipping_thresh = 100.0
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.h_scale = 256.0
        self.w_scale = 256.0
        self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3

        self._define_layers()

    def _define_layers(self):
       self.stage1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.PReLU(32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 32),
            BlazeBlock(32, 64, stride=2),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
            BlazeBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            BlazeBlock(64, 128, stride=2),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            BlazeBlock(128, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )
        self.stage4 = nn.Sequential(
            BlazeBlock(256, 256, stride=2),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
            # resize_bilinear
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=1, padding=0, bias=True
            ),
            nn.PReLU(256),
        )
        self.stage5 = nn.Sequential(
            BlazeBlock(256, 256),
            BlazeBlock(256, 256),
        )
        self.stage6 = nn.Sequential(
            BlazeBlock(128, 128),
            BlazeBlock(128, 128),
        )

        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.PReLU(128),
        )
        
        self.convs1 =  nn.Conv2d(in_channels=128, out_channels=2,
                      kernel_size=1, stride=1, padding=0, bias=True)

        self.convs2 =  nn.Conv2d(in_channels=256, out_channels=6,
                      kernel_size=1, stride=1, padding=0, bias=True)
        
        self.convs3 =  nn.Conv2d(in_channels=128, out_channels=36,
                      kernel_size=1, stride=1, padding=0, bias=True)
        
        self.convs4 =  nn.Conv2d(in_channels=256, out_channels=108,
                      kernel_size=1, stride=1, padding=0, bias=True)
        # self.deconv1_blaze = BlazeBlock(256, 256)

        # self.deconv2 = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, 2, stride=2),
        #     nn.ReLU(inplace=True),
        #     )
        # self.deconv2_blaze = BlazeBlock(128, 128)

        # self.classifier_8 = nn.Conv2d(128, 2, 1, bias=True)
        # self.classifier_16 = nn.Conv2d(256, 2, 1, bias=True)
        # self.classifier_32 = nn.Conv2d(256, 6, 1, bias=True)

        # self.regressor_8 = nn.Conv2d(128, 36, 1, bias=True)
        # self.regressor_16 = nn.Conv2d(256, 36, 1, bias=True)
        # self.regressor_32 = nn.Conv2d(256, 108, 1, bias=True)

    def forward(self, x):
        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)

        b = x.shape[0]      # batch size, needed for reshaping later

        x = self.stage1(x)           # (b, 32, 128, 128)
        x = self.stage2(x)           # (b, 64, 64, 64)
        h_1=x

        x_s3 = self.stage3(x)           # (b, 128, 32, 32)
        x = self.stage4(x_s3) +x_s3
        x_s5 = self.stage5(x)

        x_d1=h_1+self.deconv1(x_s5)
        x_s6=self.stage6(x_d1)
        x_s7=self.convs1(x_s6)
        x_s7.reshape(1,-1,1)
        x_s8=self.convs2(x_s5)
        x_s8.reshape(1,-1,1)
        x_s9=self.convs3(x_s6)
        x_s9.reshape(1,-1,18)
        x_s10=self.convs4(x_s5)
        x_s10.reshape(1,-1,18)

        classificators=torch.cat((x_s7,x_s8),dim=1)
        regressors=torch.cat((x_s9,x_s10),dim=1)
        # x_s4 = self.stage4(x_s3)     # (b, 256, 16, 16)
        # x_s5 = self.stage5(x_s4)     # (b, 256, 8, 8)

        # x_d1 = self.deconv1(x_s5)
        # x_d1_blaze = self.deconv1_blaze(x_d1 + x_s4)  # (b, 256, 16, 16)

        # x_d2 = self.deconv2(x_d1_blaze)
        # x_d2_blaze = self.deconv2_blaze(x_d2 + x_s3)  # (b, 128, 32, 32)

        # c1 = self.classifier_8(x_d2_blaze)
        # c2 = self.classifier_16(x_d1_blaze)
        # c3 = self.classifier_32(x_s5)

        # c1 = c1.permute(0, 2, 3, 1)
        # c1 = c1.reshape(b, -1, 1)
        # c2 = c2.permute(0, 2, 3, 1)
        # c2 = c2.reshape(b, -1, 1)
        # c3 = c3.permute(0, 2, 3, 1)
        # c3 = c3.reshape(b, -1, 1)
        # c = torch.cat((c1, c2, c3), dim=1)  # (b, 2944, 1)

        # r1 = self.regressor_8(x_d2_blaze)
        # r2 = self.regressor_16(x_d1_blaze)
        # r3 = self.regressor_32(x_s5)

        # r1 = r1.permute(0, 2, 3, 1)
        # r1 = r1.reshape(b, -1, 18)
        # r2 = r2.permute(0, 2, 3, 1)
        # r2 = r2.reshape(b, -1, 18)
        # r3 = r3.permute(0, 2, 3, 1)
        # r3 = r3.reshape(b, -1, 18)
        # r = torch.cat((r1, r2, r3), dim=1)  # (b, 2944, 18)

        return [classificators,regressors]

    def _device(self):
        """Which device (CPU or GPU) is being used by this model?"""
        return self.classifier_8.weight.device

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_anchors(self, path):
        self.anchors = torch.tensor(np.load(path), dtype=torch.float32, device=self._device())
        assert(self.anchors.ndimension() == 2)
        assert(self.anchors.shape[0] == self.num_anchors)
        assert(self.anchors.shape[1] == 4)

    def _preprocess(self, x):
        """Converts the image pixels to the range [-1, 1]."""
        return x.float() / 127.5 - 1.0

    def predict_on_image(self, img):
        """Makes a prediction on a single image.

        Arguments:
            img: a NumPy array of shape (H, W, 3) or a PyTorch tensor of
                 shape (3, H, W). The image's height and width should be
                 128 pixels.

        Returns:
            A tensor with face detections.
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1)).to(self._device())

        return self.predict_on_batch(img.unsqueeze(0))[0]

    def predict_on_batch(self, x):
        """Makes a prediction on a batch of images.

        Arguments:
            x: a NumPy array of shape (b, H, W, 3) or a PyTorch tensor of
               shape (b, 3, H, W). The height and width should be 128 pixels.

        Returns:
            A list containing a tensor of face detections for each image in
            the batch. If no faces are found for an image, returns a tensor
            of shape (0, 17).

        Each face detection is a PyTorch tensor consisting of 17 numbers:
            - ymin, xmin, ymax, xmax
            - x,y-coordinates for the 6 keypoints
            - confidence score
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        assert x.shape[1] == 3
        assert x.shape[2] == 256
        assert x.shape[3] == 256

        # 1. Preprocess the images into tensors:
        x = x.to(self._device())
        x = self._preprocess(x)

        # 2. Run the neural network:
        with torch.no_grad():
            out = self.__call__(x)

        # 3. Postprocess the raw predictions:
        detections = self._tensors_to_detections(out[0], out[1], self.anchors)

        # 4. Non-maximum suppression to remove overlapping detections:
        filtered_detections = []
        for i in range(len(detections)):
            faces = self._weighted_non_max_suppression(detections[i])
            faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, 17))
            filtered_detections.append(faces)

        return filtered_detections

    def _tensors_to_detections(self, raw_box_tensor, raw_score_tensor, anchors):
        """The output of the neural network is a tensor of shape (b, 896, 16)
        containing the bounding box regressor predictions, as well as a tensor
        of shape (b, 896, 1) with the classification confidences.

        This function converts these two "raw" tensors into proper detections.
        Returns a list of (num_detections, 17) tensors, one for each image in
        the batch.

        This is based on the source code from:
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.proto
        """
        assert raw_box_tensor.ndimension() == 3
        assert raw_box_tensor.shape[1] == self.num_anchors
        assert raw_box_tensor.shape[2] == self.num_coords

        assert raw_score_tensor.ndimension() == 3
        assert raw_score_tensor.shape[1] == self.num_anchors
        assert raw_score_tensor.shape[2] == self.num_classes

        assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

        detection_boxes = self._decode_boxes(raw_box_tensor, anchors)

        thresh = self.score_clipping_thresh
        raw_score_tensor = raw_score_tensor.clamp(-thresh, thresh)
        detection_scores = raw_score_tensor.sigmoid().squeeze(dim=-1)

        # Note: we stripped off the last dimension from the scores tensor
        # because there is only has one class. Now we can simply use a mask
        # to filter out the boxes with too low confidence.
        mask = detection_scores >= self.min_score_thresh

        # Because each image from the batch can have a different number of
        # detections, process them one at a time using a loop.
        output_detections = []
        for i in range(raw_box_tensor.shape[0]):
            boxes = detection_boxes[i, mask[i]]
            scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
            output_detections.append(torch.cat((boxes, scores), dim=-1))

        return output_detections

    def _decode_boxes(self, raw_boxes, anchors):
        """Converts the predictions into actual coordinates using
        the anchor boxes. Processes the entire batch at once.
        """
        boxes = torch.zeros_like(raw_boxes)

        x_center = raw_boxes[..., 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[..., 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]

        w = raw_boxes[..., 2] / self.w_scale * anchors[:, 2]
        h = raw_boxes[..., 3] / self.h_scale * anchors[:, 3]

        boxes[..., 0] = y_center - h / 2.  # ymin
        boxes[..., 1] = x_center - w / 2.  # xmin
        boxes[..., 2] = y_center + h / 2.  # ymax
        boxes[..., 3] = x_center + w / 2.  # xmax

        for k in range(6):
            offset = 4 + k*2
            keypoint_x = raw_boxes[..., offset    ] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            keypoint_y = raw_boxes[..., offset + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            boxes[..., offset    ] = keypoint_x
            boxes[..., offset + 1] = keypoint_y

        return boxes

    def _weighted_non_max_suppression(self, detections):
        """The alternative NMS method as mentioned in the BlazeFace paper:

        "We replace the suppression algorithm with a blending strategy that
        estimates the regression parameters of a bounding box as a weighted
        mean between the overlapping predictions."

        The original MediaPipe code assigns the score of the most confident
        detection to the weighted detection, but we take the average score
        of the overlapping detections.

        The input detections should be a Tensor of shape (count, 17).

        Returns a list of PyTorch tensors, one for each detected face.

        This is based on the source code from:
        mediapipe/calculators/util/non_max_suppression_calculator.cc
        mediapipe/calculators/util/non_max_suppression_calculator.proto
        """
        if len(detections) == 0: return []

        output_detections = []

        # Sort the detections from highest to lowest score.
        remaining = torch.argsort(detections[:, 16], descending=True)

        while len(remaining) > 0:
            detection = detections[remaining[0]]

            # Compute the overlap between the first box and the other
            # remaining boxes. (Note that the other_boxes also include
            # the first_box.)
            first_box = detection[:4]
            other_boxes = detections[remaining, :4]
            ious = overlap_similarity(first_box, other_boxes)

            # If two detections don't overlap enough, they are considered
            # to be from different faces.
            mask = ious > self.min_suppression_threshold
            overlapping = remaining[mask]
            remaining = remaining[~mask]

            # Take an average of the coordinates from the overlapping
            # detections, weighted by their confidence scores.
            weighted_detection = detection.clone()
            if len(overlapping) > 1:
                coordinates = detections[overlapping, :16]
                scores = detections[overlapping, 16:17]
                total_score = scores.sum()
                weighted = (coordinates * scores).sum(dim=0) / total_score
                weighted_detection[:16] = weighted
                weighted_detection[16] = total_score / len(overlapping)

            output_detections.append(weighted_detection)

        return output_detections


# IOU code from https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ??? B / A ??? B = A ??? B / (area(A) + area(B) - A ??? B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box.unsqueeze(0), other_boxes).squeeze(0)

if __name__ == '__main__':
    model = BlazePalm()
    test_input = torch.randn(1, 3, 256, 256)
    r, c = model(test_input)
    print('regression and classification shape: ', r.shape, c.shape)
