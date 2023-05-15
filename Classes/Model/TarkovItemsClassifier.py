import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, box_iou

# train params
BATCH_SIZE = 3
NUM_EPOCHS = 50
# DEVICE = torch.device('cuda')
# optimizer params
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
# GAMMA = 0.1
# GAMMA_STEP = 10
BACKBONE = torchvision.models.mobilenet_v3_large(weights="DEFAULT").features
NUM_CLASSES = 2974
# anchor generator params
ANCHOR_SIZES = ((32, 64, 96, 128, 160, 192, 224, 256,),)
ANCHOR_ASPECT_RATIOS = ((0.5, 1., 2.,),) * len(ANCHOR_SIZES)
# RoI poller params
ROI_FEATMAP_NAMES = ['0']
ROI_OUTPUT_SIZE = 8
ROI_SAMPLING_RATIO = 2
# RPN params
RPN_FG_IOU_TRESH = 0.75
RPN_BG_IOU_TRESH = 0.25
RPN_POSITIVE_FRACTION = 0.3
RPN_NMS_THRESH = 0.75
RPN_BATCH_SIZE_PER_IMAGE = 512
RPN_PRE_NMS_TOP_N_TRAIN = 2000
RPN_PRE_NMS_TOP_N_TEST = 1000
RPN_POST_NMS_TOP_N_TRAIN = 2000
RPN_POST_NMS_TOP_N_TEST = 1000
# BOX params
BOX_BATCH_SIZE_PER_IMAGE = 512
BOX_NMS_THRESH=0.5


MIN_SIZE = 1024
MAX_SIZE = 1024


class TarkovItemsClassifier(FasterRCNN):
    def __init__(self, weight='DEFAULT'):
        anchor_generator = AnchorGenerator(sizes=ANCHOR_SIZES,
                                           aspect_ratios=ANCHOR_ASPECT_RATIOS)
        roi_pooler = MultiScaleRoIAlign(featmap_names=ROI_FEATMAP_NAMES,
                                        output_size=ROI_OUTPUT_SIZE,
                                        sampling_ratio=ROI_SAMPLING_RATIO)
        out = BACKBONE(torch.rand(size=(1, 3, 224, 224)))
        # out.shape[1] return count output channels for backbone
        BACKBONE.out_channels = out.shape[1]
        super().__init__(num_classes=NUM_CLASSES,
                         backbone=BACKBONE,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         rpn_positive_fraction=RPN_POSITIVE_FRACTION,
                         min_size=MIN_SIZE,
                         max_size=MAX_SIZE,
                         rpn_nms_thresh=RPN_NMS_THRESH,
                         box_batch_size_per_image=BOX_BATCH_SIZE_PER_IMAGE,
                         rpn_batch_size_per_image=RPN_BATCH_SIZE_PER_IMAGE,
                         rpn_pre_nms_top_n_train=RPN_PRE_NMS_TOP_N_TRAIN,
                         rpn_pre_nms_top_n_test=RPN_PRE_NMS_TOP_N_TEST,
                         rpn_post_nms_top_n_train=RPN_POST_NMS_TOP_N_TRAIN,
                         rpn_post_nms_top_n_test=RPN_POST_NMS_TOP_N_TEST,
                         box_nms_thresh=BOX_NMS_THRESH)
        if weight == 'DEFAULT':
            # TODO add state_dict load from url
            state_dict = torch.load('/opt/conda/tmp/last_epoch_state.pt')
        else:
            state_dict = weight
        super().load_state_dict(state_dict)
