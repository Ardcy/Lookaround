from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.TRAIN_MODE = ''  
_C.SEED = 1
_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"
_C.MODEL.GPU_IDS = ""
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'resnet50'

_C.MODEL.RESNEXT = CN()
_C.MODEL.RESNEXT.CARDINALITY = 8
_C.MODEL.RESNEXT.DEPTH = 29
_C.MODEL.RESNEXT.BASE_WIDTH = 64
_C.MODEL.RESNEXT.WIDEN_FACTOR = 4
_C.MODEL.CLASSES = 100
_C.MODEL.PRETRAINED = False

_C.OUTPUT_DIR = 'base'
_C.DATA_DIR = 'dataset'

# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.Adam_LR = 0.001
_C.SOLVER.Adam_Beta1 = 0.9
_C.SOLVER.Adam_Beta2 = 0.999
_C.SOLVER.Adam_weight_decay = 0.00005


_C.SOLVER.MAX_EPOCHS = 200
_C.SOLVER.MIN_LR = 1e-5
_C.SOLVER.LR = 0.01
_C.SOLVER.BATCH_SIZE = 128

_C.SOLVER.SCHEDULER = 'MultiStepLR'
_C.SOLVER.SCHEDULER_GAMMA = 0.2
_C.SOLVER.SCHEDULER_MVALUE = [0]
_C.SOLVER.LOSS = 'CrossEntropy'
_C.SOLVER.HEAD_NUM = 3
_C.SOLVER.FREQUENCY = 1

_C.SOLVER.WARM = 3

_C.CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
_C.CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

_C.CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
_C.CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

_C.BUILD_TRANSFORM = 'RandomHorizontalFlip'
_C.BUILD_TRANSFORM_NUM = 3

_C.DATASET = CN()

_C.DATASET.IMAGESIZE = 32
_C.DATASET.NAME = 'cifar100'

_C.DATALOADER = ''
_C.SAVE_FREQ = 1
