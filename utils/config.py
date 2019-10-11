from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
cfg = __C

# seed
__C.RANDOM_SEED = 42

# train parameters
__C.TRAIN = edict()

__C.TRAIN.EPOCHS = 100
__C.TRAIN.LEARNING_RATE = 1.0e-3
__C.TRAIN.BATCH_SIZE = 10


# scheduler
__C.TRAIN.SCHEDULER = edict()
__C.TRAIN.SCHEDULER.STEP_SIZE = 10
__C.TRAIN.SCHEDULER.GAMMA = 0.1

# sgd
__C.TRAIN.SGD = edict()
__C.TRAIN.SGD.MOMENTUM = 0.9

# L2 regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005
