import os
import warnings

from src.model import ImageRecognitionModel

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
deprecation._PRINTED_WARNING = False


model = ImageRecognitionModel("/home/naor/projects/efficientnetretrainedmodel/bin/model-3-pred.h5")
print("start benchmarking")
model.benchmarkng("/home/naor/projects/Image-Recognition/test-4/train/0/", 344)
