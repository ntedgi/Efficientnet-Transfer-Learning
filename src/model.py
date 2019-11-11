import os
import time
import math

import numpy as np


from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from skimage.io import imread

from src.efficientnet_lib import get_custom_objects
from src.efficientnet_lib.keras import center_crop_and_resize, preprocess_input


def file_exist(path):
    return os.path.exists(path)


def read_all_files_inside_dir(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file or '.jpg' in file or '.JPEG' in file or '.bmp' in file:
                files.append(os.path.join(r, file))
    return files


class Prediction:
    def __init__(self, label, probability):
        self.label = label
        self.probability = probability

    def __repr__(self):
        return f'({self.label},{self.probability}) '


class ImageRecognitionModelResponse:
    def __init__(self, transfer_learning_result, efficientnet_result, image_path):
        self.image_path = image_path
        self.transfer_learning_result = transfer_learning_result
        self.efficientnet_result = efficientnet_result


class TransferLearningResult:
    def __init__(self, prediction):
        labels_map = ["green", "guns", "knife", "mosque", "cars"]
        results = []
        for index, probability in enumerate(prediction):
            results.append(Prediction(labels_map[index], probability))
        self.predictions = results

    def __repr__(self):
        return f'TransferLearningResult({self.predictions[0], self.predictions[1], self.predictions[2], self.predictions[3], self.predictions[4]})'


class EfficientnetResult:
    def __init__(self, prediction):
        results = []
        for pred in prediction[0]:
            results.append(Prediction(pred[1], pred[2]))
        self.predictions = results

    def __repr__(self):
        return f'EfficientnetResult({self.predictions[0], self.predictions[1], self.predictions[2], self.predictions[3], self.predictions[4],})'


class ImageRecognitionModel:
    def __init__(self, model_path):
        if file_exist(model_path):
            self.model = load_model(model_path, custom_objects=get_custom_objects())
        else:
            print(f'model not found in :{model_path}')
            raise FileNotFoundError()

    def create_response_from_prediction(self, prediction):
        transfer_learning_result = self.extract_retraind_results_from_predections(prediction[0])
        efficientnet_result = self.extract_efficientnet_results_from_predections(prediction[1])
        return transfer_learning_result, efficientnet_result

    def single_image_prediction(self, image_path):
        image_bytes_repr = self.prepare_image_for_prediction(image_path)
        prediction = self.model.predict(image_bytes_repr)
        return self.create_response_from_prediction(prediction)

    def predict(self, image_directory):
        data_set = read_all_files_inside_dir(image_directory)
        if len(data_set) == 0:
            raise Exception("input dir is empty !")
        results = []
        for image_index, image_path in enumerate(data_set):
            try:
                print(f' {image_index} / {len(data_set)} process')
                transfer_learning_result, efficientnet_result = self.single_image_prediction(image_path)
                results.append(ImageRecognitionModelResponse(transfer_learning_result, efficientnet_result, image_path))
            except Exception as e:
                print(e)
                print(f'exeption thrown when trying to process : {image_path}')
        return results

    def predict_on_batch(self, image_directory):
        data_set = read_all_files_inside_dir(image_directory)
        data_set_image_bytes_repr = map(self.prepare_image_for_prediction, data_set)
        data_set_image_bytes_repr = list(map(lambda x: x[0], data_set_image_bytes_repr))
        prediction = self.model.predict_on_batch(np.array(data_set_image_bytes_repr, dtype=np.int))
        results = []
        for index in range(0, len(data_set)):
            transfer_learning_result, efficientnet_result = self.create_response_from_prediction(
                [[prediction[0][index]], np.array([prediction[1][index]])])
            results.append(
                ImageRecognitionModelResponse(transfer_learning_result, efficientnet_result, data_set[index]))
        return results

    def extract_efficientnet_results_from_predections(self, prediction):
        predictions = decode_predictions(prediction)
        return EfficientnetResult(predictions)

    def extract_retraind_results_from_predections(self, prediction):
        return TransferLearningResult(
            [prediction[0][0], prediction[0][1], prediction[0][2], prediction[0][3], prediction[0][4]])

    def prepare_image_for_prediction(self, image_path):
        image = imread(image_path)
        image_size = self.model.input_shape[1]
        image_bytes_repr = center_crop_and_resize(image, image_size=image_size)
        image_bytes_repr = preprocess_input(image_bytes_repr)
        image_bytes_repr = np.expand_dims(image_bytes_repr, 0)
        return image_bytes_repr

    def benchmarking(self, images_path, bulk_size):

        def take_time(start_time):
            return time.time() - start_time

        def chunkIt(seq, num):
            avg = len(seq) / float(num)
            out = []
            last = 0.0

            while last < len(seq):
                out.append(seq[int(last):int(last + avg)])
                last += avg
            return out

        start = time.time()
        print('start loading images')
        data_set = read_all_files_inside_dir(images_path)
        print(f'finish loading {len(data_set)} images path after {take_time(start)}')
        chunks = chunkIt(data_set, bulk_size)
        for chunk in chunks:
            start_chunk = time.time()
            data_set_image_bytes_repr = map(self.prepare_image_for_prediction, chunk)
            data_set_image_bytes_repr = list(map(lambda x: x[0], data_set_image_bytes_repr))
            prediction = self.model.predict_on_batch(np.array(data_set_image_bytes_repr, dtype=np.int))
            finish_t = take_time(start_chunk)
            print(f'finish predicting {len(chunk)} images in {"{0:.2f}".format(finish_t)}/sec . avg {"{0:.2f}".format(finish_t/len(chunk))}/sec - per image  . {math.floor(1/(finish_t/len(chunk)))} images per sec ')
