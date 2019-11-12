import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
parser = argparse.ArgumentParser(description='predict on images.')
parser.add_argument('-p', metavar='--path', type=str, help='path to model', required=True)
parser.add_argument('-i', metavar='--image-dir', type=str, help='image directory', required=True)

args = parser.parse_args()


def eval(model_input, input_path):
    from src.model import ImageRecognitionModel
    model = ImageRecognitionModel(model_input)
    results = model.predict_on_batch(input_path)
    for r in results:
        print(f'result for :{r.image_path}')
        print(f'{r.efficientnet_result}')
        print(f'{r.transfer_learning_result}')
        print()


eval(args.p, args.i)


