from src.model import ImageRecognitionModel


def main():
    model_input = "/home/naor/projects/efficientnetretrainedmodel/bin/model-3-pred.h5"
    model = ImageRecognitionModel(model_input)
    results = model.predict_on_batch("/home/naor/projects/efficientnetretrainedmodel/samples/knifes")
    for r in results:
        print(f'result for :{r.image_path}')
        print(f'{r.efficientnet_result}')
        print(f'{r.transfer_learning_result}')
        print()


if __name__ == "__main__":
    main()
