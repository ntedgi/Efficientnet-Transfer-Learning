import os
import unittest

from src.model import ImageRecognitionModel



class CreationTests(unittest.TestCase):

    def test_intilized_model_with_wrong_path_throws_exeption(self):
        with self.assertRaises(FileNotFoundError):
            ImageRecognitionModel("not_existing_path")

    def test_intilized_model_with_correct_model_path(self):
        try:
            ImageRecognitionModel(os.path.realpath("./../bin/model.h5"))
        except FileNotFoundError:
            self.fail("test_intilized_model_with_correct_model_path() raised ExceptionType unexpectedly!")


class PredictionsTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        self.model = ImageRecognitionModel(os.path.realpath("../bin/model.h5"))

    def test_make_simple_prediction(self):
        try:
            self.model.predict(os.path.realpath('../samples/'))
        except FileNotFoundError:
            self.fail("test_make_simple_prediction raised ExceptionType unexpectedly!")


if __name__ == '__main__':
    alltests = unittest.TestSuite()
    alltests.addTest(unittest.makeSuite(CreationTests))
    alltests.addTest(unittest.makeSuite(PredictionsTests))
    result = unittest.TextTestRunner(verbosity=2).run(alltests)
