from preprocessing.preprocessing_in import datainput
from logger.logger_app import Applogger
from modelfinder.modelfinder import ModelTrainer

if __name__ == '__main__':
        # input_data = datainput('x', 'y')
        # preprocessing_data = input_data.get_data()
        # feat_trans = input_data.preprocess(preprocessing_data)
        model_training = ModelTrainer()
        model_training.train_model()
