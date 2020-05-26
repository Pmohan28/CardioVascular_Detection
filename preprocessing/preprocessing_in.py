import pandas as pd
from logger.logger_app import Applogger
import numpy as np
from sklearn.impute import KNNImputer


class datainput():
    def __init__(self,file_object,logger):
        self.file_object = open('TestProject/Logs/logs.txt','a+')
        self.training_file = 'TestProject/InputData/cardio_train.csv'
        self.logger = Applogger()

    def get_data(self):
        self.logger.log(self.file_object,'Entering the data input class')
        try:
            self.data = pd.read_csv(self.training_file, delimiter=';')
            self.logger.log(self.file_object,'Data Loaded successfully')
            print(self.data)
            return self.data
        except Exception as e:
            self.logger.log(self.file_object,'Exception occured in reading the file')
            self.logger.log(self.file_object,'data load unsuccesful')

            raise Exception()

    def missing_vals(self,data):

        """
        This method is to verify missing values in data
        """
        self.data = data
        self.logger.log(self.file_object, 'Starting Missing values computation')
        self.null_present = False

        try:
            self.null_count = data.isna().sum()
            for x in self.null_count:
                if x > 0:
                    self.null_present = True
                    break
            if (self.null_present):
                df_null = pd.DataFrame()
                df_null['columns'] = data.columns
                # print(df_null)
                df_null['missingvalues'] = np.asarray(data.isna().sum())
                df_null.to_csv('TestProject/preprocessing/null_values.csv')
            self.logger.log(self.file_object,'exported the null values')
            return  self.null_present
        except Exception as e:
            self.logger.log(self.file_object, 'Exception occured in missing  values method')
            self.logger.log(self.file_object,'Exited the code since there is an exception occured in the null values method.')
            raise Exception()

    def missing_imputer(self,data):
        self.data = data
        self.logger.log(self.file_object,'Starting to treat the missing values')
        try:
            self.imputer = KNNImputer()
            self.new_array = self.imputer.fit_transform(data)
            self.new_data = pd.DataFrame(data = self.new_array, columns= self.data.columns)
            # data.fillna(0,inplace=Tru)
            print(data.isna().sum())
            self.logger.log(self.file_object,'Ending the imputation successfully')
            return self.new_data

        except Exception as e:
            self.logger.log(self.file_object,'Exception during imputation')
            self.logger.log(self.file_object,'Exited the code because of fault in missing imputer fx.')
            raise Exception()

    def preprocess(self,data):
        """"
        This method is to impute dummify the categorical variables.
        """
        self.data = data
        self.logger.log(self.file_object,'Starting preprocessing')
        try:
            self.X = data.drop(columns = ['cardio', 'id', 'age'], axis=1)
            self.y = data['cardio']
            self.X = pd.get_dummies(self.X, columns=['gender'])
            print(self.X.shape, self.y.shape)
            return self.X, self.y
        except Exception as e:
            self.logger.log(self.file_object, 'Exception occured')
        raise Exception()


class datainput_ext(datainput):

    def missing_imputer(self,data):
        self.data = data
        self.logger.log(self.file_object,'Starting to treat the missing values in Inherited classs data_ext')
        try:
            data.fillna(0)
            print(data.isna().sum())
            self.logger.log(self.file_object,'Ending the imputation of extended class successfully')
            return data

        except Exception as e:
            self.logger.log(self.file_object,'Exception during imputation of extended ')
            self.logger.log(self.file_object,'Exited the code because of fault in missing imputer fx.')
            raise Exception()


# if __name__ == '__main__':
#     s = datainput('x','y')
#     x = s.get_data()
#     s.preprocess(x)


