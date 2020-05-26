from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from logger.logger_app import Applogger
from preprocessing import preprocessing_in
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


class ModelTrainer():
    def __init__(self):
        self.file_object = open('TestProject/Logs/logstraining.txt','a+')
        self.logger = Applogger()

    def train_model(self):
        self.logger.log(self.file_object,'Entering the model training method')
        try:
            self.logger.log(self.file_object,'Starting the split')

            # Getting the data from the source
            data_getter = preprocessing_in.datainput_ext(self.file_object,self.logger)
            data = data_getter.get_data()
            null_present = data_getter.missing_vals(data)
            data = data_getter.missing_imputer(data)
            X,y = data_getter.preprocess(data)
            print(X.head)
            X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1234)
            rfc = RandomForestClassifier()
            rfc.fit(X_train,y_train)
            pred = rfc.predict(X_test)
            print(classification_report(pred,y_test))
        except Exception as e:
            self.logger.log(self.file_object,'Exception in the model finder')
            raise Exception()
