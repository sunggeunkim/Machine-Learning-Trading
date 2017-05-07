import pandas as pd

class Model:

    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def query(self, X_test):
        return self.model.predict(X_test)
