

class BasePredictor():

    def __init__(self):
        raise NotImplementedError("Load your models in the constructor")

    def predict(self):
        raise NotImplementedError("Override the predict method")