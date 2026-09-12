"""Train-only numeric scaling and explicit missingness for tabular neural models."""
import numpy as np
from sklearn.impute import SimpleImputer
from copper_mvp.classical_models import StableNumericScale


class NeuralInputs:
    numeric_count=220
    def fit(self,X):
        self.imputer=SimpleImputer(strategy="median",keep_empty_features=True)
        numerical=self.imputer.fit_transform(X[:,:110])
        self.scaler=StableNumericScale().fit(numerical)
        return self

    def transform(self,X):
        X=np.asarray(X,float)
        missing=np.isnan(X[:,:110]).astype(np.float32)
        numerical=self.scaler.transform(self.imputer.transform(X[:,:110]))
        continuous=np.ascontiguousarray(np.column_stack((numerical,missing)),dtype=np.float32)
        categories=np.ascontiguousarray(X[:,110:],dtype=np.int64)
        return continuous,categories

    def fit_transform(self,X):
        return self.fit(X).transform(X)
