from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# --------------------------------------------------
# Base class
# --------------------------------------------------
class Base(BaseEstimator, TransformerMixin):

    def get_feature_names_out(self, input_features=None):
        
        if hasattr(self, "columns_"):
            return np.array(self.columns_)
            
        raise AttributeError(
            f"{self.__class__.__name__} must define self.columns_"
        )


# --------------------------------------------------
# PCLASS
# --------------------------------------------------
class PClassEncoder(Base):

    def __init__(self):
        self.columns_ = ['pclass_1', 'pclass_3']

    def fit(self, X, y=None):
        if 'pclass' not in X.columns:
            raise ValueError("pclass column not found")
        return self

    def transform(self, X):
        X = X.copy()
        X["pclass_1"] = (X["pclass"] == 1).astype(int)
        X["pclass_3"] = (X["pclass"] == 3).astype(int)
        return X[self.columns_]


# --------------------------------------------------
# NAME
# --------------------------------------------------
class NameExtractor(Base):

    def __init__(self):
        self.columns_ = ['title', 'last_name']

    def fit(self, X, y=None):
        if 'name' not in X.columns:
            raise ValueError("name column not found")
        return self

    def transform(self, X):
        X = X.copy()
        X['title'] = X['name'].str.extract(r',\s*([^.]*)\.')
        X['last_name'] = X['name'].str.extract(r'^([^,]+),')
        return X[self.columns_]


# --------------------------------------------------
# SEX
# --------------------------------------------------
class SexEncoder(Base):

    def __init__(self):
        self.columns_ = ['is_male']

    def fit(self, X, y=None):
        if 'sex' not in X.columns:
            raise ValueError("sex column not found")
        return self

    def transform(self, X):
        X = X.copy()
        X['is_male'] = (X['sex'] == 'male').astype(int)
        return X[self.columns_]


# --------------------------------------------------
# SIBSP
# --------------------------------------------------
class SibspBinning(Base):

    def __init__(self):
        self.columns_ = ['bin_sibsp']

    def fit(self, X, y=None):
        if 'sibsp' not in X.columns:
            raise ValueError("sibsp column not found")
        return self

    def transform(self, X):
        X = X.copy()

        def helper(x):
            if x == 0:
                return 'alone'
            elif x <= 2:
                return 'small'
            else:
                return 'large'

        X['bin_sibsp'] = X['sibsp'].apply(helper)
        return X[self.columns_]


# --------------------------------------------------
# PARCH
# --------------------------------------------------
class ParchBinning(Base):

    def __init__(self):
        self.columns_ = ['bin_parch']

    def fit(self, X, y=None):
        if 'parch' not in X.columns:
            raise ValueError("parch column not found")
        return self

    def transform(self, X):
        X = X.copy()

        def helper(x):
            if x == 0:
                return 'alone'
            elif 1 <= x <= 2:
                return 'small'
            else:
                return 'large'

        X['bin_parch'] = X['parch'].apply(helper)
        return X[self.columns_]


# --------------------------------------------------
# TICKET
# --------------------------------------------------
class TicketExtractorAdvanced(Base):

    def __init__(self, top_k=10):
        self.top_k = top_k
        self.top_prefixes_ = None
        self.columns_ = ['ticket_prefix', 'ticket_number']

    def fit(self, X, y=None):
        if 'ticket' not in X.columns:
            raise ValueError("ticket column not found")

        prefixes = X['ticket'].str.extract(r'^([a-z./]+)', expand=False).fillna('none')
        self.top_prefixes_ = prefixes.value_counts().head(self.top_k).index.tolist()
        return self

    def transform(self, X):
        X = X.copy()

        X['ticket_prefix'] = X['ticket'].str.extract(r'^([a-z./]+)', expand=False).fillna('none')
        X['ticket_prefix'] = X['ticket_prefix'].apply(
            lambda x: x if x in self.top_prefixes_ else 'rare'
        )

        X['ticket_number'] = (
            X['ticket']
            .str.extract(r'(\d+)$', expand=False)
            .astype(float)
            .fillna(0)
        )

        return X[self.columns_]


# --------------------------------------------------
# FARE
# --------------------------------------------------
class FareBinning(Base):

    def __init__(self, method='quantile', bins=None, labels=None, q=4):
        self.method = method
        self.bins = bins
        self.labels = labels
        self.q = q
        self.bin_edges_ = None
        self.labels_ = None
        self.columns_ = ['fare_bin']

    def fit(self, X, y=None):
        if 'fare' not in X.columns:
            raise ValueError("fare column not found")

        if self.method == 'manual':
            if self.bins is None:
                raise ValueError("For manual method, bins must be provided")
            self.bin_edges_ = self.bins

        elif self.method == 'quantile':
            self.bin_edges_ = np.unique(
                X['fare'].quantile(np.linspace(0, 1, self.q + 1))
            )

        else:
            raise ValueError("method must be 'manual' or 'quantile'")

        if self.labels is None:
            self.labels_ = [f'bin{i+1}' for i in range(len(self.bin_edges_) - 1)]
        else:
            self.labels_ = self.labels

        return self

    def transform(self, X):
        X = X.copy()

        X['fare_bin'] = pd.cut(
            X['fare'],
            bins=self.bin_edges_,
            labels=self.labels_,
            include_lowest=True
        )

        return X[self.columns_]


# --------------------------------------------------
# AGE
# --------------------------------------------------
class AgeImmputer(Base):

    def __init__(self):
        self.mean_ = None
        self.columns_ = ['age']

    def fit(self, X, y=None):
        if 'age' not in X.columns:
            raise ValueError("age column not found")
        self.mean_ = X['age'].mean()
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("The transformer is not fitted yet")
        X = X.copy()
        X['age'] = X['age'].fillna(self.mean_)
        return X[self.columns_]


# --------------------------------------------------
# EMBARKED
# --------------------------------------------------
class EmbarkedEncoder(Base):

    def __init__(self):
        self.mode_ = None
        self.columns_ = ['is_s']

    def fit(self, X, y=None):
        if 'embarked' not in X.columns:
            raise ValueError("embarked column not found")
        self.mode_ = X['embarked'].mode()[0]
        return self

    def transform(self, X):
        X = X.copy()
        X['embarked'] = X['embarked'].fillna(self.mode_)
        X['is_s'] = (X['embarked'] == 's').astype(int)
        return X[self.columns_]