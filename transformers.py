from sklearn.base import BaseEstimator, TransformerMixin

class AdultsBinaryTransformer(BaseEstimator, TransformerMixin):
    """
    'adults_is_0': 1 if adults == 0, else 0
    'adults_is_1': 1 if adults == 1, else 0
    """
    def __init__(self, adults_col='adults'):
        self.adults_col = adults_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['adults_is_0'] = (X[self.adults_col] == 0).astype(int)
        X['adults_is_1'] = (X[self.adults_col] == 1).astype(int)
        return X

class SpecialRequestsBinaryTransformer(BaseEstimator, TransformerMixin):
    """
    'special_requests_is_0': 1 if total_of_special_requests == 0, else 0
    'special_requests_2_or_more': 1 if total_of_special_requests >= 2, else 0
    """
    def __init__(self, special_requests_col='total_of_special_requests'):
        self.special_requests_col = special_requests_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['special_requests_is_0'] = (X[self.special_requests_col] == 0).astype(int)
        X['special_requests_2_or_more'] = (X[self.special_requests_col] >= 2).astype(int)
        return X

class DistributionChannelBinaryTransformer(BaseEstimator, TransformerMixin):
    """
    'distribution_channel_corporate': 1 if distribution_channel == 'Corporate', else 0
    'distribution_channel_direct': 1 if distribution_channel == 'Direct', else 0
    """
    def __init__(self, distribution_channel_col='distribution_channel'):
        self.distribution_channel_col = distribution_channel_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['distribution_channel_corporate'] = (X[self.distribution_channel_col].str.lower() == 'corporate').astype(int)
        X['distribution_channel_direct'] = (X[self.distribution_channel_col].str.lower() == 'direct').astype(int)
        return X
    