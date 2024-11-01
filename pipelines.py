from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import (
    AdultsBinaryTransformer,
    SpecialRequestsBinaryTransformer,
    DistributionChannelBinaryTransformer
)

def create_pipeline():
    feature_engineering = Pipeline(steps=[
        ('adults_binary', AdultsBinaryTransformer()),
        ('special_requests_binary', SpecialRequestsBinaryTransformer()),
        ('distribution_channel_binary', DistributionChannelBinaryTransformer())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('reserved_room_type_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['reserved_room_type']),
        ('market_segment_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['market_segment']),
        ('country', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['country']),
        ('country_missing', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['country_missing']),
        ('avg_daily_rate_scaler', StandardScaler(), ['average_daily_rate'])
    ], remainder='drop') # drop all other features not selected

    pipeline = Pipeline(steps=[
        ('feature_engineering', feature_engineering),
        ('preprocessing', preprocessor)
    ])

    return pipeline
