import polars as pl
import functime.feature_reduction._utils as _utils
import logging
import sys
from functime.feature_reduction._feat_calculator import FeatureCalculator
from functime.feature_reduction._dim_reducer import DimensionReducer
import plotly.express as px


logger = logging.getLogger(__name__)


class features_dim_reduction:
    def __init__(self, col_values: str, model: str = "PCA", precompute_feat: str = ""):
        self.feature_calculator = FeatureCalculator(col_values)
        self.dimension_reducer = DimensionReducer()
        self.model = model

        if precompute_feat == "small":
            self.feature_calculator.add_multi_features(
                _utils.get_small(col_values)
            )
        elif precompute_feat == "medium":
            self.feature_calculator.add_multi_features(
                _utils.get_medium(col_values)
            )
        elif precompute_feat == "large":
            self.feature_calculator.add_multi_features(
                _utils.get_large(col_values)
            )

    def add_feature(self, feature: pl.Expr):
        self.feature_calculator.add_feature(feature)
        return self

    def add_multi_features(self, features: list[pl.Expr]):
        self.feature_calculator.add_multi_features(features)
        return self

    def calculate_features(self, X: pl.DataFrame) -> pl.DataFrame:
        return self.feature_calculator.calculate_features(X = X)

    def X_features(self)-> pl.DataFrame:
        return self.feature_calculator.X_features

    def X_reduced(self)-> pl.DataFrame:
        id = self.feature_calculator.X_features.columns[0]
        return self.dimension_reducer.state_model.transform(
            self.feature_calculator.X_features.select(pl.exclude(id))
        )

    def fit(self, X: pl.DataFrame, dim: int = 2, **kwargs):
        id = X.columns[0]
        X_features = (
            self.calculate_features(X)
            .select(
                pl.exclude(id)
            )
        )
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(X_features, dim, **kwargs)
            return self.dimension_reducer.state_model
        elif self.model == "TSNE":
            pass
        else:
            logger.info(
                "The dimension algorithm requested has not been implemented yet."
            )
            sys.exit(1)

    def fit_transform(self, X: pl.DataFrame, dim: int = 2, **kwargs) -> pl.DataFrame:
        id = X.columns[0]
        X_features = (
            self.calculate_features(X)
            .select(
                pl.exclude(id)
            )
        )
        if self.model == "PCA":
            self.dimension_reducer.fit_pca(X_features, dim, **kwargs)
            return self.dimension_reducer.state_model.transform(X_features)
        elif self.model == "TSNE":
            pass
        else:
            logger.info(
                "The dimension algorithm requested has not been implemented yet."
            )
            sys.exit(1)


df = pl.read_parquet("data/commodities.parquet").with_columns(
    (pl.col("price")-pl.col("price").mean())/pl.col("price").std()
).filter(
    pl.col("commodity_type").is_in(["Tin", "Nickel", "Copper"]).not_()
)


print(df)
ts_proc = features_dim_reduction(model = "PCA", col_values = "price", precompute_feat="medium")

fitted_pca = (
    ts_proc
    .fit(X = df, dim = 3)
)

print(fitted_pca)

# # Use sklearn parameters
print(fitted_pca.explained_variance_ratio_)

# Get the X_reduced
X_reduced = ts_proc.X_reduced()
print(X_reduced)

# Get the table of the features
X_feat = ts_proc.X_features()

df_pca = pl.DataFrame({
    "id": X_feat["commodity_type"],
    "PC1": X_reduced[:, 0], 
    "PC2": X_reduced[:, 1],
    "PC3": X_reduced[:, 2]
})

fig = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='id', title='PCA 3D Plot with Tickers')

fig.show()

