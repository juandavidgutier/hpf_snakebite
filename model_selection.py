import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner, ForestDRLearner, LinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import expon
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
from dowhy import CausalModel
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier
from dowhy.causal_estimator import CausalEstimate
from sklearn.preprocessing import StandardScaler
from econml.dr import DRLearner
from sklearn.linear_model import LassoCV
from econml.dml import DML, SparseLinearDML

# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')

# Import data
# Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Load the CSV file from your Drive
import pandas as pd

# Path to the file in Google Drive
file_path = 'D:/data.csv'  # Adjust the path if it is in a folder
data_all = pd.read_csv(file_path, encoding='latin-1')

data_all = data_all.dropna()

columns_to_drop = ['Total_pop', 'altitude', 'cases_rural', 'exp_rural',
                   'cases_urban', 'exp_urban']

# 1. Label Encoding DANE
label_encoder_dane = LabelEncoder()
data_all['DANE_encoded'] = label_encoder_dane.fit_transform(data_all['DANE'])
minmax_scaler_dane = MinMaxScaler()
data_all['DANE_normalized'] = minmax_scaler_dane.fit_transform(
    data_all[['DANE_encoded']]
)

# 2. Label Encoding Department_DANE
label_encoder_year = LabelEncoder()
data_all['DANE_year_encoded'] = label_encoder_year.fit_transform(data_all['DANEYear'])
minmax_scaler_year = MinMaxScaler()
data_all['DANE_year_normalized'] = minmax_scaler_year.fit_transform(
    data_all[['DANE_year_encoded']]
)

data_all.drop(columns=columns_to_drop, inplace=True)

std_hfp_rural = data_all['HFP_rural'].std()
print(f"std of HFP_rural: {std_hfp_rural}")

median_hfp_rural = data_all['HFP_rural'].median()
print(f"median of HFP_rural: {median_hfp_rural}")

standard_scaler = StandardScaler()
data_all['Temperature'] = standard_scaler.fit_transform(data_all[['Temperature']])
data_all['Rainfall'] = standard_scaler.fit_transform(data_all[['Rainfall']])
data_all['EVI'] = standard_scaler.fit_transform(data_all[['EVI']])
data_all['Forest'] = standard_scaler.fit_transform(data_all[['Forest']])
data_all['Deforest'] = standard_scaler.fit_transform(data_all[['Deforest']])
data_all['Fire'] = standard_scaler.fit_transform(data_all[['Fire']])
data_all['Mining'] = standard_scaler.fit_transform(data_all[['Mining']])
data_all['Coca'] = standard_scaler.fit_transform(data_all[['Coca']])
data_all['HFP_rural'] = standard_scaler.fit_transform(data_all[['HFP_rural']])

# Standardize data
data_standardized = data_all

# Ensure correct temporal order
data_standardized = data_standardized.sort_values(
    by=['DANE_normalized', 'DANE_year_normalized']
).reset_index(drop=True)

# =========================
# Variables at t0 (lags)
# =========================

# Deforest: previous state of Deforestation
data_standardized['Deforest_t0'] = (
    data_standardized
    .groupby('DANE_normalized')['Deforest']
    .shift(1)
)

# Fire_t0: previous state of Fire
data_standardized['Fire_t0'] = (
    data_standardized
    .groupby('DANE_normalized')['Fire']
    .shift(1)
)

data_standardized = data_standardized.dropna()

data_standardized = data_standardized[['DANE_normalized', 'DANE_year_normalized',
                                       'Temperature', 'Rainfall', 'EVI', 'Forest', 'Deforest', 'Fire', 'Mining', 'Coca', 'MPI_rural',
                                       'Fire_t0', 'Deforest_t0', 
                                       'HFP_rural', 'excess']]

causal_model_hfp_rural = CausalModel(
    data=data_standardized,
    treatment=['HFP_rural'],
    outcome=['excess'],
    graph="""
    digraph {
        
        Rainfall -> Temperature;
        Rainfall -> Forest;
        Rainfall -> HFP_rural;
        Rainfall -> excess;
        
        Temperature -> Forest;
        Temperature -> HFP_rural;
        Temperature -> excess;
        
        Forest -> EVI;
        Forest -> Coca;
        Forest -> HFP_rural;
        Forest -> excess;
        
        EVI -> HFP_rural;
        EVI -> excess;
        
        Deforest_t0 -> Forest;
        Deforest_t0 -> Fire_t0;
        Deforest_t0 -> HFP_rural;
        Deforest_t0 -> excess;
        
        MPI_rural -> Deforest_t0;
        MPI_rural -> Coca;
        MPI_rural -> Mining;
        MPI_rural -> Fire_t0;
        MPI_rural -> HFP_rural;
        MPI_rural -> excess;
        
        Coca -> HFP_rural;
        Coca -> excess;
        
        Mining -> Forest;
        Mining -> HFP_rural;
        Mining -> excess;
        
        Fire_t0 -> HFP_rural;
        Fire_t0 -> excess;
        
        DANE_normalized -> excess;
        DANE_year_normalized -> excess;
        
        HFP_rural -> excess
        
    }
    """
)

from PIL import Image
import matplotlib.pyplot as plt

# Generate the model graph
causal_model_hfp_rural.view_model()

# Identifying effects
identified_estimand_hfp_rural = causal_model_hfp_rural.identify_effect(proceed_when_unidentifiable=None)
print(identified_estimand_hfp_rural)

from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV

# ============================================================================
# WRAPPER FOR PROBABILISTIC CLASSIFIERS
# ============================================================================

class ProbClassifierWrapper(BaseEstimator):
    """
    Wrapper for sklearn classifiers that returns probabilities in predict().

    For binary outcomes, EconML expects model_y.predict(X) to return
    E[Y|X] as continuous probabilities, not discrete classes.
    """
    def __init__(self, base_classifier=None, use_calibration=True, random_state=123):
        if base_classifier is None:
            base_classifier = XGBClassifier(
                n_estimators=200,
                n_jobs=1,
                random_state=random_state,
                class_weight='balanced'
            )
        self.base_classifier = base_classifier
        self.use_calibration = use_calibration
        self.random_state = random_state
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        """Fits the base classifier (with or without calibration)"""
        y = np.asarray(y).ravel()

        if self.use_calibration:
            self.model_ = CalibratedClassifierCV(
                estimator=clone(self.base_classifier),
                cv=3
            )
            self.model_.fit(X, y)
        else:
            self.model_ = clone(self.base_classifier)
            self.model_.fit(X, y)

        self._is_fitted = True
        return self

    def predict(self, X):
        """Returns positive class probabilities (P(Y=1|X))"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before calling predict()")
        return self.model_.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        """Returns the complete probability matrix"""
        if not self._is_fitted:
            raise ValueError("ProbClassifierWrapper must be fitted before calling predict_proba()")
        return self.model_.predict_proba(X)


# Required imports
from xgboost import XGBRegressor, XGBClassifier
from econml.score import RScorer
import numpy as np

# ProbClassifierWrapper is assumed to be defined and available from a previous cell.

# Define fixed parameters for XGBoost
xgb_fixed_parameters = {
    "random_state": 123,
    "eta": 0.0001,
    "reg_lambda": 1.5,
    "alpha": 0.001,
}

# Define model configurations to test
model_configurations = [
    {"name": "Model 1", "n_estimators": 10, "max_depth": 2},
    {"name": "Model 2", "n_estimators": 10, "max_depth": 3},
    {"name": "Model 3", "n_estimators": 50, "max_depth": 3},
    {"name": "Model 4", "n_estimators": 50, "max_depth": 4},
    {"name": "Model 5", "n_estimators": 100, "max_depth": 2},
    {"name": "Model 6", "n_estimators": 100, "max_depth": 3},
    {"name": "Model 7", "n_estimators": 150, "max_depth": 2},
    {"name": "Model 8", "n_estimators": 150, "max_depth": 3},
    {"name": "Model 9", "n_estimators": 200, "max_depth": 3},
    {"name": "Model 10", "n_estimators": 200, "max_depth": 4}
]

# Prepare data (already existing in the cell)
Y = data_standardized['excess'].values.astype(int)
T = data_standardized['HFP_rural'].values
X = data_standardized[['MPI_rural', 'DANE_normalized', 'DANE_year_normalized']].values
W = data_standardized[['Forest', 'Coca', 'Temperature', 'Rainfall',
                       'Mining', 'EVI', 'MPI_rural',
                       'Fire_t0', 'Deforest_t0']].values

# Clean NaNs (already existing in the cell)
valid_mask = (~np.isnan(Y)) & (~np.isnan(T))
for arr in (X, W):
    valid_mask = valid_mask & (~np.isnan(arr).any(axis=1))

if not valid_mask.all():
    idx = np.where(valid_mask)[0]
    Y = Y[idx]
    T = T[idx]
    X = X[idx, :]
    W = W[idx, :]

print(f"Shapes after filtering: Y={Y.shape}, T={T.shape}, X={X.shape}, W={W.shape}")

rscore_results = []

print("\n============================================================")
print("Calculating R-Score for multiple DML models")
print("============================================================")

for config in model_configurations:
    model_name = config["name"]
    n_est = config["n_estimators"]
    depth = config["max_depth"]

    print(f"\n--- Evaluating {model_name} (n_estimators={n_est}, max_depth={depth}) ---")

    # 1) Define nuisance models (XGBoost)
    nuisance_model_y = XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        **xgb_fixed_parameters,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    # Wrap for probability prediction
    wrapped_model_y = ProbClassifierWrapper(
        base_classifier=nuisance_model_y, 
        use_calibration=True, 
        random_state=xgb_fixed_parameters["random_state"]
    )

    nuisance_model_t = XGBRegressor(
        n_estimators=n_est,
        max_depth=depth,
        **xgb_fixed_parameters,
        n_jobs=1
    )

    # 2) Instantiate and fit the causal DML model
    dml_estimator = DML(
        model_y=wrapped_model_y,
        model_t=nuisance_model_t,
        model_final=LassoCV(
            alphas=[0.0001, 0.001, 0.005, 0.05, 0.01, 0.1],
            fit_intercept=False,
            max_iter=50000,
            tol=1e-3,
            cv=3,
            n_jobs=-1
        ),
        featurizer=PolynomialFeatures(degree=3, include_bias=False),
        discrete_outcome=True,
        discrete_treatment=False,
        cv=3,
        random_state=xgb_fixed_parameters["random_state"]
    )
    
    print("Fitting DML...")
    dml_estimator.fit(Y=Y, T=T, X=X, W=W)
    print("DML fitted.")

    # 3) Calculate R-Score using the .score() method of the dml_estimator itself
    # This is preferable since the dml_estimator is already correctly configured for discrete_outcome
    print("Calculating R-Score with dml_estimator.score()...")
    r_score_value = dml_estimator.score(Y=Y, T=T, X=X, W=W)

    rscore_results.append((model_name, r_score_value))
    print(f"  R-Score for {model_name}: {r_score_value:.6f}")

print("\n============================================================")
print("Final R-Score Results:")
print("============================================================")
for name, score in rscore_results:
    print(f"  {name}: {score:.6f}")
print("============================================================")
