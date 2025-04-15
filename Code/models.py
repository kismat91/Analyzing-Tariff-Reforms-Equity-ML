#%% 
import numpy as np
def recode_s10q01(x):
    if pd.isna(x):
        return np.nan
    val = str(x).strip().lower()
    if val in ['1', '1.0', 'yes']:
        return 1
    elif val in ['2', '2.0', 'no']:
        return 0
    else:
        return np.nan

#%%

import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

df_gsec10_1_final = pd.read_csv(r"..\Datasets\Uganda\output_gsec10_1_all_years.csv")
df_gsec10_3_final = pd.read_csv(r"..\Datasets\Uganda\output_gsec10_3_all_years.csv")
tariff_df = pd.read_csv(r"..\Datasets\Uganda\tariffs.csv")
df_gsec10_1_final['s10q01'] = df_gsec10_1_final['s10q01'].apply(recode_s10q01)
print("Tariff data columns:", tariff_df.columns)
years_of_interest = [2013, 2015, 2018, 2019]
tariff_filtered = tariff_df[tariff_df['Year'].isin(years_of_interest)]
tariff_filtered['Domestic'] = pd.to_numeric(tariff_filtered['Domestic'], errors='coerce')
tariff_yearly = tariff_filtered.groupby('Year', as_index=False)['Domestic'].mean()
tariff_yearly.rename(columns={'Domestic': 'tariff_rate'}, inplace=True)
print("Yearly tariff rates:\n", tariff_yearly)

df_gsec10_1_final = pd.merge(df_gsec10_1_final, tariff_yearly, left_on='year', right_on='Year', how='left')
df_gsec10_1_final.drop(columns=['Year'], inplace=True)
df_gsec10_3_final = pd.merge(df_gsec10_3_final, tariff_yearly, left_on='year', right_on='Year', how='left')
df_gsec10_3_final.drop(columns=['Year'], inplace=True)


#%% 
# # Model A: Impact of Tariff Changes on Household Electricity Expenditure
#
# Model:
#   s10q05a = β₀ + β₁ * tariff_rate + β₂ * s10q02 + β₃ * s10q01 + ε
#
# Variables:
#   s10q05a: Electricity expenditure
#   tariff_rate: Quarterly tariff level (external/merged)
#   s10q02: Power hours available per day
#   s10q01: Grid access (assumed binary: 1 for yes, 0 for no)
# ----------------------------
model_A = smf.ols("s10q05a ~ tariff_rate + s10q02 + s10q01", data=df_gsec10_1_final).fit()
print("Model A: Electricity Expenditure")
print(model_A.summary())

# %%

# ----------------------------
# Model B: Fuel Switching and Energy Substitution
#
# We use a multinomial logit model to examine the determinants of the stove type (s10q09).
# Model:
#   P(StoveType = k) = f(tariff_rate, s10q01, s10q06, s10q05a)
#
# Here, we assume that s10q09 represents the stove used most often.
# For MNLogit, we need the dependent variable as numeric codes.
# ----------------------------
# Convert stove type to categorical codes
df_gsec10_1_final['stove_type_code'] = df_gsec10_1_final['s10q09'].astype('category').cat.codes

# Prepare independent variables (with constant)
X_B = df_gsec10_1_final[['tariff_rate', 's10q01', 's10q06', 's10q05a']].copy()
X_B = sm.add_constant(X_B)
X_B = X_B.apply(pd.to_numeric, errors='coerce')
y_B = df_gsec10_1_final['stove_type_code']
y_B = pd.to_numeric(y_B, errors='coerce')
X_B = X_B.dropna()
y_B = y_B.loc[X_B.index] 
model_B = sm.MNLogit(y_B, X_B).fit(method='newton', maxiter=100, disp=False)
print("\nModel B: Stove Type Multinomial Logit")
print(model_B.summary())
#%%
# ----------------------------
# Model C: Generator Use as an Adaptive Strategy
#
# Model:
#   s10q06 = β₀ + β₁ * tariff_rate + β₂ * s10q02 + β₃ * s10q05a + ε
#
# Here, s10q06 is a binary variable indicating generator use.
# We use a logistic regression (Logit model).
# ----------------------------
X_C = df_gsec10_1_final[['tariff_rate', 's10q02', 's10q05a']].copy()
X_C = sm.add_constant(X_C)
y_C = df_gsec10_1_final['s10q06']
y_C = df_gsec10_1_final['s10q06'].apply(
    lambda x: 1 if str(x).strip().lower() in ['1', '1.0', 'yes'] else (
        0 if str(x).strip().lower() in ['2', '2.0', 'no'] else np.nan
    )
)
X_C= X_C.dropna()
y_C = y_C.loc[X_C.index]  
model_C = sm.Logit(y_C, X_C).fit(disp=False)
print("\nModel C: Generator Use (Logit)")
print(model_C.summary())
#%%
# ----------------------------
# Model D: Electricity Affordability Index (EAI)
#
# Define:
#   EAI = (s10q05a + s10q07a) / hh_income
#
# Then, regress EAI on tariff_rate and grid access (s10q01) (and possibly other controls).
# ----------------------------
if 'hh_income' in df_gsec10_1_final.columns:
    df_gsec10_1_final['EAI'] = (df_gsec10_1_final['s10q05a'] + df_gsec10_1_final['s10q07a']) / df_gsec10_1_final['hh_income']
    model_D = smf.ols("EAI ~ tariff_rate + s10q01", data=df_gsec10_1_final).fit()
    print("\nModel D: Electricity Affordability Index (EAI)")
    print(model_D.summary())
else:
    print("\nModel D: hh_income variable not found. Please merge household income data to compute the Electricity Affordability Index (EAI).")

#%%
# XGBoost Regression Model: Predict electricity expenditure using features.
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

features = ['tariff_rate', 's10q02', 's10q01']
target = 's10q05a'

print("Missing values in features:")
print(df_gsec10_1_final[features].isnull().sum())
df_gsec10_1_final = df_gsec10_1_final.dropna(subset=['s10q05a'])

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_gsec10_1_final[features], df_gsec10_1_final[target], test_size=0.2, random_state=42)
dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
dtest = xgb.DMatrix(X_test, label=y_test, missing=np.nan)

# Setting parameters for XGBoost regression
params = {
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'eta': 0.1,
    'seed': 42
}

# Training the XGBoost model
num_boost_round = 100
model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

preds = model.predict(dtest)
rmse = mean_squared_error(y_test, preds, squared=False)
print("XGBoost RMSE:", rmse)

# feature importance plot for XGBoost
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.title("Feature Importance")
plt.show()

# %%
# Difference-in-Differences (DID) Regression Model:
# This model examines the effect of reform (post-2018) on electricity expenditure.
import pandas as pd
import statsmodels.formula.api as smf

# post-intervention indicator: 1 if year >= 2018, else 0
df_gsec10_1_final['post'] = (df_gsec10_1_final['year'] >= 2018).astype(int)
# Defining treated group based on grid access (s10q01 == 1)
df_gsec10_1_final['treated'] = df_gsec10_1_final['s10q01']
# Creating the interaction term for DID (treated × post)
df_gsec10_1_final['treated_post'] = df_gsec10_1_final['treated'] * df_gsec10_1_final['post']

# DID regression controlling for power hours (s10q02)
did_formula = "s10q05a ~ treated + post + treated_post + s10q02"
did_model = smf.ols(did_formula, data=df_gsec10_1_final).fit()
print("Difference-in-Differences (DID) Model Summary:")
print(did_model.summary())

# %%
# Causal Forest using econml:
# This model estimates individual treatment effects (effect of post-reform) on electricity expenditure.
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

# Defining outcome, treatment, and covariates
Y = df_gsec10_1_final['s10q05a'].values
T = df_gsec10_1_final['post'].values  # treatment indicator: post reform indicator
X = df_gsec10_1_final[['s10q02', 'tariff_rate', 's10q01']].values


causal_forest = CausalForestDML(
    model_t=RandomForestRegressor(n_estimators=100, random_state=42),
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    n_estimators=100,
    random_state=42
)
causal_forest.fit(Y, T, X=X)

# treatment effects and display summary statistics
treatment_effects = causal_forest.effect(X)
print("Mean Estimated Treatment Effect (Post Reform):", np.mean(treatment_effects))
print("First 10 individual treatment effects:\n", treatment_effects[:10])

# %%
# Second Causal Forest Block using econml:
# This block estimates treatment effects using a causal forest on the post-reform indicator.
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

Y = df_gsec10_1_final['s10q05a'].values
T = (df_gsec10_1_final['year'] >= 2018).astype(int).values
X = df_gsec10_1_final[['s10q02', 'tariff_rate', 's10q01']].values
causal_forest = CausalForestDML(model_t=RandomForestRegressor(n_estimators=100, random_state=42),
                                model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                                n_estimators=100, random_state=42)
causal_forest.fit(Y, T, X=X)
print("Mean Estimated Treatment Effect:", np.mean(causal_forest.effect(X)))

# %%
# LightGBM Regression Model:
# Uses LightGBM to predict electricity expenditure with basic features.
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

features = ['tariff_rate', 's10q02', 's10q01']
target = 's10q05a'

df_model = df_gsec10_1_final.dropna(subset=[target])


X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train, free_raw_data=False)

# initial parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1
}

# Train with early stopping
num_boost_round = 500
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=num_boost_round,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=['train', 'valid'])
                               
# Predicting and evaluating the model using RMSE and R^2
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("LightGBM RMSE:", rmse)
print("LightGBM R^2:", r2)

# %%
# Tuned LightGBM Regression Model:
# Uses GridSearchCV to find the best hyperparameters for LightGBM.
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

features = ['tariff_rate', 's10q02', 's10q01']  
target = 's10q05a'


df_model = df_gsec10_1_final.dropna(subset=[target] + features)

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500]
}

lgb_estimator = lgb.LGBMRegressor(objective='regression', random_state=42)
grid_search = GridSearchCV(lgb_estimator, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Tuned LightGBM RMSE:", rmse, "R^2:", r2)

# %%
# Panel DID and Improved LightGBM Regression Model:

#----------------------------------------------------------------
# 2. Feature Engineering
#----------------------------------------------------------------
# Create a log-transformed expenditure variable to reduce skewness
df_gsec10_1_final['log_expenditure'] = np.log(df_gsec10_1_final['s10q05a'] + 1)

df_gsec10_1_final['cost_per_hour'] = df_gsec10_1_final.apply(
    lambda row: row['s10q05a'] / row['s10q02'] if row['s10q02'] > 0 else np.nan, axis=1
)
df_gsec10_1_final['post'] = (df_gsec10_1_final['year'] >= 2018).astype(int)
df_gsec10_1_final['treated'] = df_gsec10_1_final['s10q01']
df_gsec10_1_final['treated_post'] = df_gsec10_1_final['treated'] * df_gsec10_1_final['post']

#----------------------------------------------------------------
# 3. Panel DID Model Using Fixed Effects (Commented Out)
#----------------------------------------------------------------
# The following block is commented because the DID part is not used.
# It shows how to set up a panel DID model with fixed effects.
#
# df_panel = df_gsec10_1_final.dropna(subset=['s10q05a', 's10q02', 's10q01', 'post', 'treated_post']).set_index(['hhid', 'year'])
#
# # Define exogenous variables for the panel model (drop time-invariant variables)
# exog_vars = ['post', 'treated_post', 's10q02']
# exog = sm.add_constant(df_panel[exog_vars])
#
# # Fit the Panel DID model with fixed effects and clustered standard errors
# panel_model = PanelOLS(df_panel['s10q05a'], exog, entity_effects=True, drop_absorbed=True, check_rank=False)
# panel_res = panel_model.fit(cov_type='clustered', cluster_entity=True)
# print("Panel DID Model (Fixed Effects) Summary:")
# print(panel_res.summary)

#----------------------------------------------------------------
# 4. Improved LightGBM Regression Model with Additional Features
#----------------------------------------------------------------
# Usinglog-transformed target and additional features for better prediction performance.
features = ['tariff_rate', 's10q02', 's10q01', 'cost_per_hour', 'post', 'treated_post']
target = 'log_expenditure'
model_data = df_gsec10_1_final.dropna(subset=[target] + features)
X = model_data[features]
y = model_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'num_leaves': [31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500]
}
lgb_estimator = lgb.LGBMRegressor(objective='regression', random_state=42)
grid_search = GridSearchCV(lgb_estimator, param_grid, cv=5, 
                           scoring='neg_root_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)
print("Best LightGBM Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Tuned LightGBM RMSE (log-scale):", rmse, "R^2:", r2)

# feature importance for the best LightGBM model
best_model = grid_search.best_estimator_
lgb.plot_importance(best_model)
plt.title("LightGBM Feature Importance")
plt.show()


# %%
import shap

# Explainer for the tuned LightGBM model
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Bar plot: Mean absolute SHAP values for feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Summary plot: Detailed overview of the impact of each feature on the predictions
shap.summary_plot(shap_values, X_test)

# Dependence plot: How the value of "tariff_rate" affects the prediction along with its interaction effects
shap.dependence_plot("tariff_rate", shap_values, X_test)

# Force plot: Visual explanation for the first test instance 
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])



# %%
