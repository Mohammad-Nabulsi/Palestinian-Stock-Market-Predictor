#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =======================
# Core
# =======================
import numpy as np
import pandas as pd
import os
import importlib
import sys
from pathlib import Path
# For Development
import joblib  

# =======================
# Model Selection & CV
# =======================
from sklearn.model_selection import (
    TimeSeriesSplit,
    GridSearchCV,
    RandomizedSearchCV
)
##Prolly to be deleted
from sklearn.base import clone
from scipy.stats import uniform, randint  

# =======================
# Calibration
# =======================

# =======================
# Metrics
# =======================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)


# In[2]:


PROJECT_ROOT = Path().resolve().parents[0]
sys.path.append(str(PROJECT_ROOT))


# In[3]:


import scripts.utils as utils
from scripts.utils import (
    create_and_plot as cp,
    take_stock,
    normalize_range,
    add_temporal_features
    )



# In[4]:


from scripts.preprocess import (
    datize_date, 
    numerize_value_and_volume, 
    remove_nas, 
    time_series_split, 
    preprocessor
)



# In[5]:


from scripts.visualize import plot_feature_distributions, plot_stocks_closing_200MA
from scripts.create_features import fix_stock_change_pctg_leakge


# In[6]:


from scripts.model import print_validation_scores, print_test_score, calibrate_and_plot



# # 0. `Load the data and see summary`

# `Load Dataframe`

# In[7]:


FILE_PATH = os.path.join(os.getcwd(), "..", "FINAL_STOCKS.csv")


# In[8]:


df = pd.read_csv(FILE_PATH)
df.tail()


# `Data summary`

# In[9]:


df.info()


# In[10]:


df.isna().sum()


# In[11]:


df.describe()


# `Reorder columns`

# In[12]:


df.rename(columns={'max_price':'highest', 'min_price':'lowest'}, inplace=True)
df.columns


# In[13]:


desired_order = ['stock', 'date'] + [ col for col in df.columns if col not in ['stock', 'date'] ]
desired_order


# In[14]:


df = df[desired_order]
df.tail()


# # `1. EDA && Feature Engineering & Preprocessing`

# ## 1.1 `Standard EDA and Preprocessing`

# `Datize the date column`

# In[15]:


df = datize_date(df)
df.dtypes


# `See which stocks ahs the largest volume and #of trades`

# In[16]:


df.dtypes


# `Numerize the (value, volume) columns`

# In[17]:


df = numerize_value_and_volume(df)
df[['value','volume']].dtypes


# In[18]:


df.groupby('stock')['volume'].sum().sort_values(ascending=False).head().to_frame()


# In[19]:


df.groupby('stock')['n_deals'].sum().sort_values(ascending=False).head().to_frame()


# `Set stock, date as the index`

# In[20]:


df.set_index('stock', inplace=True)
df.index.unique()


# In[21]:


plot_stocks_closing_200MA(df)


# `Working with individual stocks`

# In[22]:


bop = take_stock(df, 'BoP')


# `As we can see from the first graph we cacn tell that opening and closing prices are almost always identical indicating the low volatility in PEX in general`
# 
# ----------------------------------

# `The initial model we will try is to predict the direction of the movement for stock`

# ## 1.2 `Creating informative variables`

# `First of all create the movemetn direction target variable`

# `Create 1 Day Classification target variable with concern for predicting up movements`

# ### 1.2.1 `Create The Target Variable`

# In[23]:


bop = cp(bop, "target", duration=20)


# `Class 0 occurs much more than class 1, so the dataset is imbalanced.`
# 
# `Models may bias toward predicting “no meaningful up move” unless handled. (class weight = balanced)`

# ### 1.2.2 `Create Time Since Last Trade Variable`

# In[24]:


bop = cp(bop, "time_since_last")


# `Most trades happen with very short gaps (0–2 days).`
# `Long inactivity periods are rare and form a long right tail.`
# 

# ### 1.2.3 `Create Moving Averages`

# In[25]:


bop = cp(bop, "moving_averages")


# `This chart makes total sense since PEX ahs very low volatility which is why even the 20MA almost perfectly hugs the closign price`

# `MA won't reveal much information and alone is not enough for ML model to revela hideen patterns we need to create more features`

# ### 1.2.4 `Create Range Variable`

# In[26]:


bop = cp(bop, "target", plot=False, corr=True)
bop = cp(bop, "range")


# Higher volatility generally appears when trading activity increases.
# No clean class separation, so predictive power alone is weak.
# 

# ### 1.2.5 `Crate Daily Returns Variables`

# In[27]:


bop = cp(bop, "returns")


# `As we can see the returns distibution has fatter tails on the longer run which makes total sense for a low volatility market`

# ### 1.2.6 `volatility X days`

# In[28]:


bop = cp(bop, "volatility")


# `For reference comparing BoP with oreedo who is riskier`

# In[29]:


ord = take_stock(df, 'oreedo', plot=False)


# In[30]:


ord = cp(ord, "returns", plot=False, corr=False)
ord = cp(ord, "volatility", corr=False)


# Short- and medium-term volatility move together and spike during unstable periods.
# Extreme volatility is rare and usually short-lived.
# Volatility tends to rise when the price is below its long-term trend.
# 

# ### 1.2.7 `RSI`

# In[31]:


bop = cp(bop, "rsi", plot=True)


# Trading activity fluctuates strongly over time with frequent sharp spikes.
# Most values stay between the lower and upper thresholds, indicating normal activity.
# Extreme bursts appear intermittently, suggesting event-driven trading periods.
# 

# ### 1.2.8 `Liquidity features`

# In[32]:


bop = cp(bop, "liquidity")


# Changes in the number of deals are usually small, with rare but sharp spikes.
# Most activity increases are modest, while extreme jumps are infrequent and event-driven.
# The distribution is heavily right-skewed, indicating occasional bursts of trading activity.
# 

# ---------------------------------------

# In[33]:


bop = normalize_range(bop)


# ### 1.2.9 `Temporal Features`

# In[34]:


bop = add_temporal_features(bop)


# ### 1.2.10 `MACD`

# In[35]:


bop = cp(bop, "macd")


# The lines move above and below zero, showing when movement speeds up or slows down.
# Bigger gaps mean stronger movement, while small gaps mean little change.
# Most of the time changes are small, with occasional short bursts of activity.
# 

# ### 1.2.11 `Stochastic`

# In[36]:


bop = cp(bop, "stochastic")


# In[37]:


plot_feature_distributions(bop, target_col=None)


# Most features are not normally distributed and show skewness or heavy tails.
# This indicates the presence of outliers and non-stationary behavior in the data.
# The distributions suggest that scaling or transformation may be necessary before modeling.
# 

# ### 1.3.0 `After training the model we found out that there is a leakge in the data where change pctg was calculated based on tomorrow's price isntead of today's`

# In[38]:


bop = fix_stock_change_pctg_leakge(bop, 'BoP')


# ## 1.3 `Dropping redundant/highly multi cllinearity features`

# In[39]:


corr = bop.corr()
pairs = [
    (i, j, corr.loc[i, j])
    for idx, i in enumerate(corr.columns)
    for j in corr.columns[idx + 1:]
    if abs(corr.loc[i, j]) > 0.85
]



# In[40]:


pairs


# In[41]:


bop.drop(columns=['opening', 'highest', 'lowest', '20_MA', '50_MA', '200_MA', 'volume', 'value'], inplace=True)


# ## 1.4 `Check for  noisy, missy (either logical or physical), inconsistent, and duplicated Data.`

# In[42]:


bop.duplicated().sum()


# In[43]:


bop.isna().sum()


# In[44]:


bop[199:].isna().sum()


# In[45]:


bop.describe()


# In[46]:


bop.drop(columns='clf_target_1d', inplace=True)


# `All of these makes total sense  except for volume_z which is a huge number for a z_score`

# `The entire preprocess, feature engineering and dropping initial workflow will look like this`
# 
# `1.Datize date`
# 
# `2.Set date as index`
# 
# `3.numerize volume`
# 
# `4.Create all the default features`
# 
# `5.Drop unnecessary features`

# -----------------------------------------------------------------------

# # `2.Pipeline Creation`

# In[47]:


df = remove_nas(bop)


# In[48]:


import scripts.utils


# In[49]:


from scripts.utils import nyears
df = nyears(df, 6)


# In[50]:


X_train, X_test, y_train, y_test, feature_cols, target_col = time_series_split(df)


# # 3. `Model Training`

# ## 3.1 `Logistic Regression`

# In[51]:


num_cols = [c for c in feature_cols if c not in ['first_week_of_month', 'day_of_week']]

sk = df[num_cols].skew()
mean_cols = [c for c in num_cols if abs(sk[c]) < 0.5]
median_cols = [c for c in num_cols if abs(sk[c]) >= 0.5]


# In[64]:


neg_cols = []
for c in num_cols:
    s = df[c]
    has_negative = (s < 0).any()
    if(has_negative):
        neg_cols.append(c)


# In[67]:


set(mean_cols).difference(set(neg_cols))


# In[71]:


mean_yeo_cols = set(mean_cols).intersection(set(neg_cols))
mean_log_cols = list(set(mean_cols) - mean_yeo_cols)
median_yeo_cols = set(median_cols).intersection(set(neg_cols))
median_log_cols = list(set(median_cols) - median_yeo_cols)


# In[79]:


preprocess = preprocessor(
    mean_log_cols=mean_log_cols,
    mean_ye_cols=list(mean_yeo_cols),
    median_log_cols=median_log_cols,
    median_ye_cols=list(median_yeo_cols)
)


# In[80]:


import scripts.model as model
importlib.reload(model)
from scripts.model import create_models


# In[81]:


lr_clf = create_models("lr", preprocess)['lr']


# In[82]:


import scripts.model as model
importlib.reload(model)
from scripts.model import print_validation_scores


# In[83]:


print_validation_scores(lr_clf, X_train, y_train)


# In[84]:


lr_clf.fit(X_train, y_train)


# In[85]:


test_proba = lr_clf.predict_proba(X_test)[:, 1]


# In[86]:


print_test_score(test_proba, y_test, 0.5)


# `Calibrate the model for better interpretability and making more informative relaible decisions according to the respectful threshold`

# ## 3.1.2 `Calibrate the model`

# In[94]:


#clr_lr, proba_uncal, proba_cal, calib_data = calibrate_and_plot(lr_clf, X_train, y_train, X_test, y_test)


# In[95]:


#print_test_score(proba_cal, y_test, 0.4)


# ## 3.2 `Linear SVC`

# In[96]:


lin_svc_clf = create_models("svc_lin", preprocess)['svc_lin']


# ## 3.3 `RBF SVC`

# In[97]:


rbf_svc_clf = create_models("svc_rbf", preprocess)['svc_rbf']


# ## 3.4 `RF`

# In[98]:


rf_clf = create_models("rf", preprocess)['rf']


# ## 3.5 `XGB`

# In[99]:


xgb_clf = create_models("xgb", preprocess)['xgb']


# `Compare`

# In[100]:


base_models = {
    "logistic": lr_clf,
    "svc_rbf": rbf_svc_clf,
    "svc_lin": lin_svc_clf,
    "random_forest": rf_clf,
    "xgboost": xgb_clf
}


# In[101]:


def compare_models_time_series(
    models,
    X,
    y,
    n_splits=5
):
    """
    Compare base models using TimeSeriesSplit.
    Metrics:
      - PR-AUC (primary)
      - ROC-AUC (secondary)

    Returns a sorted DataFrame.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for name, model in models.items():
        pr_aucs = []
        roc_aucs = []

        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)

            scores = m.predict_proba(X_va)[:, 1]

            pr_aucs.append(average_precision_score(y_va, scores))
            roc_aucs.append(roc_auc_score(y_va, scores))

        results.append({
            "model": name,
            "pr_auc_mean": np.mean(pr_aucs),
            "pr_auc_std": np.std(pr_aucs),
            "roc_auc_mean": np.mean(roc_aucs),
            "roc_auc_std": np.std(roc_aucs)
        })

    return (
        pd.DataFrame(results)
        .sort_values("pr_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


# In[102]:


results_df = compare_models_time_series(
    models=base_models,
    X=X_train,
    y=y_train,
    n_splits=5
)

print(results_df)


# `Choose XGB and LR (best performance xgb and before transformation logistic was best so keep using it and see which is better)`

# In[103]:


lr_param_grid = {
    "clf__C": np.logspace(-4, 2, 12),  
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"],
    "clf__class_weight": [None, "balanced"]
}

lr_search = GridSearchCV(
    lr_clf,
    param_grid=lr_param_grid,
    scoring="average_precision",
    cv=TimeSeriesSplit(n_splits=10),
    n_jobs=-1
)

lr_search.fit(X_train, y_train)
best_lr = lr_search.best_estimator_


# In[ ]:


neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos


# In[150]:


xgb_clf.set_params(
    clf__scale_pos_weight=scale_pos_weight
)
xgb_param_dist = {
    "clf__max_depth": randint(3, 5),              # narrower
    "clf__learning_rate": uniform(0.02, 0.05),
    "clf__n_estimators": randint(200, 600),       # fewer trees
    "clf__subsample": uniform(0.6, 0.3),
    "clf__colsample_bytree": uniform(0.6, 0.3),
    "clf__min_child_weight": randint(10, 25),
    "clf__reg_lambda": uniform(2.0, 6.0),
    "clf__reg_alpha": uniform(0.0, 1.0),
    "clf__gamma": uniform(0.0, 1.0),
}


xgb_search = RandomizedSearchCV(
    xgb_clf,
    param_distributions=xgb_param_dist,
    n_iter=80,                    
    scoring="precision",
    cv=TimeSeriesSplit(n_splits=5),
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_


# In[151]:


tuned_models = {
    "xgb": best_xgb,
    "lr": best_lr
}
results_df = compare_models_time_series(
    models=tuned_models,
    X=X_train,
    y=y_train,
    n_splits=5

)
print(results_df)


# ## continue with xgb

# In[152]:


xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
print_test_score(xgb_proba, y_test, 0.5)


# `Given that xgb gave the best perfromance across threhsolds what we can itnerpret fro mthis that given our class imbalacne the PR AUC score is solid and ROC AUC for a stock project is decent BUT using a ML model for trading remain a flawed approach even in  a low volatiltiy market like the palestinian market`

# 
