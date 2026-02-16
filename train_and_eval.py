import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ========= targets =========
target_main = "Customer_return3Months"
target_aux = "Customer_return6Months"

df = pd.read_csv("Processed_Achareh_Orders_Sampled_Tehran_900000.csv")
# ========= y =========
y = df[target_main].astype(int)

# ========= X (remove both targets) =========
X = df.drop(columns=[target_main, target_aux])

# ========= remove useless column =========
if "Unnamed: 0" in X.columns:
    X = X.drop(columns=["Unnamed: 0"])

# ========= convert bool to int =========
bool_cols = X.select_dtypes("bool").columns
X[bool_cols] = X[bool_cols].astype(int)

# ========= categorical encoding =========
cat_cols = X.select_dtypes("object").columns
X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

# ========= fill missing =========
X = X.fillna(-1)

# ========= split data =========
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ========= model =========
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",      # fast for big data
    eval_metric="logloss",
    n_jobs=-1
)

# ========= train =========
model.fit(X_train, y_train)

# ========= evaluate =========
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Test Accuracy:", acc)
