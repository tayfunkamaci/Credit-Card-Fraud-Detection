import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#################################################################################
# 1. DOSYA YOLLARI
#################################################################################

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "creditcard.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#################################################################################
# 2. FEATURE ENGINEERING (TIME-AWARE)
#################################################################################

def create_features(df):
    df = df.copy()

    df["Amount_Log"] = np.log1p(df["Amount"])
    df["Time_Diff"] = df["Time"].diff().fillna(0)

    df["Hour"] = (df["Time"] // 3600) % 24
    df["Is_Night"] = df["Hour"].apply(lambda x: 1 if (x < 6 or x >= 22) else 0)

    pca_cols = [c for c in df.columns if c.startswith("V")]
    df["PCA_Abs_Mean"] = df[pca_cols].abs().mean(axis=1)
    df["PCA_Pos_Sum"] = df[pca_cols].clip(lower=0).sum(axis=1)
    df["PCA_Neg_Sum"] = df[pca_cols].clip(upper=0).sum(axis=1)

    df.drop(["Time", "Amount"], axis=1, inplace=True)
    return df

#################################################################################
# 3. MODEL OLUŞTURMA (CLASS WEIGHT)
#################################################################################

def get_model(y_train):

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos,
        random_state=17
    )

    lgbm = LGBMClassifier(
        class_weight="balanced",
        random_state=17,
        verbosity=-1
    )

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=17,
        max_depth=6,
        n_estimators=200
    )

    voting = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("lgbm", lgbm),
            ("rf", rf)
        ],
        voting="soft"
    )

    return voting

#################################################################################
# 4. MAIN
#################################################################################

def main():

    if not os.path.exists(DATA_PATH):
        print("Veri seti bulunamadı.")
        return

    print("Veri yükleniyor...")
    df = pd.read_csv(DATA_PATH)

    print("Zamana göre sıralanıyor...")
    df = df.sort_values("Time").reset_index(drop=True)

    #############################################################################
    # TIME-BASED SPLIT
    #############################################################################

    split_idx = int(len(df) * 0.8)

    df_train = df.iloc[:split_idx]
    df_test  = df.iloc[split_idx:]

    #############################################################################
    # FEATURE ENGINEERING (AYRI AYRI)
    #############################################################################

    print("Feature engineering (train)...")
    train_fe = create_features(df_train)

    print("Feature engineering (test)...")
    test_fe = create_features(df_test)

    X_train = train_fe.drop("Class", axis=1)
    y_train = train_fe["Class"]

    X_test = test_fe.drop("Class", axis=1)
    y_test = test_fe["Class"]

    #############################################################################
    # MODEL EĞİTİMİ
    #############################################################################

    print("Model eğitiliyor...")
    model = get_model(y_train)
    model.fit(X_train, y_train)

    #############################################################################
    # TAHMİN OLASILIKLARI
    #############################################################################

    y_proba = model.predict_proba(X_test)[:, 1]

    #############################################################################
    # COST-BASED THRESHOLD OPTIMIZATION
    #############################################################################

    FP_COST = 10      # ₺
    FN_COST = 1000    # ₺

    thresholds = np.arange(0.01, 0.99, 0.01)

    total_costs = []
    fp_list = []
    fn_list = []

    for t in thresholds:
        y_pred_temp = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()

        cost = fp * FP_COST + fn * FN_COST

        total_costs.append(cost)
        fp_list.append(fp)
        fn_list.append(fn)

    best_idx = np.argmin(total_costs)
    best_threshold = thresholds[best_idx]
    best_cost = total_costs[best_idx]

    print(f"\nEn düşük maliyetli threshold: {best_threshold:.2f}")
    print(f"Toplam maliyet: {best_cost:,.0f} ₺")

    #############################################################################
    # COST CURVE GRAFİĞİ
    #############################################################################

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, total_costs, label="Toplam Maliyet (₺)")
    plt.axvline(best_threshold, color="red", linestyle="--",
                label=f"Best Threshold = {best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Toplam Maliyet (₺)")
    plt.title("Cost-Based Threshold Optimization")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_curve.png"))
    plt.close()

    #############################################################################
    # FİNAL DEĞERLENDİRME
    #############################################################################

    y_pred_final = (y_proba >= best_threshold).astype(int)

    print("\n=== CLASSIFICATION REPORT (COST-BASED) ===")
    print(classification_report(y_test, y_pred_final))

    cm = confusion_matrix(y_test, y_pred_final)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (Threshold={best_threshold:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

    #############################################################################
    # MODEL + METADATA KAYDI
    #############################################################################

    model_package = {
        "model": model,
        "threshold": best_threshold,
        "features": X_train.columns.tolist(),
        "fp_cost": FP_COST,
        "fn_cost": FN_COST
    }

    joblib.dump(model_package, os.path.join(MODEL_DIR, "fraud_model.pkl"))

    print("\nModel ve çıktılar başarıyla kaydedildi.")

#################################################################################

if __name__ == "__main__":
    main()
