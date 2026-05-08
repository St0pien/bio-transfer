import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from data_utils import scaffold_split_three_way # Import synchronizacji splitu
from classic_utils import load_classic_data

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_eval_classic(target="BACE1", mode="downstream"):
    logging.info(f"🚀 Start: {mode.upper()} | Target: {target}")
    
    data = load_classic_data(target, mode)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), labels = data

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42),
        "SVM": SVC(probability=True, kernel='rbf', random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=1000, 
            learning_rate=0.05, 
            n_jobs=-1, 
            tree_method='hist', # Szybsze dla dużych danych
            random_state=42,
            early_stopping_rounds=50
        )
    }

    results = []
    os.makedirs("models/classic", exist_ok=True)

    for name, model in models.items():
        logging.info(f"Trenuję {name}...")
        
        try:
            if mode == "upstream":
                if name == "SVM": 
                    logging.warning("SVM jest zbyt wolny dla Upstream Multitask. Pomijam.")
                    continue
                # MultiOutputClassifier trenuje osobny model dla każdego z 17 targetów
                clf = MultiOutputClassifier(model).fit(X_train, y_train)
            else:
                if name == "XGBoost":
                    clf = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    clf = model.fit(X_train, y_train)

            # --- Ewaluacja ---
            if mode == "downstream":
                probs = clf.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
                prc = average_precision_score(y_test, probs)
                
                logging.info(f"  [{name}] ROC-AUC: {auc:.4f} | PR-AUC: {prc:.4f}")
                results.append({"model": name, "target": target, "auc": auc, "prc": prc})
            
            # Zapisywanie
            joblib.dump(clf, f"models/classic/{name}_{target}_{mode}.pkl")
            
        except Exception as e:
            logging.error(f"Błąd podczas trenowania {name}: {e}")
    
    return results

if __name__ == "__main__":
    summary = []
    # Pętla po wszystkich celach Downstream
    # for target in ["BACE1", "TYK2", "A2a"]:
    #     res = train_and_eval_classic(target, mode="downstream")
    #     summary.extend(res)
    summary.extend(train_and_eval_classic("multi", mode="upstream"))
    
    # Wyświetlenie tabeli wyników
    print("\n" + "="*50)
    print("PODSUMOWANIE WYNIKÓW (DOWNSTREAM)")
    print("="*50)
    df_res = pd.DataFrame(summary)
    print(df_res.to_string(index=False))
    