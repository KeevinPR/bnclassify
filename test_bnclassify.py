
import pandas as pd
import bnclassify
import sys

def test_bnclassify():
    print("Loading data...")
    try:
        df = pd.read_csv('/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/bnclassify/carwithnames.data')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    class_var = 'class'
    print(f"Data loaded. Shape: {df.shape}")

    # 1. Structure Learning
    print("\n--- Testing Structure Learning (TAN_HC) ---")
    try:
        model = bnclassify.tan_hc(class_var, df, k=2, epsilon=0)
        print("Structure learned successfully.")
    except Exception as e:
        print(f"Structure learning failed: {e}")
        return

    # 2. Parameter Learning
    print("\n--- Testing Parameter Learning (Bayes) ---")
    try:
        model = bnclassify.lp(model, df, smooth=0.5)
        print("Parameters learned successfully.")
    except Exception as e:
        print(f"Parameter learning failed: {e}")
        return

    # 3. Inspection
    print("\n--- Testing Inspection ---")
    try:
        np = bnclassify.nparams(model)
        na = bnclassify.narcs(model)
        cv = bnclassify.class_var(model)
        feats = bnclassify.features(model)
        ms = bnclassify.modelstring(model)
        print(f"nparams: {np}")
        print(f"narcs: {na}")
        print(f"class_var: {cv}")
        print(f"features: {feats[:3]}...")
        print(f"modelstring: {ms}")
        
        cpts = bnclassify.params(model)
        print(f"CPTs retrieved: {len(cpts)} tables")
        
        edges = bnclassify.edges(model)
        print(f"Edges extracted: {len(edges)}")
    except Exception as e:
        print(f"Inspection failed: {e}")
        # continue testing others

    # 4. Prediction
    print("\n--- Testing Prediction ---")
    try:
        # Predict classes
        preds_class = bnclassify.predict(model, df)
        print(f"Class predictions shape: {preds_class.shape}")
        
        # Predict probs
        preds_prob = bnclassify.predict(model, df, prob=True)
        print(f"Prob predictions shape: {preds_prob.shape}")
        print(preds_prob.head())
    except Exception as e:
        print(f"Prediction failed: {e}")

    # 5. Cross Validation & Scores
    print("\n--- Testing Evaluation ---")
    try:
        acc = bnclassify.cv(model, df, k=2, dag=True)
        print(f"CV Accuracy: {acc}")
        
        aic = bnclassify.aic(model, df)
        print(f"AIC: {aic}")
        
        bic = bnclassify.bic(model, df)
        print(f"BIC: {bic}")
        
        ll = bnclassify.log_lik(model, df)
        print(f"LogLik: {ll}")
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    test_bnclassify()
