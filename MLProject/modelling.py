import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Parsing argumen dari CLI
    parser = argparse.ArgumentParser(description="Re-training Model Random Forest")
    parser.add_argument("--n-estimators", type=int, default=100, help="Jumlah estimators")
    parser.add_argument("--max-depth", type=int, default=10, help="Kedalaman maksimum tree")
    args = parser.parse_args()

    # Mengambil dataset bersih secara dinamis dari path di repositori (lokal) dengan fallback
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "dataset_clean.csv")
    
    if not os.path.exists(data_path):
        # Fallback 1: Folder Eksperimen_SML_Julianda-Putra-Mansur di dalam subfolder preprocessing
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "Eksperimen_SML_Julianda-Putra-Mansur", "preprocessing", "loan_approval_preprocessing", "dataset_clean.csv")
        
        # Fallback 2: Jika folder Eksperimen ada di luar folder workspace
        if not os.path.exists(data_path):
            parent_of_base = os.path.dirname(base_dir)
            data_path = os.path.join(parent_of_base, "Eksperimen_SML_Julianda-Putra-Mansur", "preprocessing", "loan_approval_preprocessing", "dataset_clean.csv")
            
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data bersih tidak ditemukan di {data_path}!")
        
    df = pd.read_csv(data_path)
    
    # Memisahkan Fitur dan Target
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tentukan apakah kita di bawah mlflow run
    is_mlflow_run = "MLFLOW_RUN_ID" in os.environ
    
    if not is_mlflow_run:
        # Inisialisasi DagsHub & set experiment & start run jika dijalankan manual (python modelling.py)
        try:
            import dagshub
            repo_owner, repo_name = "hikalputra12", "loan-prediction-ci-workflow"
            # Coba deteksi repo owner & name dari git remote URL secara dinamis
            try:
                import subprocess
                remote_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
                if remote_url.endswith(".git"):
                    remote_url = remote_url[:-4]
                parts = remote_url.split("/")
                if len(parts) >= 2:
                    repo_name = parts[-1]
                    repo_owner = parts[-2].split(":")[-1]
            except Exception:
                pass
                
            print(f"[INFO] Inisialisasi DagsHub dengan owner={repo_owner}, repo={repo_name}")
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            print("[INFO] Sukses terhubung ke DagsHub remote tracking.")
        except Exception as e:
            print(f"[WARN] Gagal inisialisasi DagsHub: {e}. Mencoba fallback ke repo Eksperimen_SML_Julianda-Putra-Mansur...")
            try:
                dagshub.init(repo_owner='hikalputra12', repo_name='Eksperimen_SML_Julianda-Putra-Mansur', mlflow=True)
                print("[INFO] Sukses terhubung ke DagsHub remote tracking (fallback).")
            except Exception as e_fallback:
                print(f"[WARN] Gagal inisialisasi DagsHub fallback: {e_fallback}. Menggunakan tracking default.")
            
        mlflow.set_experiment("Loan_Approval_CI")
        run_context = mlflow.start_run(run_name="RandomForest_CI_Run")
    else:
        # Jika di bawah mlflow run, gunakan active run yang disediakan oleh environment
        print("[INFO] Dijalankan di bawah MLflow Run secara langsung. Menggunakan environment tracking bawaan.")
        run_context = mlflow.start_run()
        
    with run_context:
        # Logging parameter secara manual
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("model_type", "RandomForest")
        
        # Melatih model
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi performa model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Logging metrik secara manual
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        
        # Logging model sebagai artefak resmi
        mlflow.sklearn.log_model(model, "ci_model")
        
        # Membuat dan mengunggah Artefak Tambahan 1: Confusion Matrix Plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix - CI')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)
            
        # Membuat dan mengunggah Artefak Tambahan 2: Feature Importance Plot
        import numpy as np
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = X.columns
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances - CI Model")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        plt.close()
        mlflow.log_artifact(fi_path)
        if os.path.exists(fi_path):
            os.remove(fi_path)
        
        print("="*50)
        print("[+] Re-training Model Berhasil Dilatih & Di-log!")
        print(f"Accuracy : {acc:.4f} | F1-Score: {f1:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()
