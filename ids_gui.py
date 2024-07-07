import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.feature_selection import RFE
import itertools
import tkinter as tk
from tkinter import ttk, messagebox

# Load data
def load_data():
    try:
        train = pd.read_csv('Train_data.csv')
        test = pd.read_csv('Test_data.csv')
        return train, test
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"File not found: {e}")
        return None, None

# Encode categorical features
def encode_features(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            label_encoder = LabelEncoder()
            df[col] = label_encoder.fit_transform(df[col])

# Feature selection
def select_features(X_train, Y_train):
    rfc = RandomForestClassifier()
    rfe = RFE(rfc, n_features_to_select=10)
    rfe = rfe.fit(X_train, Y_train)
    selected_features = [v for i, v in itertools.zip_longest(rfe.get_support(), X_train.columns) if i]
    return selected_features

# Calculate specificity
def calculate_specificity(y_true, y_pred, class_label):
    true_negative = np.sum((y_true != class_label) & (y_pred != class_label))
    false_positive = np.sum((y_true != class_label) & (y_pred == class_label))
    return true_negative / (true_negative + false_positive)

# Main IDS Application
class IDSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intrusion Detection System")
        self.root.geometry("800x600")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create buttons and labels
        ttk.Label(self.root, text="Intrusion Detection System", font=("Helvetica", 16)).pack(pady=20)
        
        self.start_knn_btn = ttk.Button(self.root, text="Run K-Nearest Neighbor", command=self.run_knn)
        self.start_knn_btn.pack(pady=10)
        
        self.start_svm_btn = ttk.Button(self.root, text="Run Support Vector Machine", command=self.run_svm)
        self.start_svm_btn.pack(pady=10)
        
        self.start_lr_btn = ttk.Button(self.root, text="Run Logistic Regression", command=self.run_lr)
        self.start_lr_btn.pack(pady=10)
        
        self.start_gb_btn = ttk.Button(self.root, text="Run Gradient Boosting", command=self.run_gb)
        self.start_gb_btn.pack(pady=10)
        
        self.start_rf_btn = ttk.Button(self.root, text="Run Random Forest", command=self.run_rf)
        self.start_rf_btn.pack(pady=10)
        
        self.result_text = tk.Text(self.root, height=10, width=80)
        self.result_text.pack(pady=20)

    def display_results(self, model_name, accuracy, error_rate, sensitivity, specificity, f1_score):
        self.result_text.insert(tk.END, f"{model_name} Results:\n")
        self.result_text.insert(tk.END, f"Accuracy: {accuracy}\n")
        self.result_text.insert(tk.END, f"Error Rate: {error_rate}\n")
        self.result_text.insert(tk.END, f"Sensitivity: {sensitivity}\n")
        self.result_text.insert(tk.END, f"Specificity: {specificity}\n")
        self.result_text.insert(tk.END, f"F1-score: {f1_score}\n")
        self.result_text.insert(tk.END, "-"*50 + "\n")
    
    # Define functions to run each model
    def run_knn(self):
        knn = KNeighborsClassifier(n_neighbors=25, metric='minkowski')
        knn.fit(x_train, y_train)
        knn_y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, knn_y_pred)
        error_rate = 1.0 - accuracy
        sensitivity = recall_score(y_test, knn_y_pred, average='weighted')
        specificity = calculate_specificity(y_test, knn_y_pred, 0)
        f1 = f1_score(y_test, knn_y_pred, average='weighted')
        self.display_results("K-Nearest Neighbor", accuracy, error_rate, sensitivity, specificity, f1)

    def run_svm(self):
        svc = SVC(kernel="linear", random_state=0)
        svc.fit(x_train, y_train)
        svc_y_pred = svc.predict(x_test)
        accuracy = accuracy_score(y_test, svc_y_pred)
        error_rate = 1.0 - accuracy
        sensitivity = recall_score(y_test, svc_y_pred, average='weighted')
        specificity = calculate_specificity(y_test, svc_y_pred, 0)
        f1 = f1_score(y_test, svc_y_pred, average='weighted')
        self.display_results("Support Vector Machine", accuracy, error_rate, sensitivity, specificity, f1)
    
    def run_lr(self):
        model = LogisticRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1.0 - accuracy
        sensitivity = recall_score(y_test, y_pred, average='weighted')
        specificity = calculate_specificity(y_test, y_pred, 0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.display_results("Logistic Regression", accuracy, error_rate, sensitivity, specificity, f1)
    
    def run_gb(self):
        model = GradientBoostingClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1.0 - accuracy
        sensitivity = recall_score(y_test, y_pred, average='weighted')
        specificity = calculate_specificity(y_test, y_pred, 0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        self.display_results("Gradient Boosting", accuracy, error_rate, sensitivity, specificity, f1)
    
    def run_rf(self):
        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(x_train, y_train)
        clf_y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, clf_y_pred)
        error_rate = 1.0 - accuracy
        sensitivity = recall_score(y_test, clf_y_pred, average='weighted')
        specificity = calculate_specificity(y_test, clf_y_pred, 0)
        f1 = f1_score(y_test, clf_y_pred, average='weighted')
        self.display_results("Random Forest", accuracy, error_rate, sensitivity, specificity, f1)

if __name__ == "__main__":
    # Load data
    train, test = load_data()
    if train is not None and test is not None:
        # Encode features
        encode_features(train)
        encode_features(test)
        
        # Drop irrelevant features
        train.drop(['num_outbound_cmds'], axis=1, inplace=True)
        test.drop(['num_outbound_cmds'], axis=1, inplace=True)
        
        # Prepare data
        X_train = train.drop(['class'], axis=1)
        Y_train = train['class']
        
        selected_features = select_features(X_train, Y_train)
        
        X_train = X_train[selected_features]
        X_test = test[selected_features]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.30, random_state=2)
        
        # Run the application
        root = tk.Tk()
        app = IDSApp(root)
        root.mainloop()
