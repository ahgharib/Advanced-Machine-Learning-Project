import tkinter as tk
from tkinter import filedialog, Label, Button
from tkinter import *
import pandas as pd
import numpy as np
from joblib import load
import joblib


# Paths
scaler_path = "D:/Advanced Machine Learining Project/Classification Loan Aproval/Artifitial Neural Network Classification/scaler.pkl"
model_path = "D:/Advanced Machine Learining Project/Classification Loan Aproval/Artifitial Neural Network Classification/ANN.joblib"

# For error Handling
feature_numbers = 11
feature_names = {"no_of_dependents", "education", "self_employed", "income_annum", "loan_amount", "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"}

# Load Saved Models
loaded_model = load(model_path)
scaler = joblib.load(scaler_path)
global file_path
file_path = ""

# For Scroll bar
class ScrollableFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)

        # Create a custom scrollbar with background color
        self.scrollbar = tk.Scrollbar(self, orient="vertical", bg="pink", troughcolor="pink")

        # Create a custom canvas with background color
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg="pink", yscrollcommand=self.scrollbar.set)

        self.scrollbar.config(command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg="pink")  # Set background color for the scrollable frame

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Pack the custom scrollbar and canvas
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


# Preprocessing the User Inputs
def preprocess_data(scaler, csv_file_path):
    df = pd.read_csv(csv_file_path)
    df.drop('loan_id', axis=1, inplace=True)
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["self_employed"] = df["self_employed"].apply(lambda x: 1 if x == "Yes" else 0)
    df["education"] = df["education"].apply(lambda x: 1 if x == "Graduate" else 0)
    df["loan_status"] = df["loan_status"].apply(lambda x: 1 if x == "Approved" else 0)
    X = df.drop(['loan_status'], axis=1)
    X_scaled = scaler.transform(X)
    
    # Convert X_scaled back to a DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df


# Prediction Function
def get_predictions(model, df):
    predictions = model.predict(df)
    return predictions




# Start of GUI Fucntion
def open_ann_window():
    global result_labels

    # For Taking the CSV File from the User
    def open_csv_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)

            # Preprocess the data
            df = preprocess_data(scaler, file_path)

            # Check if the number of columns matches the expected number of features
            if len(df.columns) != feature_numbers:
                error_message = f"Error: The CSV file must have {feature_numbers} columns."
                show_error_message(error_message)
                return

            # Check if the column names match the expected feature names
            if set(df.columns) != set(feature_names):
                missing_features = set(feature_names) - set(df.columns)
                extra_features = set(df.columns) - set(feature_names)
                error_message = f"Error: The CSV file must have the following features: {feature_names}."
                if missing_features:
                    error_message += f"\nMissing features: {missing_features}."
                if extra_features:
                    error_message += f"\nExtra features: {extra_features}."
                show_error_message(error_message)
                return

            num_rows = len(df)

            # Create result labels for each row For Diplaying
            for i in range(num_rows):
                result_label = Label(scrollable_frame.scrollable_frame, font=("Arial", 20, "bold"), width=35, height=1, bg="lightblue", foreground="purple")
                result_label.grid(row=i, column=0, pady=5, sticky="nsew", padx=(675, 675))
                result_labels.append(result_label)

            # Using Prediction Function
            predictions = get_predictions(loaded_model, df)
            for row_number, prediction in enumerate(predictions):
                pred = np.argmax(prediction)  # Get the index of the class with the highest probability
                predicted_status = "APPROVED" if pred == 0 else "REJECTED"  # Map the index to the class label
                if row_number < len(result_labels):
                    result_labels[row_number].config(text="Row {}: {}".format(row_number+1, predicted_status))





    ANN_window = tk.Toplevel()
    ANN_window.title("ANN Model For Loan Aproval")
    ANN_window.geometry("800x800") 
    ANN_window.minsize(400,400) 
    ANN_window.iconbitmap("/logo.ico")
    ANN_window.configure(bg="pink")

    my_button1 = Button(ANN_window, text="Open CSV", font=("Arial", 15), width=20, bg="red", command=open_csv_file, borderwidth=3, activebackground="blue")
    my_button1.pack(pady=10)  # Use pack instead of grid

    # Initialize result labels list
    result_labels = []

    # Create scrollable frame
    scrollable_frame = ScrollableFrame(ANN_window)
    scrollable_frame.pack(fill="both", expand=True)


def show_error_message(message):
    error_window = tk.Toplevel()
    error_window.title("Error")
    error_window.geometry("400x200")
    error_window.configure(bg="pink")
    
    error_label = Label(error_window, text=message, font=("Arial", 12), wraplength=380, justify="center", bg="pink")
    error_label.pack(expand=True, fill="both", padx=20, pady=20)
