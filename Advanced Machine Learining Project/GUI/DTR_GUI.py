import tkinter as tk
from tkinter import filedialog, Label, Button
from tkinter import *
import pandas as pd
from joblib import load
import joblib

model_path = "D:/Advanced Machine Learining Project/Regression Car Price Prediction/Descion Tree Regression/random_search_dt.joblib"
pipeline_path = "D:/Advanced Machine Learining Project/Regression Car Price Prediction/Descion Tree Regression/preprocess_pipeline.joblib"

feature_numbers = 10
feature_names = {"model", "year", "price", "transmission", "mileage", "fuelType", "tax", "mpg", "engineSize", "Manufacturer"}


loaded_model = load(model_path)
global file_path
file_path = ""

# Load the Pipeline For preprocessing
preprocess_pipeline_loaded = joblib.load(pipeline_path)


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


# for Loading the File, Preprocessing, Predicting
def predict_car_price_from_df_file(file_path, preprocess_pipeline_loaded, loaded_model):
    # Load the DataFrame from the file
    user_data = pd.read_csv(file_path)

    # Apply preprocessing pipeline to user data
    preprocessed_data = preprocess_pipeline_loaded.transform(user_data)

    # Predict car prices using the preprocessed data
    predicted_prices = loaded_model.predict(preprocessed_data)

    return predicted_prices




def get_predictions(model, df):
    predictions = model.predict(df)
    return predictions





def open_dtr_window():
    global result_labels

    def open_csv_file():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = pd.read_csv(file_path)

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

            # Create result labels for each row
            for i in range(num_rows):
                result_label = Label(scrollable_frame.scrollable_frame, font=("Arial", 20, "bold"), width=35, height=1, bg="lightblue", foreground="purple")
                result_label.grid(row=i, column=0, pady=5, sticky="nsew", padx=(675, 675))
                result_labels.append(result_label)
            
            # Predicting
            predictions = predict_car_price_from_df_file(file_path, preprocess_pipeline_loaded, loaded_model)
            for row_number, prediction in enumerate(predictions):
                # Ensure the index is within the range of result_labels
                if row_number < len(result_labels):
                    result_labels[row_number].config(text="Row {}: {}".format(row_number+1, round(prediction, 3)))





    DTR_window = tk.Toplevel()
    DTR_window.title("DTR Model For Car Price Prediction")
    DTR_window.geometry("800x800") 
    DTR_window.minsize(400,400) 
    DTR_window.iconbitmap("/logo.ico")
    DTR_window.configure(bg="pink")

    my_button1 = Button(DTR_window, text="Open CSV", font=("Arial", 15), width=20, bg="red", command=open_csv_file, borderwidth=3, activebackground="blue")
    my_button1.pack(pady=10)  # Use pack instead of grid

    # Initialize result labels list
    result_labels = []

    # Create scrollable frame
    scrollable_frame = ScrollableFrame(DTR_window)
    scrollable_frame.pack(fill="both", expand=True)


def show_error_message(message):
    error_window = tk.Toplevel()
    error_window.title("Error")
    error_window.geometry("400x200")
    error_window.configure(bg="pink")
    
    error_label = Label(error_window, text=message, font=("Arial", 12), wraplength=380, justify="center", bg="pink")
    error_label.pack(expand=True, fill="both", padx=20, pady=20)
