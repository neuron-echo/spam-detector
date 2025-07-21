import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load and preprocess data
data = pd.read_csv('mail_data.csv')
data = data.where(pd.notnull(data), '')
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

x = data['Message']
y = data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=3)

# Feature extraction
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Model training
model = LogisticRegression()
model.fit(x_train, y_train)

# Accuracy
print("Training Accuracy:", accuracy_score(y_train, model.predict(x_train)))
print("Test Accuracy:", accuracy_score(y_test, model.predict(x_test)))

# GUI
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("500x300")
root.config(bg="#f0f0f0")

def classify_message():
    user_input = text_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Warning", "Please enter a message.")
        return
    input_data = vectorizer.transform([user_input])
    prediction = model.predict(input_data)[0]
    result = "NOT SPAM" if prediction == 1 else "SPAM"
    result_label.config(text=f"Result: {result}", fg="green" if prediction == 1 else "red")

# Widgets
header = tk.Label(root, text="Email/SMS Spam Classifier", font=("Arial", 16, "bold"), bg="#f0f0f0")
header.pack(pady=10)

text_entry = tk.Text(root, height=6, width=50, font=("Arial", 12))
text_entry.pack(pady=10)

predict_button = tk.Button(root, text="Classify", font=("Arial", 12), command=classify_message, bg="#4CAF50", fg="white")
predict_button.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=10)

root.mainloop()
