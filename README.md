# MSc_Data-Analytics_Thesis_Intrusion_Detection_System_Cyber_Attacks
This project builds an IoT Intrusion Detection System using deep learning to classify attacks like DoS, DDoS, BruteForce, Scan, and Web-based. After merging packet and flow data, we clean, engineer features, select top ones via Random Forest, and balance classes with SMOTE. CNN, LSTM, and Conv-LSTM are compared, with Conv-LSTM chosen for deployment

🎯 Objectives
Build a robust IDS for IoT traffic using real-world datasets.
Merge packet and flow data for richer feature representation.
Apply feature selection and SMOTE balancing to improve accuracy.
Compare CNN, LSTM, and Conv-LSTM for multi-class classification.
Save the best model for deployment in a web application.

📂 Dataset
The dataset consists of packet-level and flow-level CSV files for multiple attack categories:
DoS-HTTP Flood
DDoS-ACK Fragmentation
Dictionary BruteForce
Vulnerability Scan
Web-based Uploading Attacks
Benign Traffic

Each record includes network-level features such as IP addresses, ports, protocols, packet sizes, and timestamps.

🛠️ Technologies Used
Languages: Python
Libraries & Frameworks:

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Plotly, Seaborn

Feature Engineering: scikit-learn, LabelEncoder

Balancing: imbalanced-learn (SMOTE)

Models: TensorFlow, Keras

Model Evaluation: scikit-learn metrics, confusion matrix, classification report

⚙️ Project Workflow
Data Loading & Merging – Merge packet and flow CSVs per attack type.

Data Cleaning – Remove duplicates, drop high-missing-value columns, fill NaNs with medians.

Feature Engineering – Extract temporal features (hour, day, weekend flag), encode categorical features, and select top features using Random Forest importance.

Class Balancing – Apply SMOTE to handle class imbalance.

Data Splitting & Scaling – Train-test split with StandardScaler for numeric features.

Model Training – Build and train CNN, LSTM, and Conv-LSTM with EarlyStopping.

Model Evaluation – Compare models using accuracy, precision, recall, F1-score, loss curves, and confusion matrices.

Model Saving – Save best Conv-LSTM model (conv_lstm_model.keras) and scaler (scaler.pkl).

📊 Results Summary
CNN Accuracy: ~98%

LSTM Accuracy: ~98%

Conv-LSTM Accuracy: ~99.5% (Selected for deployment)

Conv-LSTM showed best generalization, minimal loss, and strong multi-class performance.

🚀 How to Run
1️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
2️⃣ Prepare Data
Place packet-level and flow-level CSV files in data/ folder.

Ensure naming matches the merge keys (src_ip, dst_ip, src_port, dst_port).

3️⃣ Run the Notebook
bash
Copy
Edit
jupyter notebook Tushar_IoT_code.ipynb
or open in Google Colab.

4️⃣ Output Files
conv_lstm_model.keras → Best-trained model.

scaler.pkl → StandardScaler for new data.

X_test_for_prediction.csv → Processed test features for prediction demo.

📦 Deployment
The saved Conv-LSTM model and scaler are used in a simple Flask-based web app for real-time intrusion detection.

📌 Key Features of the Code
Modular functions for data loading, preprocessing, feature engineering, and model evaluation.

Automatic feature selection (95% cumulative importance).

Balanced dataset creation using SMOTE.

Detailed performance visualization (metrics, confusion matrices, learning curves).

🧪 Future Improvements
Incorporate additional IoT-specific datasets for broader generalization.

Implement online learning for real-time adaptation on WebApp.
