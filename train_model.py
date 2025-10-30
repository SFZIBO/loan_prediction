import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("loan_data_train.csv")
df = df.drop('Loan_ID', axis=1)

# Handle missing values
for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History']:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in ['LoanAmount', 'Loan_Amount_Term']:
    df[col] = df[col].fillna(df[col].median())

# Encode fitur kategorikal
label_encoders = {}
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Pisahkan fitur & target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Latih model dengan class_weight='balanced'
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← INI KUNCI UTAMA
    random_state=42
)
model.fit(X, y)

# Simpan
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("✅ Model dan encoder berhasil disimpan dengan class_weight='balanced'")