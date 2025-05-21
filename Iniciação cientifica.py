import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

dataset_path = 'MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'  # Substitua pelo nome correto do CSV

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")

df = pd.read_csv(dataset_path)


print("Colunas:", df.columns)

for col in ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']:
    if col in df.columns:
        df = df.drop(columns=col)

df = df.dropna()

if df['Label'].dtype == 'object':
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])

X = df.drop('Label', axis=1)
y = df['Label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42

)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
                                             