# modelo_fraude_treino.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Carregar dataset
df = pd.read_csv("data/creditcard.csv")  # Certifica-te que tens este ficheiro

# Separar features e target
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir em treino/teste
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Treinar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Guardar modelo e scaler
joblib.dump(modelo, "modelo_fraude.pkl")
joblib.dump(scaler, "scaler.pkl")
