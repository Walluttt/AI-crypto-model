import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(X, y, sequence_length=60):
    X_seq = []
    y_seq = []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


df = pd.read_csv('coins/coin_Ethereum.csv')
print(df.head())

df.dropna(inplace=True)


X = df[['High','Low','Open','Volume','Marketcap']]
y = df['Close'].values

# Division chronologique : 70% premières données / 30% dernières
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalisation avec MinMaxScaler (0-1)
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Normalisation des targets avec MinMaxScaler
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()




print(f"Données d'entraînement: {len(X_train_scaled)} échantillons")
print(f"Données de test: {len(X_test_scaled)} échantillons")
print(f"Forme X_train normalisées: {X_train_scaled.shape}")
print(f"Forme y_train normalisées: {y_train_scaled.shape}")


# Créer séquences pour LSTM
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled)

print(f"Forme X_train_seq: {X_train_seq.shape}")  # (samples, timesteps, features)
print(f"Forme y_train_seq: {y_train_seq.shape}")

# ---------- Création du modèle LSTM ----------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prédiction du prix de clôture
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# ---------- Entraînement ----------
model.fit(X_train_seq, y_train_seq, epochs=20, batch_size=32)

# ---------- Prédictions ----------
predictions_scaled = model.predict(X_test_seq)
predictions = scaler_y.inverse_transform(predictions_scaled)

# ---------- Évaluation ----------
actual = y_test[60:]  # Alignement avec les prédictions
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
plt.plot(actual, label='Prix réel')
plt.plot(predictions, label='Prix prédit')
plt.title('Prédiction du prix de clôture')
plt.xlabel('Temps')
plt.ylabel('Prix (USD)')
plt.legend()
plt.show()
plt.savefig('prediction.png', dpi=300, bbox_inches='tight')