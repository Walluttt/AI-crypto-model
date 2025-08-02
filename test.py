import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Attention, Dense, Flatten, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import yfinance as yf
import matplotlib.pyplot as plt

# Téléchargement des données Bitcoin
btc = yf.download('BTC-USD', start='2015-01-01', interval='1d')
print(btc.head())

# Préparation des données - utilisation des données OHLCV de yfinance
df = btc.reset_index()
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
features = ['open', 'high', 'low', 'close', 'volume']

# Normalisation des features
scaler = StandardScaler()
data = scaler.fit_transform(df[features])

# Normalisation des targets aussi
scaler_y = StandardScaler()

# Création de séquences sliding windows (lookback = 30 jours)
X, y = [], []
for i in range(30, len(data)-1):  # Réduire la prédiction à +1 jour au lieu de +15
    X.append(data[i-30:i])
    y.append(data[i+1][3])  # prévoir close le lendemain
X, y = np.array(X), np.array(y)

# Normaliser y
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Division chronologique train/test (70% premières années / 30% dernières)
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Données d'entraînement: {len(X_train)} échantillons")
print(f"Données de test: {len(X_test)} échantillons")
print(f"Forme X_train: {X_train.shape}")
print(f"Forme y_train: {y_train.shape}")

# Modèle simplifié avec régularisation pour éviter l'overfitting
input_layer = tf.keras.Input(shape=(30, X.shape[2]))

# LSTM bidirectionnel simple avec dropout
x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(input_layer)
x = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2))(x)

# Couches denses avec régularisation
x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
output = Dense(1)(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping pour éviter l'overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print("\n=== DÉBUT DE L'ENTRAÎNEMENT ===")
# Entraînement avec les bonnes données
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.15,
    callbacks=[early_stopping],
    verbose=1
)

print("\n=== ÉVALUATION DU MODÈLE ===")

# Prédictions sur le vrai test set (pas de redivision !)
y_pred = model.predict(X_test)

# Dénormaliser les prédictions et les vraies valeurs
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculer les métriques
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_real, y_pred_real)

print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE: {mae:.6f}")

# Convertir en prix réels
print(f"\n=== PERFORMANCE EN DOLLARS ===")
print(f"Prix réel moyen: ${np.mean(y_test_real):,.2f}")
print(f"Prix prédit moyen: ${np.mean(y_pred_real):,.2f}")
print(f"Erreur absolue moyenne: ${mae:,.2f}")

# Prédiction du prochain jour
last_sequence = data[-30:].reshape(1, 30, X.shape[2])
next_day_pred_scaled = model.predict(last_sequence)
next_day_pred = scaler_y.inverse_transform(next_day_pred_scaled.reshape(-1, 1))[0, 0]

current_price = df['close'].iloc[-1]
print(f"\n=== PRÉDICTION FUTURE ===")
print(f"Prix actuel du Bitcoin: ${current_price:,.2f}")
print(f"Prix prédit demain: ${next_day_pred:,.2f}")
print(f"Variation prédite: {((next_day_pred/current_price-1)*100):+.2f}%")

# Sauvegarde du modèle
model.save('bitcoin_prediction_model.h5')
print(f"\nModèle sauvegardé: bitcoin_prediction_model.h5")

# Visualisations
plt.figure(figsize=(15, 10))

# 1. Evolution de la loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Évolution de la Loss pendant l\'entraînement')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 2. Prédictions vs Réalité
plt.subplot(2, 2, 2)
plt.scatter(y_test_real, y_pred_real, alpha=0.6)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
plt.xlabel('Prix Réels ($)')
plt.ylabel('Prix Prédits ($)')
plt.title('Prédictions vs Réalité')

# 3. Comparaison temporelle sur les dernières prédictions
plt.subplot(2, 2, 3)
last_50 = min(50, len(y_test_real))
plt.plot(range(last_50), y_test_real[-last_50:], 'o-', label='Prix Réels', markersize=4)
plt.plot(range(last_50), y_pred_real[-last_50:], 's-', label='Prix Prédits', markersize=4)
plt.title(f'Comparaison des {last_50} dernières prédictions')
plt.xlabel('Échantillons')
plt.ylabel('Prix ($)')
plt.legend()

# 4. Distribution des erreurs
plt.subplot(2, 2, 4)
errors = y_pred_real - y_test_real
plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.title('Distribution des erreurs de prédiction')
plt.xlabel('Erreur de prédiction ($)')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.savefig('bitcoin_prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Graphiques sauvegardés: bitcoin_prediction_results.png")