# 🍄 Mushroom Classification API (Flask + ML)

API REST per analisi dati ed applicazione di modelli di Machine Learning sul **Mushroom Dataset (UCI)**.
Il progetto include EDA, preprocessing avanzato e due modelli ML:

* 📉 Regressione Logistica
* 📈 Support Vector Machine (SVM)

---

## 🚀 Features

* Pulizia automatica del dataset (missing values + outliers handling)
* Imputazione intelligente (media / mediana / moda)
* Encoding variabili categoriche
* Train/test split stratificato
* Modelli ML pronti all’uso
* Metriche complete (accuracy, precision, recall, F1, AUC)
* Visualizzazioni (matrici di confusione, coefficienti, distribuzioni)
* API REST con Flask
* Containerizzazione Docker

---

## 📁 Struttura progetto

```
it/
 └── valtellina/
      ├── eda.py
      ├── logistic_regression.py
      ├── svc.py

main.py
requirements.txt
Dockerfile
```

---

## 📊 Dataset

* 📦 UCI Mushroom Dataset
* ID: `73` (ucimlrepo)

---

## ⚙️ Installazione locale

### 1. Clona il repository

```bash
git clone https://github.com/tuo-username/mushroom-api.git
cd mushroom-api
```

---

### 2. Installa dipendenze

```bash
pip install -r requirements.txt
```

---

### 3. Avvia il server

```bash
python main.py
```

API disponibile su:

```
http://localhost:5000
```

---

## 🐳 Docker

### 📦 Build immagine

```bash
docker build -t mushroom-api .
```

### ▶️ Run container

```bash
docker run -p 5000:5000 mushroom-api
```

---

## 📡 API Endpoints

### 🔎 Analisi dati

#### Missing values

```
GET /na
```

---

#### Statistiche dataset

```
GET /stats
```

---

#### Distribuzioni variabili

```
GET /grafici
```

---

## 🤖 Machine Learning

---

### 📉 Regressione Logistica

#### Coefficienti modello

```
GET /logreg/coeff
```

#### Matrice di confusione

```
GET /logreg/confusionmatrix
```

#### Summary metriche

```
GET /logreg/summary
```

---

### 📈 Support Vector Machine

#### Matrice di confusione

```
GET /svc/confusionmatrix
```

#### Summary metriche

```
GET /svc/summary
```

---

## 📊 Output esempio

### Metriche modello

```json
{
  "accuracy": 0.98,
  "precision": 0.98,
  "recall": 0.98,
  "f1_score": 0.98,
  "balanced_accuracy": 0.98
}
```

---

## 🧠 Pipeline ML

1. Caricamento dataset UCI
2. Pulizia valori anomali (`?, NA, unknown, ecc.`)
3. Eliminazione colonne con troppi missing values
4. Imputazione intelligente:

   * Moda → categoriche
   * Media/Mediana → numeriche (Jarque-Bera test)
5. Encoding:

   * One-hot encoding (feature)
   * Label encoding (target)
6. Train/test split stratificato
7. Training modelli ML
8. Valutazione + visualizzazioni

---

## 📦 Tecnologie utilizzate

* Python 3.11
* Flask
* Pandas / NumPy
* Scikit-learn
* Matplotlib / Seaborn
* SciPy
* ucimlrepo
* Docker

---

## 🐳 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /api_classificazione

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "main.py"]
```

