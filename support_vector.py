from classi import PuliziaDatasetFunghi
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns


class Support_vector_machine:
    def __init__(self, pulitore, test_size=0.2, random_state=42):
        self.pulitore = pulitore
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.results = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def prepara_dati(self, drop_first=True):
        df = self.pulitore.replace_strange_missing_values()
        X = df.dropna().copy()
        y = self.pulitore._PuliziaDatasetFunghi__y.loc[X.index].copy()

        X = pd.get_dummies(
            X,
            columns=X.select_dtypes(include=['object', 'string']).columns.tolist(),
            drop_first=drop_first,
            dtype=int
        )

        y_encoded = self.target_encoder.fit_transform(y.values.ravel())

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_encoded
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        print(f"Shape X_train: {X_train_scaled.shape}")
        print(f"Shape X_test: {X_test_scaled.shape}")
        print(f"Shape y_train: {y_train.shape}")
        print(f"Shape y_test: {y_test.shape}")
        print(f"Classi target: {self.target_encoder.classes_}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def addestra_modello(self, X_train, y_train, **params):
        self.model = SVC(**params)
        self.model.fit(X_train, y_train)
        return self.model

    def valuta_modello(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Devi prima addestrare il modello con addestra_modello().")

        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'matrice_confusione': conf_matrix,
            'y_pred': y_pred,
            'X_test': X_test,
            'y_test': y_test
        }

        print("\n" + "=" * 60)
        print("RISULTATI SUPPORT VECTOR MACHINE")
        print("=" * 60)
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"Precision:     {precision:.4f}")
        print(f"Recall:        {recall:.4f}")
        print(f"F1 Score:      {f1:.4f}")
        print(f"Balanced Acc:   {balanced_acc:.4f}")

        return self.results

    def stampa_classification_report(self):
        if not self.results:
            raise ValueError("Devi prima eseguire valuta_modello().")

        y_test = self.results['y_test']
        y_pred = self.results['y_pred']

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            target_names=self.target_encoder.classes_,
            zero_division=0
        )

        self.results['classification_report'] = report

        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)

        for classe in self.target_encoder.classes_:
            if classe in report:
                print(f"\n{classe}:")
                print(f"  Precision: {report[classe]['precision']:.4f}")
                print(f"  Recall:    {report[classe]['recall']:.4f}")
                print(f"  F1-score:  {report[classe]['f1-score']:.4f}")
                print(f"  Support:   {report[classe]['support']}")

        print(f"\nAccuracy media: {report['accuracy']:.4f}")

        return report

    def grafico_matrice_confusione(self):
        if 'matrice_confusione' not in self.results:
            raise ValueError("Devi prima eseguire valuta_modello().")

        cm = self.results['matrice_confusione']
        classes = self.target_encoder.classes_

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            ax=ax
        )
        ax.set_xlabel('Predetto')
        ax.set_ylabel('Reale')
        ax.set_title('Matrice di Confusione - SVM')
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def summary(self):
        if not self.results:
            raise ValueError("Devi prima eseguire valuta_modello().")

        print("\n" + "=" * 60)
        print("SUMMARY COMPLETO SVM")
        print("=" * 60)
        print(f"Accuracy:      {self.results['accuracy']:.4f}")
        print(f"Precision:     {self.results['precision']:.4f}")
        print(f"Recall:        {self.results['recall']:.4f}")
        print(f"F1 Score:      {self.results['f1_score']:.4f}")
        print(f"Balanced Acc:  {self.results['balanced_accuracy']:.4f}")
        print("\nMatrice di confusione:")
        print(self.results['matrice_confusione'])

