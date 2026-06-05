from classi import PuliziaDatasetFunghi
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns


class RegressioneLogistica:
    def __init__(self, pulitore, test_size=0.2, random_state=42):
        self.pulitore = pulitore
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.results = {}
        self.target_encoder = LabelEncoder()

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

        print(f"Shape X_train: {X_train.shape}")
        print(f"Shape X_test: {X_test.shape}")
        print(f"Shape y_train: {y_train.shape}")
        print(f"Shape y_test: {y_test.shape}")
        print(f"Classi target: {self.target_encoder.classes_}")

        return X_train, X_test, y_train, y_test

    def esegui_regressione(self, X_train, X_test, y_train, y_test, **params):
        self.model = LogisticRegression(**params)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            auc_roc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_roc = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')

        conf_matrix = confusion_matrix(y_test, y_pred)

        coefficients = pd.DataFrame({
            'variabile': X_train.columns,
            'coefficiente': self.model.coef_[0],
            'odds_ratio': np.exp(self.model.coef_[0])
        }).sort_values('coefficiente', ascending=False)

        intercept = self.model.intercept_[0]

        class_report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            target_names=self.target_encoder.classes_,
            zero_division=0
        )

        self.results = {
            'modello': self.model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'auc_roc': auc_roc,
            'matrice_confusione': conf_matrix,
            'coefficienti': coefficients,
            'intercept': intercept,
            'classification_report': class_report,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        print("\n" + "=" * 60)
        print("RISULTATI REGRESSIONE LOGISTICA")
        print("=" * 60)
        print(f"Accuracy:       {accuracy:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1 Score:       {f1:.4f}")
        print(f"Balanced Acc:   {balanced_acc:.4f}")
        print(f"AUC-ROC:        {auc_roc:.4f}")
        print(f"Intercept:      {intercept:.4f}")

        return self.results

    def stampa_classification_report(self):
        if 'classification_report' not in self.results:
            raise ValueError("Devi prima eseguire esegui_regressione()")

        report = self.results['classification_report']
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

    def summary(self):
        if not self.results:
            raise ValueError("Devi prima eseguire esegui_regressione()")

        print("\n" + "=" * 60)
        print("SUMMARY COMPLETO")
        print("=" * 60)
        print(f"Accuracy:      {self.results['accuracy']:.4f}")
        print(f"Precision:     {self.results['precision']:.4f}")
        print(f"Recall:        {self.results['recall']:.4f}")
        print(f"F1 Score:      {self.results['f1_score']:.4f}")
        print(f"Balanced Acc:  {self.results['balanced_accuracy']:.4f}")
        print(f"AUC-ROC:       {self.results['auc_roc']:.4f}")
        print(f"Intercept:     {self.results['intercept']:.4f}")
        print("\nMatrice di confusione:")
        print(self.results['matrice_confusione'])
        print("\nCoefficienti principali:")
        print(self.results['coefficienti'].head(10))

    def grafico_matrice_confusione(self):
        if 'matrice_confusione' not in self.results:
            raise ValueError("Devi prima eseguire esegui_regressione()")

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
        ax.set_title('Matrice di Confusione')
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    def grafico_coefficienti(self):
        if 'coefficienti' not in self.results:
            raise ValueError("Devi prima eseguire esegui_regressione()")

        coeffs = self.results['coefficienti'].head(15)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].barh(coeffs['variabile'], coeffs['coefficiente'], color='steelblue')
        axes[0].set_xlabel('Coefficiente')
        axes[0].set_title('Coefficienti')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)

        axes[1].barh(coeffs['variabile'], coeffs['odds_ratio'], color='coral')
        axes[1].set_xlabel('Odds Ratio')
        axes[1].set_title('Odds Ratio')
        axes[1].axvline(x=1, color='red', linestyle='--', linewidth=1)

        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

