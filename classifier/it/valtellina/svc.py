import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, balanced_accuracy_score
)
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns


class SupportVectorMachine:
    def __init__(self, X, y):
        self.__X = X
        self.__y = y
        self.__model = None
        self.__results = None
        self.__scaler = None

    def train(self):
        X_train, X_test, y_train, y_test = self.splitta_dataset()

        self.__model = SVC(kernel="rbf", C=1.0)
        self.__model.fit(X_train, y_train)

        y_pred = self.__model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        conf_matrix = confusion_matrix(y_test, y_pred)

        self.__results = {
            'modello': self.__model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'matrice_confusione': conf_matrix,
            'y_pred': y_pred,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }


        return self.__results

    def splitta_dataset(self, test_size=0.2, random_state=42):

        # One-Hot Encoding delle feature categoriche
        X_dumm = pd.get_dummies(
            self.__X,
            drop_first=True,
            dtype=int
        )

        # Encoding del target
        if isinstance(self.__y, pd.DataFrame):
            y = self.__y.squeeze()
        else:
            y = self.__y

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_dumm,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )

        self.__scaler = StandardScaler()

        X_train = self.__scaler.fit_transform(X_train)
        X_test = self.__scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def summary(self):
        if not self.__results:
            raise ValueError("Devi prima eseguire esegui_regressione()")

       res = {
            "accuracy": round(self.__results["accuracy"], 4),
            "precision": round(self.__results["precision"], 4),
            "recall": round(self.__results["recall"], 4),
            "f1_score": round(self.__results["f1_score"], 4),
            "balanced_accuracy": round(self.__results["balanced_accuracy"], 4),
            "matrice_confusione": self.__results["matrice_confusione"].tolist()
        }

        return res


    def grafico_matrice_confusione(self):
        if not self.__results:
            raise ValueError("Devi prima eseguire esegui_regressione()")

        cm = self.__results['matrice_confusione']
        classes = np.unique(
            np.concatenate([
                self.__results['y_test'],
                self.__results['y_pred']
            ])
        )

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


