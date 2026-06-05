from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from io import BytesIO


class PuliziaDatasetFunghi:
    def __init__(self, dataset_id=73):
        self.__dataset = fetch_ucirepo(id=dataset_id)
        self.__X = self.__dataset.data.features
        self.__y = self.__dataset.data.targets
        self.encoders = {}

    def replace_strange_missing_values(self):
        strange_tokens = ["?", "Unknown", "unknown", "NAN", "nan", "NA", "na", "", " "]
        return self.__X.replace(strange_tokens, np.nan)

    def individua_na(self):
        df = self.replace_strange_missing_values()
        null_values = df.isna().sum()
        miss_percent = df.isna().sum() * 100 / len(df)
        return null_values, miss_percent

    def elimina_na(self):
        df = self.replace_strange_missing_values()
        X = df.dropna()
        y = self.__y.loc[X.index]
        return X, y

    def rimuovi_duplicati(self):
        """Rimuove righe duplicate dal dataset e allinea y"""
        df = self.replace_strange_missing_values()

        n_prima = len(df)
        df = df.drop_duplicates()
        n_dopo = len(df)

        y = self.__y.loc[df.index]

        print(f"Righe rimosse: {n_prima - n_dopo} ({(n_prima - n_dopo) / n_prima * 100:.2f}%)")
        print(f"Righe rimanenti: {n_dopo}")

        return df, y

    def crea_dummy(self, drop_first=True):
        """
        Per ogni variabile categorica crea le dummy (one-hot encoding).
        drop_first=True elimina la prima categoria per evitare la trappola delle dummy.
        """
        df = self.replace_strange_missing_values()

        cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        num_cols = df.select_dtypes(include='number').columns.tolist()

        print(f"Colonne categoriali da espandere: {len(cat_cols)}")
        print(f"Colonne numeriche mantenute: {len(num_cols)}")

        df_dummies = pd.get_dummies(
            df,
            columns=cat_cols,
            drop_first=drop_first,
            dtype=int
        )

        print(f"Shape originale: {df.shape}")
        print(f"Shape dopo dummy: {df_dummies.shape}")

        return df_dummies

    def codifica_categoricali(self, df=None):
        """Codifica le variabili categoriche in numeriche (LabelEncoder)"""
        if df is None:
            df = self.replace_strange_missing_values()

        df_encoded = df.copy()
        self.encoders = {}

        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'string':
                encoder = LabelEncoder()
                df_encoded[col] = df_encoded[col].fillna('MISSING')
                df_encoded[col] = encoder.fit_transform(df_encoded[col])
                self.encoders[col] = encoder

        return df_encoded

    def grafici_distribuzioni(self):
        """Grafici distribuzioni per dataset categorico (bar charts)"""
        df = self.replace_strange_missing_values()
        numeric_cols = df.select_dtypes(include='number').columns

        if len(numeric_cols) == 0:
            fig, axes = plt.subplots(3, 3, figsize=(15, 12))
            axes = axes.flatten()

            cols = df.columns[:9]
            for idx, col in enumerate(cols):
                df[col].value_counts().head(10).plot(kind='bar', ax=axes[idx])
                axes[idx].set_title(f'Distribuzione: {col}')
                axes[idx].set_xlabel('')
                axes[idx].tick_params(axis='x', rotation=45)

            plt.tight_layout()
        else:
            fig = df[numeric_cols].hist(figsize=(12, 10), bins=20)
            plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def statistica_descrittiva(self):
        """Statistica descrittiva per dataset categorico"""
        df = self.replace_strange_missing_values()

        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_summary': self.individua_na()
        }


