from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from scipy.stats import jarque_bera


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

    def elimina_na(self, test_size=0.2, random_state=42):
        df = self.replace_strange_missing_values()

        # 1. elimina colonne con troppi NA
        soglia = 50.0
        cols_da_tenere = df.columns[df.isna().sum()*100/len(df) <= soglia]
        df = df[cols_da_tenere]

        # 2. elimina duplicati
        df = df.drop_duplicates()

        # 3. split PRIMA dell'imputazione
        X_train, X_test, y_train, y_test = train_test_split(
            df,
            self.__y.loc[df.index],
            test_size=test_size,
            random_state=random_state,
            stratify=self.__y.loc[df.index]
        )

        # 4. dizionari per imputazione
        fill_values = {}

        for col in X_train.columns:

            # categorica → moda
            if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'string':
                fill_values[col] = X_train[col].mode()[0]

            elif X_train[col].dtype == 'number':
                # test normalità (JB su campione)
                p_value = jarque_bera(X_train[col].dropna())[1]

                # distribuzione normale → media
                if p_value > 0.05:
                    fill_values[col] = X_train[col].mean()
                else:
                    # non normale → mediana
                    fill_values[col] = X_train[col].median()

        # 5. imputazione su train e test usando SOLO stats del train
        X_train = X_train.fillna(fill_values)
        X_test = X_test.fillna(fill_values)

        # 6. elimina colonne costanti
        const_cols = X_train.columns[
            X_train.nunique(dropna=False) > 1
            ]
        X_train = X_train[const_cols]
        X_test = X_test[const_cols]

        # 7. riallinea y
        y_train = y_train.loc[X_train.index]
        y_test = y_test.loc[X_test.index]

        # 8. unifica nuovamente X e y
        X = pd.concat([X_train, X_test], axis=0)
        y = pd.concat([y_train, y_test], axis=0)

        # riordino per sicurezza
        X = X.sort_index()
        y = y.sort_index()

        return X, y


    def grafici_distribuzioni(self):
        """Grafici distribuzioni per tutte le variabili categoriche"""
        df = self.replace_strange_missing_values()

        cat_cols = df.select_dtypes(include=['object', 'string']).columns
        numeric_cols = df.select_dtypes(include='number').columns

        if len(cat_cols) > 0:

            n_cols = 3
            n_rows = int(np.ceil(len(cat_cols) / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(6 * n_cols, 4 * n_rows)
            )

            axes = np.array(axes).flatten()

            for idx, col in enumerate(cat_cols):
                df[col].value_counts(dropna=False).plot(
                    kind='bar',
                    ax=axes[idx]
                )

                axes[idx].set_title(f'Distribuzione: {col}')
                axes[idx].set_xlabel('')
                axes[idx].tick_params(axis='x', rotation=45)

            # elimina assi inutilizzati
            for idx in range(len(cat_cols), len(axes)):
                fig.delaxes(axes[idx])

            plt.tight_layout()

        else:
            fig = df[numeric_cols].hist(
                figsize=(12, 10),
                bins=20
            )
            plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def statistica_descrittiva(self):
        df_old = self.replace_strange_missing_values()
        X, y = self.elimina_na()
        df = X

        null_values, miss_percent = self.individua_na()

        return {
            "shape": list(df_old.shape),
            "columns": df_old.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
            "missing_summary": {
                "null_values": null_values.to_dict(),
                "missing_percent": miss_percent.to_dict()
            },
            "shape after handling missing values": list(df.shape),
            "columns": df.columns.tolist(),
        }

