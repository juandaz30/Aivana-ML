import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from algorithms.DecisionTreeClassifier import DecisionTreeClassifier


class TestDecisionTree:
    def __init__(self):
        # Dataset sintético
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        # Tu árbol
        self.my_tree = DecisionTreeClassifier(
            criterion="gini", 
            max_depth=5, 
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            verbose=1   # para ver splits en consola
        )

        # Árbol de sklearn para comparar
        self.sklearn_tree = SklearnTree(
            criterion="gini", 
            max_depth=5, 
            random_state=42
        )

    def run(self):
        print("\n--- ENTRENANDO MI ÁRBOL ---")
        self.my_tree.fit(self.X_train, self.y_train)

        print("\n--- ENTRENANDO ÁRBOL SKLEARN ---")
        self.sklearn_tree.fit(self.X_train, self.y_train)

        print("\n================ RESULTADOS MI ÁRBOL ================")
        y_pred_my = self.my_tree.predict(self.X_test)
        y_proba_my = self.my_tree.predict_proba(self.X_test)

        print("Clases detectadas:", self.my_tree.classes_)
        print("Predicciones:", y_pred_my[:20])
        print("Probabilidades (primeras 5 filas):\n", y_proba_my[:5])
        print("Feature Importances:", self.my_tree.feature_importances_)
        print("Accuracy:", accuracy_score(self.y_test, y_pred_my))
        print("\nReporte de clasificación:\n", classification_report(self.y_test, y_pred_my))
        print("Matriz de confusión:\n", confusion_matrix(self.y_test, y_pred_my))

        print("\n================ RESULTADOS SKLEARN ================")
        y_pred_sk = self.sklearn_tree.predict(self.X_test)
        y_proba_sk = self.sklearn_tree.predict_proba(self.X_test)

        print("Clases detectadas:", self.sklearn_tree.classes_)
        print("Predicciones:", y_pred_sk[:20])
        print("Probabilidades (primeras 5 filas):\n", y_proba_sk[:5])
        print("Feature Importances:", self.sklearn_tree.feature_importances_)
        print("Accuracy:", accuracy_score(self.y_test, y_pred_sk))
        print("\nReporte de clasificación:\n", classification_report(self.y_test, y_pred_sk))
        print("Matriz de confusión:\n", confusion_matrix(self.y_test, y_pred_sk))


# ====== EJECUTAR PRUEBA ======
if __name__ == "__main__":
    tester = TestDecisionTree()
    tester.run()
