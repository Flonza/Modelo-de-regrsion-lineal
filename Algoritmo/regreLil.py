# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Cargue de datos y anlisis exploratorio
data = pd.read_csv("../Titanic-Dataset.csv")
print(data.info())
print(data.describe())

# Visualizacion
sns.countplot(data=data, x="Survived", hue="Sex")
plt.title("Supervivencia por Género")
plt.show()

# Filtrar columnas antes de calcular
numerical_data = data.select_dtypes(include=["number"])
sns.heatmap(numerical_data.corr(), annot=True, cmap="coolwarm")
plt.title("Mapa de Correlación")
plt.show()


# Preprocesar los datos
# Lleno valores faltantes
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.drop(columns=['Cabin'], inplace=True)

# Codificación de variables
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Seleccionar caracteristicas
X = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y = data['Survived']

selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y)
scores = pd.DataFrame({"Feature": X.columns, "Score": selector.scores_})
print(scores.sort_values(by="Score", ascending=False))

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluar 
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.show()

# Gráficas adicionales
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_[0]})
coefficients.sort_values(by="Coefficient", ascending=True, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients, x="Coefficient", y="Feature", palette="viridis")
plt.title("Importancia de las Características")
plt.show()

# Resultados
print("Los resultados muestran que las características más relevantes incluyen:")
print(coefficients)
