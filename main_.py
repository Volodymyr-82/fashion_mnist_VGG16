# ml_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

iris = load_iris()
X, y = iris.data, iris.target

# Розбиваємо на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Навчаємо модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Точність на тесті:", model.score(X_test, y_test))

# Зберігаємо модель у файл
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Класи квіток
class_names = ["Setosa", "Versicolor", "Virginica"]

# Створюємо застосунок
app = Dash(__name__)
app.title = "Iris Classifier"


# Макет
app.layout = html.Div([
    html.H1("🌸 Iris Flower Classifier", style={"textAlign": "center"}),

    html.Div([
        html.Label("Sepal length (см)"),
        dcc.Slider(4, 8, 0.1, value=5.0, id="sepal-length"),

        html.Label("Sepal width (см)"),
        dcc.Slider(2, 4.5, 0.1, value=3.0, id="sepal-width"),

        html.Label("Petal length (см)"),
        dcc.Slider(1, 7, 0.1, value=4.0, id="petal-length"),

        html.Label("Petal width (см)"),
        dcc.Slider(0.1, 2.5, 0.1, value=1.3, id="petal-width"),
    ], style={"width": "50%", "margin": "auto"}),

    html.Br(),
    html.H2(id="prediction-result", style={"textAlign": "center", "color": "darkblue"})
])

# Колбек: оновлює прогноз при зміні слайдерів
@app.callback(
    Output("prediction-result", "children"),
    Input("sepal-length", "value"),
    Input("sepal-width", "value"),
    Input("petal-length", "value"),
    Input("petal-width", "value"),
)
def update_prediction(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    return f"Прогноз: {class_names[prediction]} (ймовірність: {max(proba):.2f})"

if __name__ == "__main__":
    app.run(debug=True)