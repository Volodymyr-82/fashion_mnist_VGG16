import os
import base64
import io
from pathlib import Path
from typing import Optional, Tuple, List, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import plotly.express as px

# =============================
# TensorFlow Import with Graceful Fallback
# =============================
TF_AVAILABLE = False
IMPORT_ERROR_MSG = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.datasets import fashion_mnist
    from tensorflow.keras.layers import Conv2D

    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} успішно завантажено")
except ImportError as e:
    IMPORT_ERROR_MSG = f"TensorFlow не встановлено: {str(e)}"
    print(IMPORT_ERROR_MSG)
except Exception as e:
    IMPORT_ERROR_MSG = f"Помилка завантаження TensorFlow: {str(e)}"
    print(IMPORT_ERROR_MSG)

# =============================
# Configuration and Constants
# =============================
MODEL_PATH = r"C:/Users/Vova/Downloads/fashion_vgg.keras"
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Default input shape for Fashion-MNIST
DEFAULT_INPUT_SHAPE = (None, 224, 224, 3)

# =============================
# Data and Model Loading
# =============================
model = None
x_train = y_train = x_test = y_test = None
INPUT_SHAPE = DEFAULT_INPUT_SHAPE


def load_model_and_data():
    """Load the VGG model and Fashion-MNIST dataset with error handling."""
    global model, x_train, y_train, x_test, y_test, INPUT_SHAPE, IMPORT_ERROR_MSG

    if not TF_AVAILABLE:
        return False, "TensorFlow недоступний"

    # Load model
    try:
        if Path(MODEL_PATH).exists():
            model = keras.models.load_model(MODEL_PATH)
            INPUT_SHAPE = model.input_shape
            print(f"Модель завантажено: {INPUT_SHAPE}")
        else:
            return False, f"Файл моделі не знайдено: {MODEL_PATH}"
    except Exception as e:
        return False, f"Помилка завантаження моделі: {str(e)}"

    # Load Fashion-MNIST dataset
    try:
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        print(f"Dataset завантажено: train={x_train.shape}, test={x_test.shape}")
        return True, "Модель та дані успішно завантажено"
    except Exception as e:
        return False, f"Помилка завантаження датасету: {str(e)}"


# Initialize data and model
DATA_LOADED, LOAD_MESSAGE = load_model_and_data()
if not DATA_LOADED:
    IMPORT_ERROR_MSG = LOAD_MESSAGE
    x_test = np.empty((0, 28, 28))
    y_test = np.empty((0,))

_, H, W, C = INPUT_SHAPE


# =============================
# Image Processing Functions
# =============================
class ImagePreprocessor:
    """Handles all image preprocessing operations."""

    @staticmethod
    def pil_to_array(img: Image.Image, target_channels: int) -> np.ndarray:
        """Convert PIL image to normalized numpy array."""
        arr = np.array(img, dtype=np.float32) / 255.0

        if target_channels == 1 and len(arr.shape) == 2:
            arr = np.expand_dims(arr, axis=-1)
        elif target_channels == 3 and len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif target_channels == 1 and len(arr.shape) == 3:
            arr = np.mean(arr, axis=-1, keepdims=True)

        return arr

    @staticmethod
    def center_crop(img: Image.Image) -> Image.Image:
        """Apply center crop to make image square."""
        w, h = img.size
        size = min(w, h)
        left = (w - size) // 2
        top = (h - size) // 2
        return img.crop((left, top, left + size, top + size))

    @staticmethod
    def fmnist_style_transform(img: Image.Image) -> Image.Image:
        """Apply Fashion-MNIST style transformation."""
        # Convert to grayscale and resize to 28x28
        img = img.convert('L').resize((28, 28), Image.Resampling.BILINEAR)

        # Apply slight blur to simulate Fashion-MNIST characteristics
        try:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        except:
            pass  # Skip if ImageFilter not available

        return img

    @classmethod
    def preprocess_uploaded_image(cls, img: Image.Image, options: List[str]) -> np.ndarray:
        """Preprocess uploaded image for model prediction."""
        # Apply preprocessing options
        if 'center_crop' in options:
            img = cls.center_crop(img)

        if 'fmnist_style' in options:
            img = cls.fmnist_style_transform(img)

        # Convert to correct format
        if C == 1:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        # Resize to model input size
        img = img.resize((W, H), Image.Resampling.BILINEAR)

        # Convert to array and add batch dimension
        arr = cls.pil_to_array(img, C)
        return np.expand_dims(arr, axis=0)

    @classmethod
    def preprocess_fmnist_single(cls, img_array: np.ndarray) -> np.ndarray:
        """Preprocess single Fashion-MNIST image (28x28) for model."""
        # Convert to PIL and back to handle format consistently
        img_uint8 = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        return cls.preprocess_uploaded_image(pil_img, [])

    @classmethod
    def preprocess_fmnist_batch(cls, batch: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Preprocess batch of Fashion-MNIST images efficiently."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow потрібен для батчової обробки")

        # Add channel dimension
        if len(batch.shape) == 3:
            batch = np.expand_dims(batch, axis=-1)

        # Convert to target channels
        if C == 3 and batch.shape[-1] == 1:
            batch = np.repeat(batch, 3, axis=-1)
        elif C == 1 and batch.shape[-1] == 3:
            batch = np.mean(batch, axis=-1, keepdims=True)

        # Resize using TensorFlow for efficiency
        batch_tensor = tf.convert_to_tensor(batch.astype(np.float32))
        resized = tf.image.resize(batch_tensor, [H, W], method='bilinear')

        return resized.numpy()


# =============================
# Model Prediction Functions
# =============================
class ModelPredictor:
    """Handles model predictions and analysis."""

    @staticmethod
    def predict_single(image_array: np.ndarray) -> Tuple[np.ndarray, int]:
        """Make prediction on single image."""
        if not TF_AVAILABLE or model is None:
            raise RuntimeError("Модель недоступна")

        probabilities = model.predict(image_array, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        return probabilities, predicted_class

    @staticmethod
    def predict_batch(batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on batch of images."""
        if not TF_AVAILABLE or model is None:
            raise RuntimeError("Модель недоступна")

        probabilities = model.predict(batch, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)
        return probabilities, predicted_classes

    @staticmethod
    def get_top_k_predictions(probabilities: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predictions with class names."""
        top_k_indices = np.argsort(probabilities)[-k:][::-1]
        return [(CLASS_NAMES[i], float(probabilities[i])) for i in top_k_indices]


# =============================
# Visualization Functions
# =============================
class Visualizer:
    """Handles all visualization creation."""

    @staticmethod
    def create_prediction_bar_chart(probabilities: np.ndarray, title: str = "") -> go.Figure:
        """Create bar chart for prediction probabilities."""
        fig = go.Figure(data=go.Bar(
            x=CLASS_NAMES,
            y=probabilities,
            text=[f"{p:.3f}" for p in probabilities],
            textposition='auto'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Classes",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            showlegend=False
        )

        return fig

    @staticmethod
    def create_top_k_bar_chart(top_k_predictions: List[Tuple[str, float]], title: str = "") -> go.Figure:
        """Create bar chart for top-k predictions."""
        classes, probs = zip(*top_k_predictions)

        fig = go.Figure(data=go.Bar(
            x=list(classes),
            y=list(probs),
            text=[f"{p:.3f}" for p in probs],
            textposition='auto'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Classes",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            showlegend=False
        )

        return fig

    @staticmethod
    def create_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: str = 'none') -> go.Figure:
        """Create confusion matrix heatmap."""
        # Calculate confusion matrix
        num_classes = len(CLASS_NAMES)
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for true_cls, pred_cls in zip(y_true, y_pred):
            cm[true_cls, pred_cls] += 1

        # Apply normalization if requested
        if normalize == 'row':
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cm_normalized = 100.0 * cm / row_sums
            text_array = [[f"{val:.1f}%" for val in row] for row in cm_normalized]
            title_suffix = " (Row Normalized %)"
        else:
            cm_normalized = cm.astype(float)
            text_array = [[str(val) for val in row] for row in cm]
            title_suffix = ""

        fig = go.Figure(data=go.Heatmap(
            z=cm_normalized,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            text=text_array,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale='Blues',
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z}<extra></extra>"
        ))

        fig.update_layout(
            title=f"Confusion Matrix{title_suffix}",
            xaxis_title="Predicted",
            yaxis_title="True",
            template='plotly_white',
            yaxis_autorange='reversed',
            width=700,
            height=600
        )

        return fig


# =============================
# Dash Application
# =============================
app = Dash(__name__)
app.title = "Fashion-MNIST VGG Classifier"


# Create warning component for missing dependencies
def create_warning_banner():
    """Create warning banner if TensorFlow or model is unavailable."""
    if TF_AVAILABLE and DATA_LOADED:
        return html.Div(style={'display': 'none'})

    return html.Div([
        html.Div([
            html.H4("⚠️ Увага: Обмежена функціональність", style={"color": "#856404", "margin": "0"}),
            html.P(IMPORT_ERROR_MSG or "Невідома помилка", style={"margin": "5px 0"}),
            html.Details([
                html.Summary("Інструкції для вирішення"),
                html.Div([
                    html.P("Для повної функціональності встановіть TensorFlow:"),
                    html.Pre(
                        "pip install tensorflow\n# або для GPU:\npip install tensorflow[and-cuda]",
                        style={"background": "#f8f9fa", "padding": "10px", "borderRadius": "4px"}
                    ),
                    html.P("Також переконайтеся, що файл моделі існує за шляхом:", style={"marginTop": "10px"}),
                    html.Code(MODEL_PATH, style={"background": "#f8f9fa", "padding": "2px 4px"})
                ])
            ])
        ])
    ], style={
        "background": "#fff3cd",
        "border": "1px solid #ffeaa7",
        "borderRadius": "8px",
        "padding": "15px",
        "margin": "10px 0"
    })


# Define app layout
app.layout = html.Div([
    html.H1("Fashion-MNIST VGG Classifier",
            style={"textAlign": "center", "marginBottom": "30px", "color": "#2c3e50"}),

    create_warning_banner(),

    # Preprocessing options section
    html.Div([
        html.H3("Налаштування обробки зображень"),
        dcc.Checklist(
            id='preprocessing-options',
            options=[
                {'label': 'Центральне обрізання (Center Crop)', 'value': 'center_crop'},
                {'label': 'Стиль Fashion-MNIST (28x28 + розмиття)', 'value': 'fmnist_style'}
            ],
            value=['center_crop'],
            inline=True,
            style={'marginBottom': '20px'}
        )
    ], style={'marginBottom': '30px', 'padding': '20px', 'background': '#f8f9fa', 'borderRadius': '8px'}),

    # Main content area
    html.Div([
        # Left column - Image upload
        html.Div([
            html.H3("Завантаження зображення"),
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    html.I(className="fas fa-upload", style={'marginRight': '10px'}),
                    'Перетягніть файл або клікніть для вибору'
                ]),
                style={
                    'width': '100%',
                    'height': '80px',
                    'lineHeight': '80px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '8px',
                    'textAlign': 'center',
                    'margin': '10px 0',
                    'background': '#fafafa',
                    'cursor': 'pointer' if DATA_LOADED else 'not-allowed',
                    'opacity': 1 if DATA_LOADED else 0.5
                },
                multiple=False,
                disabled=not DATA_LOADED
            ),
            html.Div(id='upload-result', style={'marginTop': '20px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),

        # Right column - Fashion-MNIST validation
        html.Div([
            html.H3("Тестування на Fashion-MNIST"),
            html.Div([
                html.Label("Індекс зображення:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='validation-index',
                    min=0,
                    max=len(x_test) - 1 if len(x_test) > 0 else 0,
                    step=1,
                    value=0,
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=not DATA_LOADED,
                    marks={i: str(i) for i in range(0, min(len(x_test), 10000), 1000)} if len(x_test) > 0 else {}
                ),
                html.Button(
                    "Випадковий приклад",
                    id='random-button',
                    n_clicks=0,
                    disabled=not DATA_LOADED,
                    style={
                        'marginTop': '10px',
                        'padding': '8px 16px',
                        'background': '#007bff' if DATA_LOADED else '#6c757d',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer' if DATA_LOADED else 'not-allowed'
                    }
                )
            ], style={'marginBottom': '20px'}),
            html.Div(id='validation-result')
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ], style={'marginBottom': '40px'}),

    html.Hr(),

    # Confusion Matrix section
    html.Div([
        html.H3("Аналіз продуктивності моделі"),
        html.Div([
            dcc.RadioItems(
                id='confusion-normalize',
                options=[
                    {'label': 'Абсолютні значення', 'value': 'none'},
                    {'label': 'Нормалізація по рядках (%)', 'value': 'row'}
                ],
                value='none',
                inline=True,
                style={'marginBottom': '10px'}
            ),
            html.Button(
                "Розрахувати матрицю помилок",
                id='confusion-matrix-button',
                n_clicks=0,
                disabled=not DATA_LOADED,
                style={
                    'padding': '10px 20px',
                    'background': '#28a745' if DATA_LOADED else '#6c757d',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer' if DATA_LOADED else 'not-allowed'
                }
            )
        ], style={'marginBottom': '20px'}),
        dcc.Loading(
            id="confusion-matrix-loading",
            children=[dcc.Graph(id='confusion-matrix-graph', figure=go.Figure())],
            type="circle"
        )
    ], style={'marginBottom': '40px'}),

    html.Hr(),

    # Model info section
    html.Div([
        html.H3("Інформація про модель"),
        html.Div([
            html.P([html.Strong("Шлях до моделі: "), MODEL_PATH]),
            html.P([html.Strong("Форма входу: "), str(INPUT_SHAPE)]),
            html.P([html.Strong("TensorFlow: "), "Доступний" if TF_AVAILABLE else "Недоступний"]),
            html.P([html.Strong("Модель завантажена: "), "Так" if DATA_LOADED else "Ні"]),
            html.P([html.Strong("Розмір тестового набору: "), str(len(x_test)) if len(x_test) > 0 else "0"])
        ], style={
            'background': '#f8f9fa',
            'padding': '15px',
            'borderRadius': '8px',
            'border': '1px solid #dee2e6'
        })
    ])
])


# =============================
# Callbacks
# =============================

@app.callback(
    Output('upload-result', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('preprocessing-options', 'value')
)
def handle_uploaded_image(contents, filename, preprocessing_options):
    """Handle uploaded image classification."""
    if not contents:
        return html.Div("Завантажте зображення для класифікації",
                        style={'color': '#6c757d', 'fontStyle': 'italic'})

    if not DATA_LOADED:
        return html.Div("Модель недоступна", style={'color': '#dc3545'})

    try:
        # Decode uploaded image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))

        # Preprocess and predict
        processed_image = ImagePreprocessor.preprocess_uploaded_image(
            image, preprocessing_options or []
        )
        probabilities, predicted_class = ModelPredictor.predict_single(processed_image)
        top_5_predictions = ModelPredictor.get_top_k_predictions(probabilities, 5)

        # Create thumbnail for display
        display_image = image.copy()
        display_image.thumbnail((200, 200), Image.Resampling.LANCZOS)

        # Convert to base64 for display
        buffer = io.BytesIO()
        display_image.save(buffer, format='PNG')
        thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Create visualization
        chart = Visualizer.create_top_k_bar_chart(
            top_5_predictions,
            f"Top-5 передбачень для {filename}"
        )

        return html.Div([
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{thumbnail_b64}",
                    style={
                        'border': '2px solid #dee2e6',
                        'borderRadius': '8px',
                        'marginRight': '20px'
                    }
                ),
                html.Div([
                    html.H4(f"Передбачення: {CLASS_NAMES[predicted_class]}",
                            style={'color': '#28a745', 'margin': '0 0 10px 0'}),
                    html.P(f"Впевненість: {probabilities[predicted_class]:.1%}",
                           style={'fontSize': '18px', 'margin': '0'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),

            dcc.Graph(figure=chart, style={'height': '400px'})
        ])

    except Exception as e:
        return html.Div(f"Помилка обробки зображення: {str(e)}",
                        style={'color': '#dc3545'})


@app.callback(
    Output('validation-result', 'children'),
    Input('validation-index', 'value'),
    Input('random-button', 'n_clicks')
)
def handle_validation_prediction(index, random_clicks):
    """Handle Fashion-MNIST validation prediction."""
    if not DATA_LOADED:
        return html.Div("Дані недоступні", style={'color': '#dc3545'})

    # Determine which input triggered the callback
    ctx = callback_context
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'random-button.n_clicks':
        index = np.random.randint(0, len(x_test))

    index = int(index or 0)

    try:
        # Get image and true label
        image_array = x_test[index]
        true_label = int(y_test[index])

        # Preprocess and predict
        processed_image = ImagePreprocessor.preprocess_fmnist_single(image_array)
        probabilities, predicted_class = ModelPredictor.predict_single(processed_image)

        # Create display image
        display_image = Image.fromarray((image_array * 255).astype(np.uint8))
        display_image = display_image.resize((200, 200), Image.Resampling.NEAREST)

        # Convert to base64
        buffer = io.BytesIO()
        display_image.save(buffer, format='PNG')
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Create probability chart
        chart = Visualizer.create_prediction_bar_chart(
            probabilities,
            f"Індекс {index}: Справжній={CLASS_NAMES[true_label]}, Передбачений={CLASS_NAMES[predicted_class]}"
        )

        # Determine prediction correctness
        is_correct = predicted_class == true_label
        status_color = '#28a745' if is_correct else '#dc3545'
        status_text = 'Правильно' if is_correct else 'Помилка'

        return html.Div([
            html.Div([
                html.Img(
                    src=f"data:image/png;base64,{image_b64}",
                    style={
                        'border': f'3px solid {status_color}',
                        'borderRadius': '8px',
                        'marginRight': '20px'
                    }
                ),
                html.Div([
                    html.H4(f"Індекс: {index}", style={'margin': '0 0 5px 0'}),
                    html.P(f"Справжній клас: {CLASS_NAMES[true_label]}",
                           style={'margin': '0 0 5px 0'}),
                    html.P(f"Передбачення: {CLASS_NAMES[predicted_class]}",
                           style={'margin': '0 0 5px 0'}),
                    html.P(f"Впевненість: {probabilities[predicted_class]:.1%}",
                           style={'margin': '0 0 5px 0'}),
                    html.P(status_text,
                           style={'color': status_color, 'fontWeight': 'bold', 'margin': '0'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center'})
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),

            dcc.Graph(figure=chart, style={'height': '400px'})
        ])

    except Exception as e:
        return html.Div(f"Помилка передбачення: {str(e)}",
                        style={'color': '#dc3545'})


@app.callback(
    Output('confusion-matrix-graph', 'figure'),
    Input('confusion-matrix-button', 'n_clicks'),
    State('confusion-normalize', 'value')
)
def update_confusion_matrix(n_clicks, normalize_mode):
    """Update confusion matrix visualization."""
    if not n_clicks or not DATA_LOADED:
        return go.Figure()

    try:
        # Preprocess entire test set
        processed_test = ImagePreprocessor.preprocess_fmnist_batch(x_test)

        # Get predictions
        probabilities, predicted_classes = ModelPredictor.predict_batch(processed_test)
        true_classes = y_test.astype(int)

        # Create confusion matrix
        return Visualizer.create_confusion_matrix(
            true_classes, predicted_classes, normalize_mode
        )

    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Помилка: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )


# =============================
# Run Application
# =============================
if __name__ == '__main__':
    print("Запуск Fashion-MNIST VGG Classifier...")
    print(f"TensorFlow доступний: {TF_AVAILABLE}")
    print(f"Дані завантажені: {DATA_LOADED}")

    app.run(
        debug=True,
        host='127.0.0.1',
        port=8050,
        use_reloader=False  # Avoid reloading issues with TensorFlow
    )