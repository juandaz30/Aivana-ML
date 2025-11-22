from flask import Flask, request, jsonify, send_from_directory
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import joblib

# Inicializacion del servidor con ruta de archivos estaticos
app = Flask(__name__, static_folder="../frontend", static_url_path="/")

# Cuando el usuario abre la app (dominio raiz /), lo redirige a index.html
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Directorio base '/backend'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Soportar dos escenarios:
# 1) app.py en la raiz (algorithms esta en BASE_DIR)
# 2) app.py dentro de backend/ (algorithms esta en el padre)
CANDIDATES = [BASE_DIR, os.path.dirname(BASE_DIR)]
for p in CANDIDATES:
    if p and p not in sys.path:
        sys.path.insert(0, p)

# Importar clases de algoritmos desde el paquete 'algorithms'
from algorithms.LinearRegression import LinearRegression
from algorithms.LogisticRegression import LogisticRegression
from algorithms.Perceptron import Perceptron
from algorithms.DecisionTreeClassifier import DecisionTreeClassifier
from algorithms.NaiveBayes import NaiveBayes
from algorithms.MLP import MLPClassifier
from algorithms.KMeans import KMeans
from algorithms.PCA import PCA

def _model_class_from_key(key: str):
    """Mapea la clave a la clase del modelo."""
    mapping = {
        "linear_regression": LinearRegression,
        "logistic_regression": LogisticRegression,
        "perceptron": Perceptron,
        "decision_tree": DecisionTreeClassifier,
        "naive_bayes": NaiveBayes,
        "mlp": MLPClassifier,
        "kmeans": KMeans,
        "pca": PCA,
    }
    return mapping.get(key)


def _to_jsonable(x):
    """Convierte numpy/objetos a algo JSON-friendly."""
    import numpy as _np

    if x is None:
        return None
    if isinstance(x, (int, float, str, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (_np.ndarray,)):
        return _to_jsonable(x.tolist())
    # cualquier otro tipo simple
    try:
        return _to_jsonable(x.__dict__)
    except Exception:
        return str(x)


def _looks_numeric_list(lst):
    """Heurística: ¿lista de números o lista de listas de números?"""
    try:
        if not isinstance(lst, list) or not lst:
            return False
        if isinstance(lst[0], list):
            # matriz
            return all(isinstance(v, (int, float)) or (isinstance(v, list) and all(isinstance(w, (int, float)) for w in v)) for v in lst)
        return all(isinstance(v, (int, float)) for v in lst)
    except Exception:
        return False


def _maybe_to_numpy(x):
    """Vuelve a numpy arrays lo que parezca numérico."""
    import numpy as _np
    if isinstance(x, list):
        if _looks_numeric_list(x):
            return _np.array(x, dtype=float)
        # si es lista de strings u otros, déjalo igual
        return [_maybe_to_numpy(v) for v in x]
    if isinstance(x, dict):
        return {k: _maybe_to_numpy(v) for k, v in x.items()}
    return x


def _dump_model_state(model):
    """
    Serializa el __dict__ del modelo a JSON-friendly.
    No adivinamos nombres; guardamos TODO lo simple.
    """
    state = {}
    for k, v in getattr(model, "__dict__", {}).items():
        # Opcional: ignorar privados muy grandes si hiciera falta
        state[k] = _to_jsonable(v)
    return state


def _restore_model_from_state(alg_key, state: dict):
    """
    Reconstruye una instancia del modelo y le inyecta su estado.
    """
    cls = _model_class_from_key(alg_key)
    if cls is None:
        raise RuntimeError(f"No se puede reconstruir modelo para {alg_key}")
    model = cls()  # instancia "vacía"
    # reinyectar atributos
    for k, v in (state or {}).items():
        try:
            setattr(model, k, _maybe_to_numpy(v))
        except Exception:
            # si algún atributo no aplica, lo ignoramos hola
            pass
    return model


def _is_supervised(alg_key: str) -> bool:
    return alg_key in {"linear_regression", "logistic_regression", "perceptron", "decision_tree", "naive_bayes", "mlp"}


def _task_kind(alg_key: str) -> str:
    if alg_key in {"linear_regression"}:
        return "regression"
    if alg_key in {"logistic_regression", "perceptron", "decision_tree", "naive_bayes", "mlp"}:
        return "classification"
    if alg_key in {"kmeans"}:
        return "clustering"
    if alg_key in {"pca"}:
        return "dimensionality_reduction"
    return "unknown"


def _series_from_sequence(seq):
    if not seq:
        return None
    cleaned = []
    for val in seq:
        if val is None:
            cleaned.append(None)
            continue
        try:
            cleaned.append(float(val))
        except Exception:
            try:
                cleaned.append(float(np.asarray(val).item()))
            except Exception:
                continue
    return cleaned if cleaned else None


def _training_curve_payload(model, alg_key):
    if model is None:
        return None

    def make_series(values, name):
        seq = _series_from_sequence(values)
        if not seq:
            return None
        return {"name": name, "values": seq}

    if alg_key == "linear_regression":
        serie = make_series(getattr(model, "loss_history_", None), "MSE en entrenamiento")
        if serie:
            return {
                "title": "Error cuadrático medio por iteración",
                "xLabel": "Iteraciones",
                "yLabel": "Error cuadrático medio",
                "series": [serie]
            }

    if alg_key == "logistic_regression":
        serie = make_series(getattr(model, "loss_history", None), "Pérdida logarítmica")
        if serie:
            return {
                "title": "Pérdida durante el entrenamiento",
                "xLabel": "Iteraciones",
                "yLabel": "Log loss",
                "series": [serie]
            }

    if alg_key == "perceptron":
        series = []
        total = make_series(getattr(model, "errors_history_", None), "Errores totales")
        if total:
            series.append(total)
        per_class = getattr(model, "errors_history_per_class_", {}) or {}
        classes = list(getattr(model, "classes_", []))
        for idx, values in per_class.items():
            serie = make_series(values, "")
            if not serie:
                continue
            try:
                idx_int = int(idx)
            except Exception:
                idx_int = idx
            label = classes[idx_int] if 0 <= idx_int < len(classes) else idx_int
            if label is None:
                serie["name"] = "Errores por clase"
            else:
                serie["name"] = f"Errores clase {label}"
            series.append(serie)
        if series:
            return {
                "title": "Errores por época (Perceptrón)",
                "xLabel": "Épocas",
                "yLabel": "Errores",
                "series": series
            }

    if alg_key == "mlp":
        series = []
        loss = make_series(getattr(model, "loss_history_", None), "Pérdida entrenamiento")
        if loss:
            series.append(loss)
        val_loss = make_series(getattr(model, "val_loss_history_", None), "Pérdida validación")
        if val_loss:
            series.append(val_loss)
        if series:
            return {
                "title": "Pérdida por época (Red neuronal)",
                "xLabel": "Épocas",
                "yLabel": "Pérdida",
                "series": series
            }

    if alg_key == "kmeans":
        serie = make_series(getattr(model, "inertia_history_", None), "Inercia del mejor intento")
        if serie:
            return {
                "title": "Inercia por iteración (K-Means)",
                "xLabel": "Iteraciones",
                "yLabel": "Inercia",
                "series": [serie],
                "notes": "Valores registrados para la ejecución con mejor resultado."
            }

    return None

# ------------------- CONFIG JSON's -------------------
# Rutas de los archivos JSON
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PROJECTS_FILE = os.path.join(BASE_DIR, "projects.json")
# Directorio para datasets subidos en proyectos
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
# Crear directorio de datasets si no existe
os.makedirs(DATASETS_DIR, exist_ok=True)

def load_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f, indent=4)  # inicializar archivo JSON vacio
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []  # si el JSON esta vacio o corrupto

# Cargar datos de usuarios y proyectos
users = load_json(USERS_FILE)
projects = load_json(PROJECTS_FILE)

# ------------------- Funciones Auxiliares -------------------
# Funcion auxiliar para guardar estructura Python en un archivo JSON
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

# Funcion auxiliar para buscar un proyecto por id y usuario
def find_project(project_id, user, missing_id_msg=None, not_found_msg=None):
    if not project_id:
        return None, ((missing_id_msg or "Faltan parametros"), 400)
    if not user:
        return None, ("Faltan parametros", 400)
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return None, ((not_found_msg or "Proyecto no encontrado"), 404)
    return project, None

# Funcion auxiliar para leer un CSV a DataFrame, devolviendo error en caso de fallo
def read_csv_or_error(file_path, descriptor="dataset"):
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, f"Error leyendo {descriptor}: {e}"

# Funcion auxiliar para obtener vista previa (columnas y primeras filas) de un CSV
def get_preview_from_csv(file_path, n=5, error_prefix="CSV"):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return None, None, f"Error leyendo {error_prefix}: {e}"
    preview = df.head(n).to_dict(orient="records")
    columns = list(df.columns)
    # Reemplazar NaN con None para JSON valido
    preview = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in preview]
    return columns, preview, None

# ------------------- Registro -------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    # Verificar que los campos no esten vacios
    if not name or not email or not password:
        return jsonify({"success": False, "msg": "Todos los campos de registro son obligatorios"}), 400
    if any(u["email"] == email for u in users):
        return jsonify({"success": False, "msg": "El correo ya está registrado"}), 400
    new_user = {"name": name, "email": email, "password": password}
    users.append(new_user)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)
    return jsonify({"success": True, "msg": "Registro exitoso"}), 201

# ------------------- Iniciar Sesion -------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    user = next((u for u in users if u["email"] == email and u["password"] == password), None)
    if user:
        return jsonify({
            "success": True,
            "msg": f"Bienvenido {user['name']}!",
            "user": {
                "name": user["name"],
                "email": user["email"]
            }
        }), 200
    else:
        return jsonify({"success": False, "msg": "Credenciales inválidas"}), 401

# ------------------- Crear Proyecto -------------------
@app.route("/create_project", methods=["POST"])
def create_project():
    data = request.get_json()
    # Extraer campos del JSON
    name = data.get("name")
    desc = data.get("description", "")
    user = data.get("user")
    # Validar que se defina un nombre para el proyecto
    if not name:
        return jsonify({"success": False, "msg": "El nombre del proyecto es obligatorio"}), 400
    # Crear objeto de proyecto con ID unico
    new_id = max((p["id"] for p in projects), default=0) + 1
    project = {
        "id": new_id,
        "name": name,
        "description": desc,
        "user": user,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    # Agregar proyecto al listado y guardar en JSON
    projects.append(project)
    save_json(PROJECTS_FILE, projects)
    return jsonify({"success": True, "msg": "Proyecto creado con éxito", "project": project}), 201

# ------------------- Cargar Proyectos -------------------
@app.route("/get_projects", methods=["GET"])
def get_projects():
    # Leer parametro de usuario (ej: /get_projects?user=correo)
    user = request.args.get("user")
    if user:
        user_projects = [p for p in projects if p["user"] == user]
        return jsonify({"success": True, "projects": user_projects}), 200
    # Si no se envio usuario, retornar error (uso solo para administrador)
    return jsonify({"success": False, "msg": "No hay un usuario asociado"}), 200

# ------------------- Editar Proyecto -------------------
@app.route("/edit_project", methods=["PUT"])
def edit_project():
    data = request.get_json()
    project_id = data.get("id")
    new_name = data.get("name")
    new_desc = data.get("description", "")
    user = data.get("user")
    # Validar campos obligatorios
    if not new_name:
        return jsonify({"success": False, "msg": "El nombre del proyecto no puede estar vacío"}), 400
    # Buscar proyecto existente
    project, error = find_project(project_id, user, "Falta el ID del proyecto", "Proyecto no encontrado o no pertenece al usuario")
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    # Validar nombre de proyecto no duplicado para el mismo usuario
    if any(p["name"] == new_name and p["user"] == user and p["id"] != project_id for p in projects):
        return jsonify({"success": False, "msg": "Ya existe un proyecto con ese nombre"}), 400
    # Actualizar datos del proyecto
    project["name"] = new_name
    project["description"] = new_desc
    save_json(PROJECTS_FILE, projects)
    return jsonify({"success": True, "msg": "Proyecto actualizado correctamente", "project": project}), 200

# ------------------- Eliminar Proyecto -------------------
@app.route("/delete_project", methods=["DELETE"])
def delete_project():
    data = request.get_json()
    project_id = data.get("id")
    user = data.get("user")
    # Buscar proyecto a eliminar
    project, error = find_project(project_id, user, "Falta el ID del proyecto", "Proyecto no encontrado o no pertenece al usuario")
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    # Eliminar proyecto del listado y guardar cambios
    projects.remove(project)
    save_json(PROJECTS_FILE, projects)
    return jsonify({"success": True, "msg": "Proyecto eliminado correctamente"}), 200

# ------------------- Subir Dataset -------------------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    # Obtener ID de proyecto, usuario y archivo del formulario
    project_id = request.form.get("project_id", type=int)
    user = request.form.get("user")
    file = request.files.get("file")
    if not project_id or not file:
        return jsonify({"success": False, "msg": "Faltan datos o archivo"}), 400
    # Verificar que el proyecto exista
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    # Crear directorio para almacenar el dataset del proyecto
    project_dir = os.path.join(DATASETS_DIR, f"project_{project_id}")
    os.makedirs(project_dir, exist_ok=True)
    # Guardar el archivo subido en el directorio del proyecto
    filename = file.filename
    file_path = os.path.join(project_dir, filename)
    file.save(file_path)
    # Guardar ruta del dataset en el proyecto
    project["dataset_path"] = file_path.replace("\\", "/")
    save_json(PROJECTS_FILE, projects)
    # Generar vista previa del dataset subido
    columns, preview, err = get_preview_from_csv(file_path, n=5, error_prefix="CSV")
    if err:
        return jsonify({"success": False, "msg": err}), 400
    return jsonify({
        "success": True,
        "msg": "Dataset subido correctamente",
        "project": project,
        "preview": preview,
        "columns": columns
    }), 200

# ------------------- Preprocesar Datos -------------------
@app.route("/preprocess_data", methods=["POST"])
def preprocess_data():
    data = request.get_json()
    project_id = data.get("project_id")
    user = data.get("user")
    # Verificar parametros obligatorios
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    # Obtener ruta del dataset del proyecto
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "No se encontró el dataset para este proyecto"}), 404
    df, err = read_csv_or_error(dataset_path, "dataset")
    if err:
        return jsonify({"success": False, "msg": err}), 400
    original_shape = df.shape
    # --- LIMPIEZA AUTOMÁTICA FINAL ---
    # 1) Eliminar columnas completamente vacías
    df.dropna(axis=1, how="all", inplace=True)
    # 2) Detectar columnas numéricas de forma robusta:
    #    - Intentamos convertir a numérico con errors='coerce'
    #    - Calculamos el % de valores convertibles (no NaN tras la conversión)
    #    - Si ese % >= 0.6, consideramos la columna como numérica
    numeric_cols = []
    for col in df.columns:
        conv = pd.to_numeric(df[col], errors='coerce')
        non_empty = df[col].replace(r'^\s*$', np.nan, regex=True).notna().sum()
        if non_empty == 0:
            continue
        convertible_ratio = conv.notna().sum() / non_empty
        if convertible_ratio >= 0.6 and conv.notna().sum() >= 2:
            numeric_cols.append(col)
    # 3) Aplicar conversión definitiva en numéricas y rellenar con la media
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
    # 4) Eliminar filas que aún tengan valores no numéricos (NaN) en columnas numéricas
    if numeric_cols:
        mask_valid = np.ones(len(df), dtype=bool)
        for col in numeric_cols:
            mask_valid &= pd.to_numeric(df[col], errors='coerce').notna()
        df = df[mask_valid]
    # 5) Redondear numéricas para presentación
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].round(4)
    # 6) Rellenar categóricas (no numéricas) vacías con "NA"
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    for col in categorical_cols:
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        df[col] = df[col].fillna("NA")
    # 7) Eliminar duplicados
    df.drop_duplicates(inplace=True)
    # 8) Sobrescribir el dataset original (mismo path)
    clean_path = project["dataset_path"]
    df.to_csv(clean_path, index=False)
    # 9) Actualizar referencia en el proyecto (ruta del dataset limpio)
    project["clean_dataset_path"] = clean_path.replace("\\", "/")
    save_json(PROJECTS_FILE, projects)
    # --- INFORME ---
    cleaned_shape = df.shape
    dropped_cols = list(set(df.columns) ^ set(pd.read_csv(dataset_path).columns))
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    missing_values = df.isnull().sum().to_dict()
    summary = {
        "filas_antes": original_shape[0],
        "filas_despues": cleaned_shape[0],
        "columnas_antes": original_shape[1],
        "columnas_despues": cleaned_shape[1],
        "duplicados_eliminados": int(original_shape[0] - cleaned_shape[0]),
        "columnas_eliminadas": dropped_cols
    }
    return jsonify({
        "success": True,
        "msg": "Preprocesamiento completado y dataset limpio guardado.",
        "summary": summary,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_values": missing_values,
        "clean_path": project["clean_dataset_path"]
    }), 200

# ------------------- Obtener el Dataset Preprocesado -------------------
@app.route("/get_clean_dataset", methods=["GET"])
def get_clean_dataset():
    project_id = request.args.get("project_id", type=int)
    user = request.args.get("user")
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado"}), 404
    columns, preview, err = get_preview_from_csv(dataset_path, n=5, error_prefix="dataset")
    if err:
        return jsonify({"success": False, "msg": err}), 400
    return jsonify({
        "success": True,
        "columns": columns,
        "preview": preview
    }), 200

# ------------------- Recomendar Modelo -------------------
@app.route("/recommend_model", methods=["POST"])
def recommend_model():
    data = request.get_json()
    project_id = data.get("project_id")
    user = data.get("user")
    target = data.get("target")
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    # Verificar existencia del dataset
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado para este proyecto"}), 404
    df, err = read_csv_or_error(dataset_path, "dataset")
    if err:
        return jsonify({"success": False, "msg": err}), 400
    n, p_total = df.shape
    # Detectar columnas numericas y no numericas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    # Si no envian target y existe columna 'Y', usarla; si no, caso no supervisado
    if not target:
        if "Y" in df.columns:
            target = "Y"
        else:
            recs = [
                {"model": "PCA", "score": 0.9, "why": "No se definió variable objetivo; reducción de dimensión para exploración."},
                {"model": "KMeans", "score": 0.8, "why": "No se definió objetivo; posible agrupamiento natural en los datos."}
            ]
            return jsonify({"success": True, "task": "unsupervised", "recommendations": recs})
    if target not in df.columns:
        return jsonify({"success": False, "msg": f"La columna objetivo '{target}' no existe en el dataset"}), 400
    # Separar X / y
    y = df[target]
    X = df.drop(columns=[target])
    # Recalcular tipos sin la y
    X_num = X.select_dtypes(include=["number"]).columns.tolist()
    X_cat = [c for c in X.columns if c not in X_num]
    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    unique_values = y.nunique(dropna=True)
    unique_ratio = unique_values / max(len(y), 1)
    looks_categorical_numeric = is_numeric_target and (unique_values <= 20 and unique_ratio < 0.2)
    task = "classification" if (not is_numeric_target or looks_categorical_numeric) else "regression"
    recommendations = []
    if task == "regression":
        # Heurística simple de linealidad: correlación absoluta media entre y y features numericas
        corr_score = 0.0
        if len(X_num) > 0:
            corrs = []
            for c in X_num:
                try:
                    corrs.append(abs(np.corrcoef(X[c], y)[0, 1]))
                except Exception:
                    pass
            if corrs:
                corr_score = float(np.nanmean(corrs))
        # Pocas features + buena correlación → LinearRegression
        score_lr = 0.6 + 0.4 * min(1.0, corr_score) - 0.1 * max(0, len(X_num) - 10) / 20
        recommendations.append({
            "model": "LinearRegression",
            "score": round(max(0.1, min(1.0, score_lr)), 3),
            "why": f"Objetivo numérico; correlación media≈{corr_score:.2f}. Regresión lineal es base fuerte y explicable."
        })
        # Si hay indicio de no linealidad (muchas variables o baja correlacion) → arbol y MLP
        nonlin_hint = (corr_score < 0.3) or (len(X_num) >= 10)
        if nonlin_hint:
            recommendations.append({
                "model": "DecisionTreeClassifier (usar versión regresor luego)",
                "score": 0.55,
                "why": "No linealidad o muchas variables; árboles capturan relaciones no lineales. (Para regresión, usar DecisionTreeRegressor)."
            })
            recommendations.append({
                "model": "MLP",
                "score": 0.5,
                "why": "Red de perceptrón multicapa puede capturar no linealidad si tienes suficientes datos."
            })
    else:  # classification
        # Cardinalidad y desbalance
        classes = y.astype(str).unique().tolist()
        k = len(classes)
        counts = y.astype(str).value_counts(normalize=True)
        imbalance = float(counts.max()) if not counts.empty else 1.0
        # Binaria → LogisticRegression como baseline
        if k == 2:
            base = 0.75 - 0.2 * (imbalance - 0.5)
            recommendations.append({
                "model": "LogisticRegression",
                "score": round(max(0.1, min(1.0, base)), 3),
                "why": f"Clasificación binaria; baseline rápido y explicable. Desbalance≈{imbalance:.2f}."
            })
            # Si hay no linealidad (muchas numericas o multiples clases) → arbol y MLP
            if len(X_num) >= 5 or k > 2:
                recommendations.append({
                    "model": "DecisionTreeClassifier",
                    "score": 0.6,
                    "why": "Puede manejar fronteras no lineales y variables mixtas."
                })
                recommendations.append({
                    "model": "MLP",
                    "score": 0.55,
                    "why": "Puede capturar patrones complejos si hay suficientes muestras."
                })
            else:
                recommendations.append({
                    "model": "Perceptron",
                    "score": 0.5,
                    "why": "Si separable casi lineal, Perceptrón funciona muy rápido."
                })
        else:
            # Multiclase → Arbol como baseline; Naive Bayes si predominan categoricas
            recommendations.append({
                "model": "DecisionTreeClassifier",
                "score": 0.65,
                "why": f"Multiclase ({k} clases); árboles son versátiles e interpretables."
            })
            if len(X_cat) > len(X_num):
                recommendations.append({
                    "model": "NaiveBayes",
                    "score": 0.6,
                    "why": "Muchas variables categóricas; Naive Bayes es simple y efectivo."
                })
            else:
                recommendations.append({
                    "model": "MLP",
                    "score": 0.55,
                    "why": "Puede modelar fronteras complejas en multiclase si hay datos."
                })
    # Si no hay recomendaciones (por algun motivo), sugerir PCA/KMeans por defecto
    if not recommendations:
        recommendations = [
            {"model": "PCA", "score": 0.6, "why": "No se pudo inferir tarea claramente; explora reducción de dimensión."},
            {"model": "KMeans", "score": 0.5, "why": "Prueba agrupamiento no supervisado."}
        ]
    # Ordenar recomendaciones por score desc y tomar top-3
    recommendations = sorted(recommendations, key=lambda r: r["score"], reverse=True)[:3]
    return jsonify({
        "success": True,
        "task": task,
        "target": target,
        "n_rows": int(n),
        "n_features": int(len(X.columns)),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "recommendations": recommendations
    }), 200

# Parametros por defecto de los modelos
# Parametros por defecto de los modelos (TODOS los del __init__ de cada clase)
DEFAULT_PARAMS = {
    "linear_regression": {
        "learning_rate": 0.01,
        "n_iterations": 1000,
        "fit_intercept": True,
        "early_stopping": True,
        "tolerance": 1e-6,
        "patience": 10,
        "normalize": True,
        "max_grad_norm": 1e6,
        "verbose": False
    },
    "logistic_regression": {
        "learning_rate": 0.1,
        "n_iterations": 2000,
        "tolerance": 1e-6,
        "early_stopping": False,
        "verbose": False,
        "fit_intercept": True,
        "l2": 0.0,
        "decision_threshold": 0.5,
        "clip": 1e-15
    },
    "perceptron": {
        "learning_rate": 1.0,
        "n_iterations": 1000,
        "fit_intercept": True,
        "shuffle": True,
        "random_state": None,
        "early_stopping": False,
        "patience": 5,
        "verbose": False,
        "callbacks": None
    },
    "decision_tree": {
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": None,
        "random_state": None,
        "verbose": False
    },
    "naive_bayes": {
        "nb_type": "gaussian",
        "var_smoothing": 1e-9,
        "alpha": 1.0,
        "class_priors": None,
        "binarize": None,
        "decision_threshold": 0.5
    },
    "mlp": {
        "hidden_layers": [32],
        "activation": "relu",
        "learning_rate": 0.01,
        "n_iterations": 2000,
        "batch_size": None,
        "l2": 0.0,
        "early_stopping": False,
        "patience": 10,
        "validation_split": 0.0,
        "tolerance": 1e-6,
        "decision_threshold": 0.5,
        "random_state": None,
        "verbose": False,
        "callbacks": None
    },
    "kmeans": {
        "n_clusters": 8,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "random_state": None,
        "verbose": False,
        "callbacks": None
    },
    "pca": {
        "n_components": None,
        "whiten": False,
        "copy": True
    }
}


@app.route("/get_default_params", methods=["GET"])
def get_default_params():
    """Devuelve el diccionario de parámetros por defecto para cada algoritmo."""
    return jsonify({"success": True, "defaults": DEFAULT_PARAMS}), 200


@app.route("/set_model_params", methods=["POST"])
def set_model_params():
    """
    Actualiza solo los parámetros del modelo del proyecto.
    Body JSON:
      - project_id (int)
      - user (str)
      - algorithm_key (str)
      - params (dict)
    """
    data = request.get_json() or {}
    project_id = data.get("project_id")
    user = data.get("user")
    algorithm_key = data.get("algorithm_key")
    params = data.get("params") or {}

    if not project_id or not user or not algorithm_key:
        return jsonify({"success": False, "msg": "Faltan parámetros (project_id, user, algorithm_key)"}), 400

    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code

    base_defaults = DEFAULT_PARAMS.get(algorithm_key, {})
    merged = {**base_defaults, **params}

    current_cfg = project.get("model_cfg") or {}
    project["model_cfg"] = {
        "category": current_cfg.get("category") or _task_kind(algorithm_key),
        "algorithm_key": algorithm_key,
        "params": merged
    }
    save_json(PROJECTS_FILE, projects)
    return jsonify({
        "success": True,
        "msg": "Parámetros del modelo actualizados",
        "model_cfg": project["model_cfg"]
    }), 200


# ------------------- Seleccionar Modelo -------------------
@app.route("/select_model", methods=["POST"])
def api_models_select():
    data = request.get_json() or {}
    project_id = data.get("project_id")
    user = data.get("user")
    category = data.get("category")
    algorithm_key = data.get("algorithm_key")
    params = data.get("params") or {}
    if not project_id or not user or not category or not algorithm_key:
        return jsonify({"success": False, "msg": "Faltan parámetros (project_id, user, category, algorithm_key)"}), 400
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    merged_params = {**DEFAULT_PARAMS.get(algorithm_key, {}), **params}
    project["model_cfg"] = {
        "category": category,
        "algorithm_key": algorithm_key,
        "params": merged_params
    }
    save_json(PROJECTS_FILE, projects)
    return jsonify({
        "success": True,
        "msg": "Modelo seleccionado y guardado",
        "model_cfg": project["model_cfg"],
        "project": project
    }), 200

# Definir funciones auxiliares de entrenamiento
def _train_test_split(X, y=None, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = max(1, int(len(X) * test_size))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    if y is None:
        return X.iloc[train_idx], X.iloc[test_idx], None, None
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

def _is_supervised(alg_key):
    return alg_key in {"linear_regression", "logistic_regression", "perceptron", "decision_tree", "naive_bayes", "mlp"}

def _task_kind(alg_key):
    if alg_key == "linear_regression":
        return "regression"
    if alg_key in {"logistic_regression", "perceptron", "decision_tree", "naive_bayes", "mlp"}:
        return "classification"
    if alg_key == "kmeans":
        return "clustering"
    if alg_key == "pca":
        return "dimensionality_reduction"
    return "unknown"

# ------------------- Entrenar Modelo -------------------
# Mapear claves de algoritmo a clases correspondientes
MODEL_CLASSES = {
    "linear_regression": LinearRegression,
    "logistic_regression": LogisticRegression,
    "perceptron": Perceptron,
    "decision_tree": DecisionTreeClassifier,
    "naive_bayes": NaiveBayes,
    "mlp": MLPClassifier,
    "kmeans": KMeans,
    "pca": PCA
}

@app.route("/train_model", methods=["POST"])
def train_model():
    data = request.get_json() or {}
    project_id = data.get("project_id")
    user = data.get("user")
    target = data.get("target")
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))
    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros (project_id, user)"}), 400
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    if "model_cfg" not in project:
        return jsonify({"success": False, "msg": "No hay modelo seleccionado aún"}), 400
    cfg = project["model_cfg"]
    alg_key = cfg.get("algorithm_key")
    params = cfg.get("params", {})
    kind = _task_kind(alg_key)
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado para este proyecto"}), 404
    df, err = read_csv_or_error(dataset_path, "dataset")
    if err:
        return jsonify({"success": False, "msg": err}), 400
    # Instanciar modelo según algoritmo seleccionado
    cls = MODEL_CLASSES.get(alg_key)
    if not cls:
        return jsonify({"success": False, "msg": f"Algoritmo no soportado: {alg_key}"}), 400
    try:
        model = cls(**params)
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error inicializando el modelo: {e}"}), 400
    metrics = {}
    preview = {}
    feature_candidates = []
    training_curve = None
    try:
        if kind in {"regression", "classification"}:
            if not target or target not in df.columns:
                return jsonify({"success": False, "msg": "Debes indicar la columna objetivo (target) válida"}), 400
            y = df[target]
            feature_candidates = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if c != target]
            if not feature_candidates:
                return jsonify({"success": False, "msg": "No hay columnas numéricas para entrenar (X)"}), 400
            X = df[feature_candidates]
            y_encoded = y.copy()
            label_map = None
            if kind == "classification" and not pd.api.types.is_numeric_dtype(y):
                classes = sorted(y.astype(str).unique().tolist())
                label_map = {c: i for i, c in enumerate(classes)}
                y_encoded = y.astype(str).map(label_map)
            Xtr, Xte, ytr, yte = _train_test_split(X, y_encoded, test_size=test_size, seed=random_state)
            model.fit(Xtr.values, ytr.values if ytr is not None else None)
            if not hasattr(model, "predict"):
                return jsonify({"success": False, "msg": "El modelo no expone método predict"}), 400
            yhat = model.predict(Xte.values)
            yhat = np.array(yhat).reshape(-1)
            if kind == "regression":
                y_true = yte.values.astype(float)
                mse = float(np.mean((yhat - y_true) ** 2))
                mae = float(np.mean(np.abs(yhat - y_true)))
                rmse = float(np.sqrt(mse))
                denom = np.where(y_true == 0, 1e-8, np.abs(y_true))
                mape = float(np.mean(np.abs((y_true - yhat) / denom)))
                ss_res = float(np.sum((y_true - yhat) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) or 1.0
                r2 = 1.0 - ss_res / ss_tot
                metrics = {"task": "regression", "mse": mse, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}
                try:
                    if alg_key == "linear_regression":
                        # pesos aprendidos en el espacio normalizado
                        w = np.asarray(getattr(model, "weights", None), dtype=float).ravel()
                        fit_intercept = bool(getattr(model, "fit_intercept", True))
                        normalized = bool(getattr(model, "normalize", False))
                        if normalized and hasattr(model, "x_mean_") and hasattr(model, "x_std_") and hasattr(model, "y_mean_"):
                            # quitar término de bias del vector de pesos
                            if fit_intercept:
                                w0 = float(w[0])          # bias en espacio normalizado
                                w_rest = w[1:]            # coeficientes asociados a cada Xn
                            else:
                                w0 = 0.0
                                w_rest = w
                            x_mean = np.asarray(model.x_mean_, dtype=float)
                            x_std  = np.asarray(model.x_std_,  dtype=float)
                            x_std  = np.where(x_std == 0, 1.0, x_std)   # evitar división por 0
                            y_mean = float(model.y_mean_)
                            # 1) pasar coeficientes al espacio original
                            coef_orig = w_rest / x_std
                            # 2) intercept en espacio original:
                            intercept_orig = y_mean + w0 - float(np.dot(coef_orig, x_mean))
                            metrics["intercept"] = float(intercept_orig)
                            metrics["coefficients"] = {c: float(v) for c, v in zip(feature_candidates, coef_orig.tolist())}
                            metrics["normalized_inputs"] = True
                        else:
                            # modelo sin normalización → usar coeficientes tal cual
                            if fit_intercept:
                                metrics["intercept"] = float(w[0])
                                metrics["coefficients"] = {c: float(v) for c, v in zip(feature_candidates, w[1:].tolist())}
                            else:
                                metrics["intercept"] = 0.0
                                metrics["coefficients"] = {c: float(v) for c, v in zip(feature_candidates, w.tolist())}
                            metrics["normalized_inputs"] = False
                except Exception:
                    pass  # si algo falla, continuar sin interrumpir
            else:
                y_true = yte.values.astype(int)
                y_pred = yhat.astype(int)
                acc = float(np.mean(y_true == y_pred))
                labels = np.unique(np.concatenate([y_true, y_pred]))
                cm = {int(lbl): {int(lbl2): int(np.sum((y_true == lbl) & (y_pred == lbl2))) for lbl2 in labels} for lbl in labels}
                precisions = []
                recalls = []
                f1s = []
                per_class = {}
                for lbl in labels:
                    tp = float(np.sum((y_true == lbl) & (y_pred == lbl)))
                    fp = float(np.sum((y_true != lbl) & (y_pred == lbl)))
                    fn = float(np.sum((y_true == lbl) & (y_pred != lbl)))
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)
                    per_class[int(lbl)] = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "support": int(np.sum(y_true == lbl))
                    }
                precision_macro = float(np.mean(precisions)) if precisions else 0.0
                recall_macro = float(np.mean(recalls)) if recalls else 0.0
                f1_macro = float(np.mean(f1s)) if f1s else 0.0
                if label_map:
                    metrics["label_map"] = label_map
                metrics.update({
                    "task": "classification",
                    "accuracy": acc,
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_macro": f1_macro,
                    "per_class": per_class,
                    "confusion_matrix": cm
                })
            prev_k = min(10, len(Xte))
            preview = {
                "columns": ["y_true", "y_pred"] + feature_candidates[:10],
                "rows": [
                    [
                        (None if yte is None else (int(yte.iloc[i]) if kind == "classification" else float(yte.iloc[i]))),
                        (int(yhat[i]) if kind == "classification" else float(yhat[i]))
                    ] + [
                        (None if pd.isna(Xte.iloc[i, j]) else float(Xte.iloc[i, j]))
                        for j in range(min(10, Xte.shape[1]))
                    ]
                    for i in range(prev_k)
                ]
            }
        elif kind == "clustering":
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            # Si el usuario eligió una columna objetivo por error, no la usemos como feature
            if target and target in num_cols:
                num_cols = [c for c in num_cols if c != target]
            if not num_cols:
                return jsonify({"success": False, "msg": "No hay columnas numéricas para K-Means"}), 400
            X = df[num_cols]
            model.fit(X.values)
            labels = model.predict(X.values) if hasattr(model, "predict") else getattr(model, "labels_", None)
            labels = np.array(labels) if labels is not None else np.zeros(len(X), dtype=int)
            inertia = getattr(model, "inertia_", None)
            metrics = {"task": "clustering", "n_samples": int(len(X))}
            if inertia is not None:
                metrics["inertia"] = float(inertia)
            # Distribución de tamaños por cluster y centros (si están disponibles)
            try:
                unique, counts = np.unique(labels, return_counts=True)
                metrics["cluster_sizes"] = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
            except Exception:
                pass
            try:
                centers = getattr(model, "cluster_centers_", None)
                if centers is not None:
                    centers = np.asarray(centers, dtype=float)
                    metrics["cluster_centers"] = {
                        int(i): {c: float(centers[i, j]) for j, c in enumerate(num_cols)}
                        for i in range(min(centers.shape[0], 20))
                    }
            except Exception:
                pass
            prev_k = min(10, len(X))
            preview = {
                "columns": ["cluster"] + num_cols[:10],
                "rows": [
                    [int(labels[i])] + [
                        (None if pd.isna(X.iloc[i, j]) else float(X.iloc[i, j]))
                        for j in range(min(10, X.shape[1]))
                    ]
                    for i in range(prev_k)
                ]
            }
        elif kind == "dimensionality_reduction":
            num_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if not num_cols:
                return jsonify({"success": False, "msg": "No hay columnas numéricas para PCA"}), 400
            X = df[num_cols]
            if hasattr(model, "fit_transform"):
                Z = model.fit_transform(X.values)
            else:
                model.fit(X.values)
                Z = model.transform(X.values) if hasattr(model, "transform") else X.values
            explained = getattr(model, "explained_variance_ratio_", None)
            if explained is not None:
                explained = [float(v) for v in np.array(explained).ravel().tolist()]
            metrics = {"task": "dimensionality_reduction", "explained_variance_ratio": explained}
            prev_k = min(10, Z.shape[0])
            prev_m = min(5, Z.shape[1]) if len(Z.shape) == 2 else 1
            preview = {
                "columns": [f"PC{i+1}" for i in range(prev_m)],
                "rows": [
                    [float(Z[i, j]) for j in range(prev_m)]
                    for i in range(prev_k)
                ]
            }
        else:
            return jsonify({"success": False, "msg": f"Tarea no soportada para {alg_key}"}), 400
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error durante el entrenamiento: {e}"}), 400
    # Persistir modelo entrenado en archivo .pkl
    try:
        if kind in {"regression", "classification"}:
            features_used = feature_candidates
            target_used = target
        elif kind == "clustering":
            # num_cols fue definido en el bloque de clustering
            num_cols = num_cols if 'num_cols' in locals() else df.select_dtypes(include=["number"]).columns.tolist()
            features_used = num_cols
            target_used = None
        elif kind == "dimensionality_reduction":
            features_used = num_cols  # definido en el bloque de PCA
            target_used = None
        else:
            features_used = []
            target_used = None
        # Adjuntar metadatos al modelo y serializar con joblib
        model.features_used = features_used
        model.target = target_used
        model.algorithm_key = alg_key
        model.task = kind
        project_dir = os.path.join(DATASETS_DIR, f"project_{project_id}")
        os.makedirs(project_dir, exist_ok=True)
        model_path = os.path.join(project_dir, "model.pkl")
        import joblib
        joblib.dump(model, model_path)
        project["model_path"] = model_path.replace("\\", "/")
        # Quitar modelo entrenado previo en JSON si existe
        project.pop("trained_model", None)
    except Exception as e:
        return jsonify({"success": False, "msg": f"No se pudo guardar el modelo entrenado: {e}"}), 500
    # Guardar información de entrenamiento en el proyecto
    if kind in {"regression", "classification"}:
        metrics["n_train"] = int(len(Xtr))
        metrics["n_test"] = int(len(Xte))
    else:
        metrics["n_train"] = int(len(X))
        metrics["n_test"] = 0
    metrics["features_used"] = features_used
    project["last_train"] = {
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm_key": alg_key,
        "target": target,
        "metrics": metrics
    }
    training_curve = _training_curve_payload(model, alg_key)
    save_json(PROJECTS_FILE, projects)
    return jsonify({
        "success": True,
        "msg": "Entrenamiento completado",
        "metrics": metrics,
        "preview": preview,
        "training_curve": training_curve
    }), 200

@app.route("/predict_model", methods=["POST"])
def predict_model():
    """
    Predice con el modelo entrenado guardado en el proyecto.
    Body JSON:
      - project_id (int)
      - user (str)
      - inputs (dict: { feature_name: value })
    """
    import numpy as np
    data = request.get_json() or {}
    project_id = data.get("project_id")
    user = data.get("user")
    inputs = data.get("inputs", {})
    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros (project_id, user)"}), 400
    project, error = find_project(project_id, user)
    if error:
        msg, code = error
        return jsonify({"success": False, "msg": msg}), code
    # Verificar existencia del archivo de modelo entrenado
    project_dir = os.path.join(DATASETS_DIR, f"project_{project_id}")
    model_path = os.path.join(project_dir, "model.pkl")
    if not os.path.exists(model_path):
        return jsonify({"success": False, "msg": "No hay un modelo entrenado guardado para este proyecto"}), 400
    # Cargar modelo entrenado desde el archivo
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return jsonify({"success": False, "msg": f"No se pudo cargar el modelo entrenado: {e}"}), 500
    # Preparar metadatos del modelo
    alg_key = getattr(model, "algorithm_key", None)
    task = getattr(model, "task", None)
    features_used = getattr(model, "features_used", None) or []
    # Validar que se proporcionen todos los features requeridos
    missing = [f for f in features_used if f not in inputs]
    if missing:
        return jsonify({"success": False, "msg": f"Faltan valores para: {', '.join(missing)}"}), 400
    try:
        xrow = []
        for f in features_used:
            val = inputs.get(f)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return jsonify({"success": False, "msg": f"El campo '{f}' está vacío"}), 400
            try:
                xrow.append(float(val))
            except Exception:
                return jsonify({"success": False, "msg": f"El campo '{f}' debe ser numérico"}), 400
        Xnew = np.array([xrow], dtype=float)
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error procesando la entrada: {e}"}), 400
    # Realizar predicción usando el modelo cargado
    try:
        if task in {"regression", "classification", "clustering"}:
            if not hasattr(model, "predict"):
                return jsonify({"success": False, "msg": "El modelo no expone método predict"}), 400
            yhat = model.predict(Xnew)
            if isinstance(yhat, (list, tuple)):
                yhat = yhat[0]
            else:
                try:
                    yhat = np.array(yhat).reshape(-1)[0]
                except Exception:
                    pass
            resp = {"success": True, "task": task, "prediction": _to_jsonable(yhat)}
            if hasattr(model, "predict_proba") and task == "classification":
                try:
                    proba = model.predict_proba(Xnew)
                    if isinstance(proba, np.ndarray):
                        proba = proba.reshape(-1).tolist()
                    resp["proba"] = proba
                    if hasattr(model, "classes_"):
                        resp["classes"] = _to_jsonable(getattr(model, "classes_"))
                except Exception:
                    pass
            return jsonify(resp), 200
        elif task == "dimensionality_reduction":
            if hasattr(model, "transform"):
                Z = model.transform(Xnew)
            elif hasattr(model, "fit_transform"):
                return jsonify({"success": False, "msg": "El modelo PCA no tiene método transform disponible"}), 400
            else:
                return jsonify({"success": False, "msg": "Modelo de reducción no soporta transform"}), 400
            import numpy as _np
            prev_m = min(5, Z.shape[1]) if len(Z.shape) == 2 else 1
            comps = [float(Z[0, j]) for j in range(prev_m)] if len(Z.shape) == 2 else [float(Z)]
            return jsonify({"success": True, "task": task, "components": comps}), 200
        else:
            return jsonify({"success": False, "msg": f"Tarea no soportada: {task}"}), 400
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error durante la predicción: {e}"}), 400



# Inicializacion del servidor
if __name__ == "__main__":
    app.run(debug=True)
