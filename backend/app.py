# Cambios a implementar:
# 1. Verificar que name, email, password existan y no vengan vacíos en register. (pendiente)
# 2. Validar formato de email básico en register. (pendiente)
# 3. Encriptar claves (from werkzeug.security import generate_password_hash   hashed = generate_password_hash(password)). (pendiente)
# 4. Implementar un token de sesión (pendiente)
# 5. Validar que solo exista una sesión (ventana) activa por usuario
# 6.
# 7.
# 8. Dejarlo local???


# jsonify construye las respuestas de API, serializando los diccionarios, configurando cabeceras y creando las respuestas, las llaves que se usan son a conveniencia. 
# 200 (OK), 201 (Created: cuando se crea un nuevo recurso), 204 (No content: no hay contenido para devolver)
# 400 (Bad Request: datos inválidos), 401 (No autenticado), 403 (Forbidden: prohibido), 404 (Not found), 500(Internal Server Error)
from flask import Flask, request, jsonify, send_from_directory
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import sys

# Inicialización del servidor con ruta de archivos estáticos
app = Flask(__name__, static_folder="../frontend", static_url_path="/")

# Cuando el usuario abre la app (dominio raíz /), lo redirige a index.html
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Directorio base '/backend'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Soportar dos escenarios:
# 1) app.py en la raíz (algorithms está en BASE_DIR)
# 2) app.py dentro de backend/ (algorithms está en el padre)
CANDIDATES = [BASE_DIR, os.path.dirname(BASE_DIR)]
for p in CANDIDATES:
    if p and p not in sys.path:
        sys.path.insert(0, p)

# ahora los imports directos del paquete algorithms funcionan
from algorithms.LinearRegression import LinearRegression
from algorithms.LogisticRegression import LogisticRegression
from algorithms.Perceptron import Perceptron
from algorithms.DecisionTreeClassifier import DecisionTreeClassifier
from algorithms.NaiveBayes import NaiveBayes
from algorithms.MLP import MLPClassifier
from algorithms.KMeans import KMeans
from algorithms.PCA import PCA

# ------------------- CONFIG JSON'S -------------------
# rutas con los json
USERS_FILE = os.path.join(BASE_DIR, "users.json")
PROJECTS_FILE = os.path.join(BASE_DIR, "projects.json")
# ruta de los datasets subidos en los proyectos
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
# crea la ruta si no existe
os.makedirs(DATASETS_DIR, exist_ok=True)

def load_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            # si no existe el json, lo crea con → []
            json.dump([], f, indent=4)
    try:
        with open(path, "r") as f:
            # devuelve el archivo recién creado (o el ya existente si lo había)
            return json.load(f)
    # archivo corrupto o vacío
    except json.JSONDecodeError:
        return []

# guarda los json en formato python
users = load_json(USERS_FILE)
projects = load_json(PROJECTS_FILE)

# ------------------- RUTAS -------------------
# ------------------- Registro -------------------
# función que se ejecutará al recibir una petición POST (crear un usuario) en /register
@app.route("/register", methods=["POST"]) # esta linea establece la dirección /register como una ruta válida para enviar peticiones desde js
def register():
    # request procesa la solicitud POST, extrae el JSON del cuerpo y deserializa los datos.
    data = request.get_json()
    # se extraen los campos del json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    # recorre el json de usuarios verifica si el correo que se está registrando ya existe
    # any lee la secuencia generada de True/False, si encuentra al menos un True, se detiene y devuelve True
    if any(u["email"] == email for u in users):
        return jsonify({"success": False, "msg": "El correo ya está registrado"}), 400

    # crea el diccionario con los datos del nuevo usuario y lo agrega al json
    new_user = {"name": name, "email": email, "password": password}
    users.append(new_user)

    # guarda el archivo json con los datos actualizados
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

    # confirma el registro
    return jsonify({"success": True, "msg": "Registro exitoso"}), 201

# ------------------- Iniciar Sesión -------------------
# función que se ejecutará al recibir una petición POST (loggearse) en /login
@app.route("/login", methods=["POST"])
def login():
    # request procesa la solicitud POST, extrae el JSON del cuerpo y deserializa los datos.
    data = request.get_json()
    # se extraen los campos del json
    email = data.get("email")
    password = data.get("password")

    # recorre el json de usuarios y busca coincidencias de correo y contraseña
    # si hay coincidencia, user guarda el diccionario del usuario coincidente y detiene la busqueda. En caso de no encontrarlo, user queda None
    user = next((u for u in users if u["email"] == email and u["password"] == password), None)

    # autoriza o desautoriza el ingreso
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
# función que se ejecutará al recibir una petición POST (crear un proyecto) en /create_project xd
@app.route("/create_project", methods=["POST"])
def create_project():
    # request procesa la solicitud POST, extrae el JSON del cuerpo y deserializa los datos.
    data = request.get_json()
    # se extraen los campos del json
    name = data.get("name")
    # vacio si no se especificó descripción
    desc = data.get("description", "")
    user = data.get("user")  # correo del usuario que creó el proyecto

    # valida que se defina un nombre para el proyecto
    if not name:
        return jsonify({"success": False, "msg": "El nombre del proyecto es obligatorio"}), 400

    # crea el objeto del proyecto
    project = {
        "id": len(projects) + 1,
        "name": name,
        "description": desc,
        "user": user,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # agrega el proyecto creado al json
    projects.append(project)

    # guarda el archivo json con los datos actualizados
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    # confirma la creación del proyecto
    return jsonify({"success": True, "msg": "Proyecto creado con éxito", "project": project}), 201

# ------------------- Cargar Proyectos -------------------
# función que se ejecutará al recibir una petición GET (cargar proyectos) en /get_projects xd
@app.route("/get_projects", methods=["GET"])
def get_projects():
    # lee los parámetros de la URL, filtrando solo los proyectos del usuario con el correo (/get_projects?user=juandaz2004@gmail.com)
    user = request.args.get("user")
    if user:
        # almacena todos los proyectos registrados con el correo del usuario en una lista
        user_projects = [p for p in projects if p["user"] == user]
        # valida la solicitud y envía los proyectos del usuario
        return jsonify({"success": True, "projects": user_projects}), 200
    # si no se mandó un user, devuelve todos los proyectos (USO DE ADMIN SOLAMENTE)
    return jsonify({"success": False, "msg": "No hay un usuario asociado"}), 200

# ------------------- Editar Proyecto -------------------
@app.route("/edit_project", methods=["PUT"])
def edit_project():
    # request procesa la solicitud PUT, extrae el JSON del cuerpo y deserializa los datos.
    data = request.get_json()
    # se extraen los campos del json
    project_id = data.get("id")
    new_name = data.get("name")
    new_desc = data.get("description", "")
    user = data.get("user")

    if not project_id:
        return jsonify({"success": False, "msg": "Falta el ID del proyecto"}), 400
    if not new_name:
        return jsonify({"success": False, "msg": "El nombre del proyecto no puede estar vacío"}), 400

    # recorre el json de proyectos y busca coincidencias de id y usuario (correo)
    # si hay coincidencia, project guarda el diccionario del proyecyo coincidente y detiene la busqueda. En caso de no encontrarlo, muestra un error
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado o no pertenece al usuario"}), 404

    # validar nombre repetido de proyecto para el mismo usuario
    if any(p["name"] == new_name and p["user"] == user and p["id"] != project_id for p in projects):
        return jsonify({"success": False, "msg": "Ya existe un proyecto con ese nombre"}), 400

    # actualizar el diccionario con los datos nuevos
    project["name"] = new_name
    project["description"] = new_desc

    # guarda los cambios en el json
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    return jsonify({"success": True, "msg": "Proyecto actualizado correctamente", "project": project}), 200

# ------------------- Eliminar Proyecto -------------------
@app.route("/delete_project", methods=["DELETE"])
def delete_project():
    # request procesa la solicitud DELETE, extrae el JSON del cuerpo y deserializa los datos.
    data = request.get_json()
    # se extraen los campos del json
    project_id = data.get("id")
    user = data.get("user")

    if not project_id:
        return jsonify({"success": False, "msg": "Falta el ID del proyecto"}), 400

    # recorre el json de proyectos y busca coincidencias de id y usuario (correo)
    # si hay coincidencia, project guarda el diccionario del proyecyo coincidente y detiene la busqueda. En caso de no encontrarlo, muestra un error
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado o no pertenece al usuario"}), 404

    # elimina el proyecto del json
    projects.remove(project)

    # sobreescribe los cambios en el json
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    return jsonify({"success": True, "msg": "Proyecto eliminado correctamente"}), 200

# ------------------- Subir Dataset -------------------
@app.route("/upload_dataset", methods=["POST"])
def upload_dataset():
    # obtiene el id del proyecto, el usuario y el archivo del formData de la petición POST
    project_id = request.form.get("project_id", type=int)
    user = request.form.get("user")
    #archivos binarios van al objeto request.files
    file = request.files.get("file")

    # si no existe el archivo o el proyecto
    if not project_id or not file:
        return jsonify({"success": False, "msg": "Faltan datos o archivo"}), 400

    # busca el proyecto con id y asociado al usuario
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    # crea la carpeta para el proyecto
    project_dir = os.path.join(DATASETS_DIR, f"project_{project_id}")
    os.makedirs(project_dir, exist_ok=True)

    # guarda el archivo subido
    filename = file.filename
    file_path = os.path.join(project_dir, filename)
    file.save(file_path)

    # guarda la ruta en el json con un nueva llave
    project["dataset_path"] = file_path.replace("\\", "/")

    # sobreescribe los cambios en el json
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    # lee primeras filas para previsualización con pandas
    try:
        # guarda el dataset en un objeto DataFrame
        df = pd.read_csv(file_path)
        # guarda las primeras 5 filas y las convierte en una lista de diccionarios python donde cada fila representa un diccionario y cada nombre de las columans es una llave
        preview = df.head(5).to_dict(orient="records")
        # crea una lista que contiene solo los nombres de las columnas
        columns = list(df.columns)

        # convertir NaN (dataset) a None para que el JSON sea válido
        preview = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in preview]

    except Exception as e:
        return jsonify({"success": False, "msg": f"Error leyendo CSV: {e}"}), 400

    # devuelve el proyecto, la información de la previsualización y los nombres de las columnas
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

    # obtiene el id del proyecto y el usuario de la petición POST
    data = request.get_json()
    project_id = data.get("project_id")
    user = data.get("user")

    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros"}), 400

    # busca el proyecto
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    # obtiene la ruta de donde esta guardado el dataset subido
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "No se encontró el dataset para este proyecto"}), 404

    try:
        # lee el csv
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error leyendo dataset: {e}"}), 400

    original_shape = df.shape

    # --- LIMPIEZA AUTOMÁTICA FINAL ---
    # 1) Eliminar columnas completamente vacías
    df.dropna(axis=1, how="all", inplace=True)

    # 2) Detectar columnas numéricas de forma robusta:
    #    - Intentamos convertir a numérico con errors='coerce'
    #    - Calculamos el % de valores convertibles (no NaN tras la conversión)
    #    - Si ese % >= 0.6 (60%), consideramos la columna como numérica
    numeric_cols = []
    for col in df.columns:
        # serie convertida a numeric (no modifica df todavía)
        conv = pd.to_numeric(df[col], errors='coerce')
        # proporción de valores convertibles respecto a valores no vacíos originales
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

    # 9) Actualizar referencia (por si la usas en el front)
    project["clean_dataset_path"] = clean_path.replace("\\", "/")
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

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
# este método se usa únicamente para mostrar la previsualización del dataset cargado al proyecto al abrir el dashboard
@app.route("/get_clean_dataset", methods=["GET"])
def get_clean_dataset():

    # lee proyecto y usuario de la solicicud GET
    project_id = request.args.get("project_id", type=int)
    user = request.args.get("user")

    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros"}), 400

    # busca el proyecto
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    # carga la ruta del dataset
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado"}), 404
    
    # muestra la vista previa del dataset existente
    try:
        df = pd.read_csv(dataset_path)
        preview = df.head(5).to_dict(orient="records")
        columns = list(df.columns)
        # convertir NaN a None
        preview = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in preview]
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error leyendo dataset: {e}"}), 400

    return jsonify({
        "success": True,
        "columns": columns,
        "preview": preview
    }), 200

# ------------------- Recomendar Modelo -------------------
@app.route("/recommend_model", methods=["POST"])
def recommend_model():
    # carga el proyecto, el usuario y el target
    data = request.get_json()
    project_id = data.get("project_id")
    user = data.get("user")
    target = data.get("target")  # nombre de la columna objetivo (opcional)
    
    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros"}), 400

    # buscar proyecto
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    # busca el dataset
    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado para este proyecto"}), 404

    # carga el dataset
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error leyendo dataset: {e}"}), 400

    n, p_total = df.shape
    # Detectar columnas numéricas / no numéricas
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Si no envían target y hay una columna llamada Y, se usa; si no, sin target (no supervisado)
    if not target:
        if "Y" in df.columns:
            target = "Y"
        else:
            # No supervisado: se recomienda PCA/KMeans según p
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

    task = "regression" if pd.api.types.is_numeric_dtype(y) else "classification"

    recommendations = []

    if task == "regression":
        # Heurística simple de linealidad: correlación absoluta media entre y y features numéricas
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

        # Si hay indicio de no linealidad (p muchas, corr baja) → árbol / MLP
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
        imbalance = float(counts.max()) if not counts.empty else 1.0  # 1.0 = muy desbalanceado

        # Binaria → LogisticRegression como baseline
        if k == 2:
            base = 0.75 - 0.2 * (imbalance - 0.5)  # un poco menos si hay desbalance fuerte
            recommendations.append({
                "model": "LogisticRegression",
                "score": round(max(0.1, min(1.0, base)), 3),
                "why": f"Clasificación binaria; baseline rápido y explicable. Desbalance≈{imbalance:.2f}."
            })
            # Si hay no linealidad (muchas numéricas o sospecha): árbol y MLP
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
            # Multiclase → Árbol como baseline, Naive Bayes si hay texto o muchas categoricas
            recommendations.append({
                "model": "DecisionTreeClassifier",
                "score": 0.65,
                "why": f"Multiclase ({k} clases); árboles son versátiles e interpretables."
            })
            if len(X_cat) > len(X_num):
                recommendations.append({
                    "model": "NaiveBayes",
                    "score": 0.6,
                    "why": "Muchas variables categóricas/bolsa de palabras; NB es simple y efectivo."
                })
            else:
                recommendations.append({
                    "model": "MLP",
                    "score": 0.55,
                    "why": "Puede modelar fronteras complejas en multiclase si hay datos."
                })

    # Si no hay target o no se generaron recomendaciones por algún motivo:
    if not recommendations:
        recommendations = [
            {"model": "PCA", "score": 0.6, "why": "No se pudo inferir tarea claramente; explora reducción de dimensión."},
            {"model": "KMeans", "score": 0.5, "why": "Prueba agrupamiento no supervisado."}
        ]

    # Ordenar por score desc y dejar top-3
    recommendations = sorted(recommendations, key=lambda r: r["score"], reverse=True)[:3]

    return jsonify({
        "success": True,
        "task": task,
        "target": target,
        "n_rows": int(n),
        "n_features": int(len(X.columns)),
        "numeric_features": X_num,
        "categorical_features": X_cat,
        "recommendations": recommendations
    }), 200

# ------------------- Seleccionar Modelo -------------------
# parametros por defecto de los modelos
DEFAULT_PARAMS = {
    "linear_regression":   {"learning_rate": 0.001, "n_iterations": 20000, "fit_intercept": True},
    "logistic_regression": {"learning_rate": 0.1,  "n_iterations": 2000, "fit_intercept": True, "decision_threshold": 0.5},
    "perceptron":          {"learning_rate": 1.0,  "n_iterations": 1000, "fit_intercept": True},
    "decision_tree":       {"criterion": "gini",   "max_depth": None, "min_samples_split": 2},
    "naive_bayes":         {"nb_type": "gaussian", "alpha": 1.0},
    "mlp":                 {"hidden_layers": [32], "activation": "relu", "learning_rate": 0.01, "n_iterations": 200},
    "kmeans":              {"n_clusters": 3, "max_iter": 300, "tol": 1e-4},
    "pca":                 {"n_components": None, "whiten": False}
}

@app.route("/select_model", methods=["POST"])
def api_models_select():

    data = request.get_json() or {}
    project_id = data.get("project_id")
    user = data.get("user")
    category = data.get("category")        # usa los valores tal como vienen del HTML
    algorithm_key = data.get("algorithm_key")
    params = data.get("params") or {}

    if not project_id or not user or not category or not algorithm_key:
        return jsonify({"success": False, "msg": "Faltan parámetros (project_id, user, category, algorithm_key)"}), 400

    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    # defaults + override del front (si no hay defaults, usa solo los params que mandes)
    merged_params = {**DEFAULT_PARAMS.get(algorithm_key, {}), **params}

    project["model_cfg"] = {
        "category": category,
        "algorithm_key": algorithm_key,
        "params": merged_params
    }

    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    return jsonify({
        "success": True,
        "msg": "Modelo seleccionado y guardado",
        "model_cfg": project["model_cfg"],
        "project": project
    }), 200

# ------------------- Entrenar Modelo -------------------
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
    return alg_key in {"linear_regression","logistic_regression","perceptron","decision_tree","naive_bayes","mlp"}

def _task_kind(alg_key):
    if alg_key == "linear_regression":
        return "regression"
    if alg_key in {"logistic_regression","perceptron","decision_tree","naive_bayes","mlp"}:
        return "classification"
    if alg_key == "kmeans":
        return "clustering"
    if alg_key == "pca":
        return "dimensionality_reduction"
    return "unknown"

@app.route("/train_model", methods=["POST"])
def train_model():

    data = request.get_json() or {}
    project_id = data.get("project_id")
    user = data.get("user")
    target = data.get("target")            # requerido para modelos supervisados
    test_size = float(data.get("test_size", 0.2))
    random_state = int(data.get("random_state", 42))

    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros (project_id, user)"}), 400

    # localizar proyecto
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    if "model_cfg" not in project:
        return jsonify({"success": False, "msg": "No hay modelo seleccionado aún"}), 400

    cfg = project["model_cfg"]
    alg_key = cfg.get("algorithm_key")
    params = cfg.get("params", {})
    kind = _task_kind(alg_key)

    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado para este proyecto"}), 404

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error leyendo dataset: {e}"}), 400

    # Seleccionar columnas numéricas para X
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    model = None
    try:
        if alg_key == "linear_regression":
            model = LinearRegression(**params)
        elif alg_key == "logistic_regression":
            model = LogisticRegression(**params)
        elif alg_key == "perceptron":
            model = Perceptron(**params)
        elif alg_key == "decision_tree":
            model = DecisionTreeClassifier(**params)
        elif alg_key == "naive_bayes":
            model = NaiveBayes(**params)
        elif alg_key == "mlp":
            model = MLPClassifier(**params)
        elif alg_key == "kmeans":
            model = KMeans(**params)
        elif alg_key == "pca":
            model = PCA(**params)
        else:
            return jsonify({"success": False, "msg": f"Algoritmo no soportado: {alg_key}"}), 400
    except Exception as e:
        return jsonify({"success": False, "msg": f"Error inicializando el modelo: {e}"}), 400

    metrics = {}
    preview = {}

    try:
        if kind in {"regression","classification"}:
            if not target or target not in df.columns:
                return jsonify({"success": False, "msg": "Debes indicar la columna objetivo (target) válida"}), 400

            # y y X
            y = df[target]
            # asegurarnos de no meter target en X
            feature_candidates = [c for c in num_cols if c != target]
            if not feature_candidates:
                return jsonify({"success": False, "msg": "No hay columnas numéricas para entrenar (X)"}), 400
            X = df[feature_candidates]

            # codificar y si no es numérica en clasificación
            y_encoded = y.copy()
            label_map = None
            if kind == "classification" and not pd.api.types.is_numeric_dtype(y):
                classes = sorted(y.astype(str).unique().tolist())
                label_map = {c:i for i,c in enumerate(classes)}
                y_encoded = y.astype(str).map(label_map)

            # split
            Xtr, Xte, ytr, yte = _train_test_split(X, y_encoded, test_size=test_size, seed=random_state)

            # fit
            model.fit(Xtr.values, ytr.values if ytr is not None else None)

            # predict y métricas
            # intentamos predict; si no existe, para algunas implementaciones de regresión lineal puede ser predict(X)
            yhat = None
            if hasattr(model, "predict"):
                yhat = model.predict(Xte.values)
            else:
                # fallback mínimo
                return jsonify({"success": False, "msg": "El modelo no expone método predict"}), 400

            yhat = np.array(yhat).reshape(-1)

            if kind == "regression":
                # métricas regresión
                y_true = yte.values.astype(float)
                mse = float(np.mean((yhat - y_true)**2))
                mae = float(np.mean(np.abs(yhat - y_true)))
                # R2
                ss_res = float(np.sum((y_true - yhat)**2))
                ss_tot = float(np.sum((y_true - np.mean(y_true))**2)) or 1.0
                r2 = 1.0 - ss_res/ss_tot
                metrics = {"task":"regression","mse":mse,"mae":mae,"r2":r2}
                if alg_key == "linear_regression" and hasattr(model, "weights"):
                    w = np.asarray(model.weights, dtype=float).ravel().tolist()
                    if getattr(model, "fit_intercept", True):
                        intercept = float(w[0]); coefs = [float(c) for c in w[1:]]
                    else:
                        intercept = 0.0;        coefs = [float(c) for c in w]
                    # mapea coeficientes a nombres de columnas
                    metrics["intercept"] = intercept
                    metrics["coefficients"] = dict(zip(feature_candidates, coefs))
                    # pista de normalización (por transparencia)
                    if hasattr(model, "normalize"):
                        metrics["normalized_inputs"] = bool(model.normalize)

            else:
                # métricas clasificación (accuracy + matriz de confusión)
                y_true = yte.values.astype(int)
                y_pred = yhat.astype(int)
                acc = float(np.mean(y_true == y_pred))
                # matriz de confusión compacta
                labels = np.unique(np.concatenate([y_true, y_pred]))
                cm = {int(lbl): {int(lbl2): int(np.sum((y_true==lbl)&(y_pred==lbl2))) for lbl2 in labels} for lbl in labels}
                # si hubo label_map, incluirlo
                if label_map:
                    metrics["label_map"] = label_map
                metrics.update({"task":"classification","accuracy":acc,"confusion_matrix":cm})

            # preview de primeras filas de test
            prev_k = min(10, len(Xte))
            preview = {
                "columns": ["y_true","y_pred"] + feature_candidates[:10],
                "rows": [
                    [ (None if yte is None else (int(yte.iloc[i]) if kind=="classification" else float(yte.iloc[i]))),
                      (int(yhat[i]) if kind=="classification" else float(yhat[i])) ]
                    + [ (None if pd.isna(Xte.iloc[i,j]) else float(Xte.iloc[i,j])) for j in range(min(10, Xte.shape[1])) ]
                    for i in range(prev_k)
                ]
            }

        elif kind == "clustering":
            # solo X numéricas
            if not num_cols:
                return jsonify({"success": False, "msg": "No hay columnas numéricas para K-Means"}), 400
            X = df[num_cols]
            model.fit(X.values)
            if hasattr(model, "predict"):
                labels = model.predict(X.values)
            else:
                # algunos KMeans guardan labels_ tras fit
                labels = getattr(model, "labels_", None)
            labels = np.array(labels) if labels is not None else np.zeros(len(X), dtype=int)
            # métrica mínima (inercia si existe)
            inertia = getattr(model, "inertia_", None)
            metrics = {"task":"clustering","n_samples": int(len(X))}
            if inertia is not None:
                metrics["inertia"] = float(inertia)
            # preview
            prev_k = min(10, len(X))
            preview = {
                "columns": ["cluster"] + num_cols[:10],
                "rows": [
                    [int(labels[i])] + [ (None if pd.isna(X.iloc[i,j]) else float(X.iloc[i,j])) for j in range(min(10, X.shape[1])) ]
                    for i in range(prev_k)
                ]
            }

        elif kind == "dimensionality_reduction":
            if not num_cols:
                return jsonify({"success": False, "msg": "No hay columnas numéricas para PCA"}), 400
            X = df[num_cols]
            # intentamos fit_transform si existe
            if hasattr(model, "fit_transform"):
                Z = model.fit_transform(X.values)
            else:
                model.fit(X.values)
                Z = model.transform(X.values) if hasattr(model, "transform") else X.values
            # var explicada si la hay
            explained = getattr(model, "explained_variance_ratio_", None)
            if explained is not None:
                explained = [float(v) for v in np.array(explained).ravel().tolist()]
            metrics = {
                "task":"dimensionality_reduction",
                "explained_variance_ratio": explained
            }
            prev_k = min(10, Z.shape[0])
            prev_m = min(5, Z.shape[1])
            preview = {
                "columns": [f"PC{i+1}" for i in range(prev_m)],
                "rows": [
                    [ float(Z[i,j]) for j in range(prev_m) ]
                    for i in range(prev_k)
                ]
            }

        else:
            return jsonify({"success": False, "msg": f"Tarea no soportada para {alg_key}"}), 400

    except Exception as e:
        return jsonify({"success": False, "msg": f"Error durante el entrenamiento: {e}"}), 400
    
    # info de sustentación
    metrics["n_train"] = int(len(Xtr))
    metrics["n_test"]  = int(len(Xte))
    metrics["features_used"] = feature_candidates


    # guarda un pequeño rastro en el proyecto (sin serializar el modelo)
    project["last_train"] = {
        "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm_key": alg_key,
        "target": target,
        "metrics": metrics
    }
    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    return jsonify({
        "success": True,
        "msg": "Entrenamiento completado",
        "metrics": metrics,
        "preview": preview
    }), 200


# inicialización del servidor
if __name__ == "__main__":
    app.run(debug=True)
