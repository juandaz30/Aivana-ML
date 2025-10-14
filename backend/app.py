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
import json, os
from datetime import datetime
import pandas as pd

# Inicialización del servidor con ruta de archivos estáticos
app = Flask(__name__, static_folder="../frontend", static_url_path="/")

# Cuando el usuario abre la app (dominio raíz /), lo redirige a index.html
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Directorio base '/backend'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    project_id = request.form.get("project_id", type=int)
    user = request.form.get("user")
    file = request.files.get("file")

    if not project_id or not file:
        return jsonify({"success": False, "msg": "Faltan datos o archivo"}), 400

    # Buscar proyecto
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    # Crear carpeta para el proyecto
    project_dir = os.path.join(DATASETS_DIR, f"project_{project_id}")
    os.makedirs(project_dir, exist_ok=True)

    # Guardar archivo
    filename = file.filename
    file_path = os.path.join(project_dir, filename)
    file.save(file_path)

    # Guardar la ruta del dataset en el proyecto
    project["dataset_path"] = file_path.replace("\\", "/")

    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    # Leer primeras filas para previsualización
    try:
        df = pd.read_csv(file_path)
        preview = df.head(5).to_dict(orient="records")
        columns = list(df.columns)

        # ✅ convertir NaN a None para que el JSON sea válido
        preview = [{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in preview]

    except Exception as e:
        return jsonify({"success": False, "msg": f"Error leyendo CSV: {e}"}), 400

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
    import pandas as pd
    import numpy as np

    data = request.get_json()
    project_id = data.get("project_id")
    user = data.get("user")

    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros"}), 400

    # Buscar proyecto
    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "No se encontró el dataset para este proyecto"}), 404

    try:
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
@app.route("/get_clean_dataset", methods=["GET"])
def get_clean_dataset():
    import pandas as pd

    project_id = request.args.get("project_id", type=int)
    user = request.args.get("user")

    if not project_id or not user:
        return jsonify({"success": False, "msg": "Faltan parámetros"}), 400

    project = next((p for p in projects if p["id"] == project_id and p["user"] == user), None)
    if not project:
        return jsonify({"success": False, "msg": "Proyecto no encontrado"}), 404

    dataset_path = project.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"success": False, "msg": "Dataset no encontrado"}), 404

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



# inicialización del servidor
if __name__ == "__main__":
    app.run(debug=True)
