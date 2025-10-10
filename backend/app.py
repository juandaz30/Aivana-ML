# Cambios a implementar:
# 1. Verificar que name, email, password existan y no vengan vacíos en register. (pendiente)
# 2. Validar formato de email básico en register. (pendiente)
# 3. Encriptar claves (from werkzeug.security import generate_password_hash   hashed = generate_password_hash(password)). (pendiente)
# 4. Implementar un token de sesión (pendiente)
# 5.
# 6.
# 7.
# 8. Migración a BD (SQLite).


# jsonify construye las respuestas de API, serializando los diccionarios, configurando cabeceras y creando las respuestas, las llaves que se usan son a conveniencia. 
# 200 (OK), 201 (Created: cuando se crea un nuevo recurso), 204 (No content: no hay contenido para devolver)
# 400 (Bad Request: datos inválidos), 401 (No autenticado), 403 (Forbidden: prohibido), 404 (Not found), 500(Internal Server Error)
from flask import Flask, request, jsonify, send_from_directory
import json, os
from datetime import datetime

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

# ------------------- CONFIGURACIÓN DE RUTAS -------------------
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
    
# ------------------- RUTAS PARA PROYECTOS -------------------
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


# inicialización del servidor
if __name__ == "__main__":
    app.run(debug=True)
