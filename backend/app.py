from flask import Flask, request, jsonify, send_from_directory
import json, os
from datetime import datetime

app = Flask(__name__, static_folder="../frontend", static_url_path="/")

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# ------------------- CONFIG USERS -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")

if os.path.exists(USERS_FILE):
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
    except json.JSONDecodeError:
        # Archivo vacío o corrupto → inicializamos vacío
        users = []
else:
    users = []

# ------------------- CONFIG PROJECTS -------------------
PROJECTS_FILE = os.path.join(BASE_DIR, "projects.json")

# Inicializar archivo si no existe
if not os.path.exists(PROJECTS_FILE):
    with open(PROJECTS_FILE, "w") as f:
        json.dump([], f)

with open(PROJECTS_FILE, "r") as f:
    try:
        projects = json.load(f)
    except json.JSONDecodeError:
        projects = []

# ------------------- RUTAS -------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if any(u["email"] == email for u in users):
        return jsonify({"success": False, "msg": "El correo ya está registrado"}), 400

    new_user = {"name": name, "email": email, "password": password}
    users.append(new_user)

    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

    return jsonify({"success": True, "msg": "Registro exitoso"})


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
        })
    else:
        return jsonify({"success": False, "msg": "Credenciales inválidas"}), 401
    
# ------------------- RUTAS PARA PROYECTOS -------------------
@app.route("/create_project", methods=["POST"])
def create_project():
    data = request.get_json()
    name = data.get("name")
    desc = data.get("description", "")
    user = data.get("user")  # correo del usuario que creó el proyecto

    if not name:
        return jsonify({"success": False, "msg": "El nombre del proyecto es obligatorio"}), 400

    # Crear objeto de proyecto
    project = {
        "id": len(projects) + 1,
        "name": name,
        "description": desc,
        "user": user,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    projects.append(project)

    with open(PROJECTS_FILE, "w") as f:
        json.dump(projects, f, indent=4)

    return jsonify({"success": True, "project": project})


@app.route("/get_projects", methods=["GET"])
def get_projects():
    # Opcional: filtrar por usuario
    user = request.args.get("user")
    if user:
        user_projects = [p for p in projects if p["user"] == user]
        return jsonify({"success": True, "projects": user_projects})
    return jsonify({"success": True, "projects": projects})



if __name__ == "__main__":
    app.run(debug=True)
