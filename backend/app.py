from flask import Flask, request, jsonify, send_from_directory
import json, os

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



if __name__ == "__main__":
    app.run(debug=True)
