// Validar sesión al cargar el dashboard
window.addEventListener("DOMContentLoaded", () => {
    const user = JSON.parse(localStorage.getItem("user"));

    if (!user) {
        // Si no hay sesión → regresar al login
        alert("Debes iniciar sesión primero.");
        window.location.href = "index.html";
    } else {
        console.log("Sesión activa:", user);
        // Aquí puedes personalizar el dashboard, ej. mostrar nombre del usuario
        const header = document.querySelector("header h1");
        header.textContent = `Aivana - Bienvenido, ${user.name}`;
        loadProjects(user.email);
    }
});

// ------------------- CREAR PROYECTO -------------------
const projectForm = document.getElementById("projectForm");
const projectsTableBody = document.getElementById("projectsTableBody");

projectForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const name = document.getElementById("projectName").value;
    const description = document.getElementById("projectDesc").value;
    const user = JSON.parse(localStorage.getItem("user"));

    if (!user) {
        alert("No hay sesión activa.");
        return;
    }

    try {
        const resp = await fetch("http://127.0.0.1:5000/create_project", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, description, user: user.email })
        });

        const data = await resp.json();

        if (data.success) {
            alert("Proyecto creado con éxito");
            addProjectToTable(data.project);
            projectForm.reset();
        } else {
            alert(data.msg);
        }
    } catch (err) {
        console.error("Error al crear proyecto:", err);
    }
});

// ------------------- FUNCIONES AUXILIARES -------------------
// ------------------- ACCIÓN CON EL BOTÓN VER EN MIS PROYECTOS -------------------
function viewProject(project) {
    // Guardar el proyecto activo
    localStorage.setItem("activeProject", JSON.stringify(project));

    // Mostrar secciones bloqueadas
    document.getElementById("upload-dataset").classList.remove("hidden");
    document.getElementById("select-model").classList.remove("hidden");
    document.getElementById("train-model").classList.remove("hidden");
    document.getElementById("results").classList.remove("hidden");

    // Personalizar título con el nombre del proyecto
    const title = document.querySelector("#upload-dataset h2");
    if (title) title.textContent = `Cargar dataset para: ${project.name}`;
}

// ------------------- FUNCIÓN PARA MOSTRAR PROYECTO EN MIS PROYECTOS -------------------
function addProjectToTable(project) {
    const row = document.createElement("tr");

    row.innerHTML = `
        <td>${project.name}</td>
        <td>${project.description || "-"}</td>
        <td>${project.created_at}</td>
        <td>
            <button class="view-btn">Ver</button>
            <button class="delete-btn">Eliminar</button>
        </td>
    `;

    // Botón Ver
    row.querySelector(".view-btn").addEventListener("click", () => {
        viewProject(project);
    });

    projectsTableBody.appendChild(row);
}

async function loadProjects(userEmail) {
    try {
        const resp = await fetch(`http://127.0.0.1:5000/get_projects?user=${encodeURIComponent(userEmail)}`);
        const data = await resp.json();

        if (data.success) {
            projectsTableBody.innerHTML = ""; // limpiar
            data.projects.forEach(addProjectToTable);
        }
    } catch (err) {
        console.error("Error al cargar proyectos:", err);
    }
}

// ------------------- CERRAR SESIÓN -------------------
const logoutLink = document.querySelector("a[href='index.html']");
if (logoutLink) {
    logoutLink.addEventListener("click", (e) => {
        localStorage.removeItem("user");
    });
}
