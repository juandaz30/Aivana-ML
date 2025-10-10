// espera a que todos los elementos del DOM hayan cargado antes de ejecutar
window.addEventListener("DOMContentLoaded", () => {
    // intenta buscar el elemento user creado en el login
    const user = JSON.parse(localStorage.getItem("user"));

    if (!user) {
        // Si no hay sesión (no hay usuario guardado el el LocalStore) → regresar al login
        alert("Debes iniciar sesión primero.");
        window.location.href = "index.html";
    } else {
        console.log("Sesión activa:", user);
        // busca el elemento h1 con sintaxis de selector CSS ("header h1")
        const header = document.querySelector("header h1");
        // lo ajusta con el nombre del usuario
        header.textContent = `Aivana - Bienvenido, ${user.name}`;
        loadProjects(user.email);
    }
});

// ------------------- CREAR PROYECTO -------------------
// obtiene el formulario de creación de proyectos y el cuerpo de la tabla donde se mostrarán los proyectos
const projectForm = document.getElementById("projectForm");
const projectsTableBody = document.getElementById("projectsTableBody");

// función que se ejecutará al presionar el botón 'Crear Proyecto'
projectForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    // obtiene los valores de los campos en el formulario
    const name = document.getElementById("projectName").value;
    const description = document.getElementById("projectDesc").value;
    const user = JSON.parse(localStorage.getItem("user"));
    // segunda verificación de sesión activa (por si acaso)
    if (!user) {
        alert("No hay sesión activa.");
        return;
    }

    // guarda la respuesta del servidor a la petición (fecth) POST de crear proyecto
    try {
        const resp = await fetch("http://127.0.0.1:5000/create_project", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, description, user: user.email })
        });
        // respuesta del servidor en formato json
        const data = await resp.json();

        // lee la respuesta del servidor a la petición
        if (data.success) {
            addProjectToTable(data.project);
            // resetea los campos del formulario
            projectForm.reset();
        }
        alert(data.msg);
    } catch (err) {
        console.error("Error al crear proyecto:", err);
    }
});

// ------------------- FUNCIONES AUXILIARES -------------------
// ------------------- ACCIÓN CON EL BOTÓN VER EN MIS PROYECTOS -------------------
function viewProject(project) {
    // guarda el proyecto activo el el localStorage del navegador
    localStorage.setItem("activeProject", JSON.stringify(project));

    // muestra las secciones bloqueadas (limitadas antes de crear un proyecto)
    document.getElementById("upload-dataset").classList.remove("hidden");
    document.getElementById("select-model").classList.remove("hidden");
    document.getElementById("train-model").classList.remove("hidden");
    document.getElementById("results").classList.remove("hidden");

    // personaliza el título (h2) de la sección 'cargar dataset' con el nombre del proyecto
    const title = document.querySelector("#upload-dataset h2");
    if (title) title.textContent = `Cargar dataset para: ${project.name}`;
}

// ------------------- FUNCIÓN PARA EDITAR PROYECTO INLINE -------------------
function enableEditMode(event, project) {
    const row = event.target.closest("tr");

    if (row.classList.contains("editing")) return;
    row.classList.add("editing");

    const nameCell = row.children[0];
    const descCell = row.children[1];
    const editBtn = row.querySelector(".edit-btn");

    const oldName = nameCell.textContent.trim();
    const oldDesc = descCell.textContent === "-" ? "" : descCell.textContent.trim();

    // Inputs visualmente armonizados
    nameCell.innerHTML = `<input type="text" class="edit-name" value="${oldName}" 
        style="width:95%; padding:6px; border:1px solid #ccc; border-radius:6px;">`;
    descCell.innerHTML = `<input type="text" class="edit-desc" value="${oldDesc}" 
        style="width:95%; padding:6px; border:1px solid #ccc; border-radius:6px;">`;
    
    editBtn.textContent = "Guardar";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancelar";
    cancelBtn.classList.add("cancel-btn");
    editBtn.after(cancelBtn);

    // Limpia eventos antiguos
    editBtn.onclick = null;
    cancelBtn.onclick = null;

    // Guardar cambios
    editBtn.onclick = async () => {
        const newName = row.querySelector(".edit-name").value.trim();
        const newDesc = row.querySelector(".edit-desc").value.trim();
        const user = JSON.parse(localStorage.getItem("user"));

        //  Si no hay cambios, actúa como cancelar
        if (newName === oldName && newDesc === oldDesc) {
            cancelBtn.click();
            return;
        } else{
            document.getElementById("upload-dataset").classList.add("hidden");
            document.getElementById("select-model").classList.add("hidden");
            document.getElementById("train-model").classList.add("hidden");
            document.getElementById("results").classList.add("hidden");
        }

        if (!newName) {
            alert("El nombre del proyecto no puede estar vacío");
            return;
        }

        try {
            const resp = await fetch("http://127.0.0.1:5000/edit_project", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    id: project.id,
                    name: newName,
                    description: newDesc,
                    user: user.email
                })
            });

            const data = await resp.json();

            if (data.success) {
                alert("Proyecto actualizado correctamente");
                project.name = newName;
                project.description = newDesc;
                nameCell.textContent = newName;
                descCell.textContent = newDesc || "-";
            } else {
                alert(data.msg);
                nameCell.textContent = oldName;
                descCell.textContent = oldDesc || "-";
            }
        } catch (err) {
            console.error("Error al editar proyecto:", err);
            alert("Error al guardar cambios del proyecto");
            nameCell.textContent = oldName;
            descCell.textContent = oldDesc || "-";
        } finally {
            // Restaurar botones y estado
            row.classList.remove("editing");
            editBtn.textContent = "Editar";
            cancelBtn.remove();
            editBtn.onclick = (e) => enableEditMode(e, project);
        }
    };

    // Cancelar edición
    cancelBtn.onclick = () => {
        nameCell.textContent = oldName;
        descCell.textContent = oldDesc || "-";
        row.classList.remove("editing");
        editBtn.textContent = "Editar";
        cancelBtn.remove();
        editBtn.onclick = (e) => enableEditMode(e, project);
    };
}



// ------------------- FUNCIÓN PARA MOSTRAR PROYECTO EN MIS PROYECTOS -------------------
function addProjectToTable(project) {
    // crea una fila en la tabla de proyectos
    const row = document.createElement("tr");
    // crea los campos con información y botones 'Ver', 'Editar' y 'Eliminar'
    row.innerHTML = `
        <td>${project.name}</td>
        <td>${project.description || "-"}</td>
        <td>${project.created_at}</td>
        <td>
            <button class="view-btn">Ver</button>
            <button class="edit-btn">Editar</button>
            <button class="delete-btn">Eliminar</button>
        </td>
    `;

    // función que se ejecutará al presionar el botón 'Ver' en los proyectos agregados
    row.querySelector(".view-btn").addEventListener("click", () => {
        viewProject(project);
    });
    // función que se ejecutará al presionar el botón 'Editar' en los proyectos agregados
    row.querySelector(".edit-btn").addEventListener("click", (e) => {
        enableEditMode(e, project);
    });
    // agrega la fila creada a la tabla
    projectsTableBody.appendChild(row);


    // Botón Eliminar
    row.querySelector(".delete-btn").addEventListener("click", async () => {
        const confirmed = confirm(`¿Seguro que deseas eliminar "${project.name}"?`);
        if (!confirmed) return;

        const user = JSON.parse(localStorage.getItem("user"));

        try {
            const resp = await fetch("http://127.0.0.1:5000/delete_project", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: project.id, user: user.email })
            });

            const data = await resp.json();

            if (data.success) {
                alert("Proyecto eliminado correctamente");
                row.remove(); // quitarlo de la tabla sin recargar
                document.getElementById("upload-dataset").classList.add("hidden");
                document.getElementById("select-model").classList.add("hidden");
                document.getElementById("train-model").classList.add("hidden");
                document.getElementById("results").classList.add("hidden");
            } else {
                alert(data.msg);
            }
        } catch (err) {
            console.error("Error al eliminar proyecto:", err);
            alert("Error al intentar eliminar el proyecto");
        }
    });
}

async function loadProjects(userEmail) {
    try {
        // guarda la respuesta del servidor a la petición (fecth) GET de obtener los proyectos
        const resp = await fetch(`http://127.0.0.1:5000/get_projects?user=${encodeURIComponent(userEmail)}`); // lo que va después de ? es información adicionar para filtrar, ordenar o modificar la solicitud.
        // respuesta del servidor en formato json
        const data = await resp.json();

        
        if (data.success) {
            // limpia los proyectos que haya en la tabla (evita duplicado)
            projectsTableBody.innerHTML = ""; 
            // agrega una fila por cada proyecto encontrado asociado al usuario
            data.projects.forEach(addProjectToTable);
        } else {
            alert(data.msg);
        }

    } catch (err) {
        console.error("Error al cargar proyectos:", err);
    }
}

// ------------------- CERRAR SESIÓN -------------------
// elemento <a> de cerrar sesión en el <nav>
const logoutLink = document.querySelector("a[href='index.html']");
if (logoutLink) {
    // al cerrar sesión, redirige a index.html (esto lo hace el HTML) y borra el usuario el localStorage
    logoutLink.addEventListener("click", (e) => {
        localStorage.removeItem("user");
    });
}
