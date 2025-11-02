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
        // muestra los proyectos del usuario
        loadProjects(user.email);
    }
});

// obtiene el formulario de creación de proyectos y el cuerpo de la tabla donde se mostrarán los proyectos
const projectForm = document.getElementById("projectForm");
const projectsTableBody = document.getElementById("projectsTableBody");

// ------------------- CREAR PROYECTO -------------------
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
async function viewProject(project) {
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

    await loadAndShowProjectDataset(project);
}

// Carga el dataset del proyecto activo (si existe) y lo pinta en la vista previa.
// Prioriza el dataset limpio (si ya pasaste por preprocesamiento), de lo contrario usa el dataset original.
// Si no hay dataset, deja la sección limpia.
async function loadAndShowProjectDataset(project) {
    const user = JSON.parse(localStorage.getItem("user"));
    if (!user) return;

    // Si el proyecto ya trae ruta de dataset guardada, intentamos leerlo
    const hasDatasetPath = !!project.dataset_path;
    const datasetInfo = document.getElementById("datasetInfo");
    const datasetPreviewHead = document.getElementById("datasetPreviewHead");
    const datasetPreviewBody = document.getElementById("datasetPreviewBody");

    // Limpia la UI antes de pintar
    if (datasetPreviewHead) datasetPreviewHead.innerHTML = "";
    if (datasetPreviewBody) datasetPreviewBody.innerHTML = "";
    if (datasetInfo) datasetInfo.classList.add("hidden");

    if (!hasDatasetPath) {
        // No hay dataset asociado aún → no mostramos nada
        console.log("Este proyecto no tiene dataset cargado todavía.");
        return;
    }

    try {
        // El backend ya expone /get_clean_dataset y devuelve primeras 5 filas + columns
        const resp = await fetch(`/get_clean_dataset?project_id=${project.id}&user=${encodeURIComponent(user.email)}`);
        const data = await resp.json();

        if (data && data.success) {
            const fileName = project.dataset_name || (project.dataset_path || "").split("/").pop();
            setDatasetNameLabel(fileName);
            renderDatasetPreview(data.columns, data.preview);
        } else {
            // Si por alguna razón no hay limpio/preview, avisamos suave
            console.warn("No se pudo obtener vista previa del dataset:", data?.msg || "Respuesta inválida");
        }
    } catch (err) {
        console.error("Error cargando el dataset del proyecto:", err);
    }
}


// ------------------- FUNCIÓN PARA EDITAR PROYECTO INLINE -------------------
function enableEditMode(event, project) {
    // guarda la fila del proyecto que se va a editar (donde hizo clic)
    const row = event.target.closest("tr");

    // si se está editando no hace nada, si no, vuelve el campo editable
    if (row.classList.contains("editing")) return;
    row.classList.add("editing");

    // guarda los elementos del nombre y la descripción del proyecto
    const nameCell = row.children[0];
    const descCell = row.children[1];
    // guarda el elemento del boton 'Editar'
    const editBtn = row.querySelector(".edit-btn");

    // guarda el nombre antes de que el usuario lo edite
    const oldName = nameCell.textContent.trim();
    // si la descripción es '-' (valor por defecto) se asigna texto vacío para la edición
    const oldDesc = descCell.textContent === "-" ? "" : descCell.textContent.trim();

    // reemplaza el contenido de los campos visibles por campos editables
    nameCell.innerHTML = `<input type="text" class="edit-name" value="${oldName}" 
        style="width:95%; padding:6px; border:1px solid #ccc; border-radius:6px;">`;
    descCell.innerHTML = `<input type="text" class="edit-desc" value="${oldDesc}" 
        style="width:95%; padding:6px; border:1px solid #ccc; border-radius:6px;">`;

    // edita el nombre del botón 'Editar'
    editBtn.textContent = "Guardar";

    // crea el botón 'Cancelar'
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancelar";
    cancelBtn.classList.add("cancel-btn");
    // lo inserta justo después de 'Editar' ('Guardar' en este caso)
    editBtn.after(cancelBtn);

    // limpia eventos antiguos para evitar acumulación
    editBtn.onclick = null;
    cancelBtn.onclick = null;

    // evento para el botón 'Guardar'
    editBtn.onclick = async () => {
        // guarda el nombre y la descripción nueva
        const newName = row.querySelector(".edit-name").value.trim();
        const newDesc = row.querySelector(".edit-desc").value.trim();
        const user = JSON.parse(localStorage.getItem("user"));

        // si no hay cambios entre el nombre nuevo y el antiguo, 'Guardar' actúa como 'Cancelar'
        if (newName === oldName && (newDesc === oldDesc || newDesc === "-")) {
            cancelBtn.click();
            return;
        // si en efecto hay cambios, oculta las propiedades del proyecto
        } else{
            document.getElementById("upload-dataset").classList.add("hidden");
            document.getElementById("select-model").classList.add("hidden");
            document.getElementById("train-model").classList.add("hidden");
            document.getElementById("results").classList.add("hidden");
        }
        // si no puso nombre muestra advertencia
        if (!newName) {
            alert("El nombre del proyecto no puede estar vacío");
            return;
        }

        // guarda la respuesta del servidor a la petición (fecth) POST de crear proyecto
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
            // respuesta del servidor en formato json
            const data = await resp.json();

            // si la respuesta fue exitosa
            if (data.success) {
                alert("Proyecto actualizado correctamente");
                // actualiza la información del proyecto 
                project.name = newName;
                project.description = newDesc;
                nameCell.textContent = newName;
                descCell.textContent = newDesc || "-";
                const activeProject = JSON.parse(localStorage.getItem("activeProject"));
                if (activeProject && activeProject.id === project.id) {
                    activeProject.name = newName;
                    activeProject.description = newDesc;
                    localStorage.setItem("activeProject", JSON.stringify(activeProject));
                }
            } else {
                // restaura los nombres si la petición fue denegada
                alert(data.msg);
                nameCell.textContent = oldName;
                descCell.textContent = oldDesc || "-";
            }
        } catch (err) {
            // si hubo un error devuelve los cambios
            console.error("Error al editar proyecto:", err);
            alert("Error al guardar cambios del proyecto");
            nameCell.textContent = oldName;
            descCell.textContent = oldDesc || "-";
        } finally {
            // restaurar botones y deshabilita la opción de editar
            row.classList.remove("editing");
            editBtn.textContent = "Editar";
            cancelBtn.remove();
            //reinicia el bucle escuchando el botón 'Editar'
            editBtn.onclick = (e) => enableEditMode(e, project);
        }
    };

    // botón 'cancelar' durante la edición del proyecto: deja todo como estaba
    cancelBtn.onclick = () => {
        nameCell.textContent = oldName;
        descCell.textContent = oldDesc || "-";
        row.classList.remove("editing");
        editBtn.textContent = "Editar";
        cancelBtn.remove();
        editBtn.onclick = (e) => enableEditMode(e, project);
    };
}



// ------------------- FUNCIÓN 'MADRE' PARA MOSTRAR PROYECTO EN MIS PROYECTOS Y HABILITAR SUS FUNCIONALIDADES -------------------
function addProjectToTable(project) {
    // crea una fila para insertar en la tabla de proyectos
    const row = document.createElement("tr");
    // crea los campos con información del proyecto y botones 'Ver', 'Editar' y 'Eliminar'
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
    // agrega la fila creada a la tabla
    projectsTableBody.appendChild(row);

    // función que se ejecutará al presionar el botón 'Ver' en los proyectos agregados
    row.querySelector(".view-btn").addEventListener("click", () => {
        viewProject(project);
    });
    // función que se ejecutará al presionar el botón 'Editar' en los proyectos agregados
    row.querySelector(".edit-btn").addEventListener("click", (e) => {
        enableEditMode(e, project);
    });

    // función que se ejecutará al presionar el botón 'Eliminar' en los proyectos agregados
    row.querySelector(".delete-btn").addEventListener("click", async () => {
        // confirmación de eliminación
        const confirmed = confirm(`¿Seguro que deseas eliminar "${project.name}"?`);
        if (!confirmed) return;

        // obtiene el usuario de la sesión
        const user = JSON.parse(localStorage.getItem("user"));

        try {
            // guarda la respuesta del servidor a la petición (fecth) DELETE de eliminar un proyecto
            const resp = await fetch("http://127.0.0.1:5000/delete_project", {
                method: "DELETE",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ id: project.id, user: user.email })
            });

            // guarda la respuesta en formato json
            const data = await resp.json();


            if (data.success) {
                alert("Proyecto eliminado correctamente");
                row.remove(); // quitarlo de la tabla sin recargar
                // oculta las propiedades del proyecto
                document.getElementById("upload-dataset").classList.add("hidden");
                document.getElementById("select-model").classList.add("hidden");
                document.getElementById("train-model").classList.add("hidden");
                document.getElementById("results").classList.add("hidden");
            } else {
                alert(data.msg);
            }
        } catch (err) {
            // si hubo un error lanza advertencia
            console.error("Error al eliminar proyecto:", err);
            alert("Error al intentar eliminar el proyecto");
        }
    });
}


// ------------------- FUNCIÓN PARA MOSTRAR LOS PROYECTOS DEL USUARIO -------------------
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

// ------------------- SUBIR DATASET -------------------
const datasetForm = document.getElementById("datasetForm");
const datasetFileInput = document.getElementById("datasetFile");
const datasetInfo = document.getElementById("datasetInfo");
const datasetPreviewHead = document.getElementById("datasetPreviewHead");
const datasetPreviewBody = document.getElementById("datasetPreviewBody");

// función al presionar el botón subir dataset
datasetForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    // si no hay archivo subido
    const file = datasetFileInput.files[0];
    if (!file) {
        alert("Por favor selecciona un archivo CSV.");
        return;
    }
    // lee el proyecto y el usuario actual de la sesión
    const activeProject = JSON.parse(localStorage.getItem("activeProject"));
    const user = JSON.parse(localStorage.getItem("user"));

    if (!activeProject) {
        alert("Debes tener un proyecto activo antes de subir un dataset.");
        return;
    }

    // guarda un formData (formato de archivos de pares clave/valor, sirve para almacenar archivos binarios) con la información que se va a mandar al backend
    const formData = new FormData();
    formData.append("project_id", activeProject.id);
    formData.append("user", user.email);
    formData.append("file", file);

    try {
        const resp = await fetch("http://127.0.0.1:5000/upload_dataset", {
            method: "POST",
            body: formData
        });

        const data = await resp.json();

        if (data.success) {
            alert(data.msg);

            // lee la información de la ruta del proyecto
            activeProject.dataset_path = data.project.dataset_path;
            // sobreescribe la información del proyecto en el navegador
            localStorage.setItem("activeProject", JSON.stringify(activeProject));

            // Mostrar vista previa
            renderDatasetPreview(data.columns, data.preview);

            activeProject.dataset_path = data.project.dataset_path;
            // 🆕 nombre legible
            activeProject.dataset_name = (data.project.dataset_path || "").split("/").pop();
            setDatasetNameLabel(activeProject.dataset_name);
            localStorage.setItem("activeProject", JSON.stringify(activeProject));
        } else {
            alert(data.msg || "Error al subir dataset");
        }
    } catch (err) {
        console.error("Error al subir dataset:", err);
        alert("Error de conexión con el servidor.");
    }
});

// FUNCION PARA CAMBIAR EL NOMBRE DEL DATASET
function setDatasetNameLabel(filename) {
  const container = document.getElementById("datasetInfo");
  if (!container) return;

  // Usa el primer h3 existente dentro de #datasetInfo (el que ya dice "Vista previa del dataset")
  let title = container.querySelector("h3");
  if (!title) {
    title = document.createElement("h3");
    container.prepend(title);
  }
  const safeName = filename || "(sin nombre)";
  title.innerHTML = `Vista previa del dataset: <span id="datasetName">${safeName}</span>`;
}



// ------------------- FUNCIÓN PARA MOSTRAR VISTA PREVIA -------------------
function renderDatasetPreview(columns, rows) {
    datasetPreviewHead.innerHTML = "";
    datasetPreviewBody.innerHTML = "";

    if (!columns || columns.length === 0) {
        datasetInfo.classList.add("hidden");
        return;
    }

    // Crear encabezados
    const headRow = document.createElement("tr");
    columns.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col;
        headRow.appendChild(th);
    });
    datasetPreviewHead.appendChild(headRow);

    // Crear filas de datos
    rows.forEach(row => {
        const tr = document.createElement("tr");
        columns.forEach(col => {
            const td = document.createElement("td");
            td.textContent = row[col];
            tr.appendChild(td);
        });
        datasetPreviewBody.appendChild(tr);
    });

    datasetInfo.classList.remove("hidden");

    const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
    const fileName = activeProject?.dataset_name || (activeProject?.dataset_path || "").split("/").pop();
    setDatasetNameLabel(fileName);
}

// ------------------- PREPROCESAR DATASET -------------------
const preprocessBtn = document.createElement("button");
preprocessBtn.textContent = "Preprocesar datos";
preprocessBtn.id = "preprocessBtn";
preprocessBtn.style.marginTop = "10px";
document.querySelector("#upload-dataset").appendChild(preprocessBtn);

const preprocessResultDiv = document.createElement("div");
preprocessResultDiv.id = "preprocessResult";
preprocessResultDiv.classList.add("hidden");
document.querySelector("#upload-dataset").appendChild(preprocessResultDiv);

preprocessBtn.addEventListener("click", async () => {
    const activeProject = JSON.parse(localStorage.getItem("activeProject"));
    const user = JSON.parse(localStorage.getItem("user"));

    if (!activeProject || !activeProject.dataset_path) {
        alert("Primero debes subir un dataset antes de preprocesar los datos.");
        return;
    }

    if (!confirm("¿Deseas limpiar y preparar el dataset para entrenamiento?")) return;

    try {
        const resp = await fetch("http://127.0.0.1:5000/preprocess_data", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                project_id: activeProject.id,
                user: user.email
            })
        });

        const data = await resp.json();

        if (data.success) {
            alert(data.msg);
            renderPreprocessSummary(data.summary, data.numeric_cols, data.categorical_cols);
            // Actualizar dataset activo
            if (data.clean_path) {
                activeProject.dataset_path = data.clean_path;
                // 🆕 refresca nombre por si cambió la ruta
                activeProject.dataset_name = (data.clean_path || "").split("/").pop();
                setDatasetNameLabel(activeProject.dataset_name);
                localStorage.setItem("activeProject", JSON.stringify(activeProject));
            }

            // Si el backend devuelve filas previsualizables (opcional)
            if (data.preview && data.columns) {
                renderDatasetPreview(data.columns, data.preview);
            } else {
                // Refrescar vista con el dataset limpio
                const resp2 = await fetch(`/static/${activeProject.dataset_path}`);
            }
            await refreshCleanPreview();
        } else {
            alert(data.msg || "Error en el preprocesamiento.");
        }
    } catch (err) {
        console.error("Error al preprocesar datos:", err);
        alert("Error de conexión con el servidor.");
    }
});

// ------------------- UI: Sugerir modelo -------------------
const selectModelSection = document.getElementById("select-model");

// Botón "Sugerir modelo"
const recommendBtn = document.createElement("button");
recommendBtn.id = "btnRecommend";
recommendBtn.textContent = "Sugerir modelo";
recommendBtn.style.marginTop = "10px";
selectModelSection.appendChild(recommendBtn);

// Contenedor para mostrar la recomendación
const modelRecDiv = document.createElement("div");
modelRecDiv.id = "model-recommendations";
modelRecDiv.style.marginTop = "10px";
selectModelSection.appendChild(modelRecDiv);

// Click → recomendar
recommendBtn.addEventListener("click", recommendModel);

// ------------------- LÓGICA: recomendar modelo -------------------
async function recommendModel() {
    const activeProject = JSON.parse(localStorage.getItem("activeProject"));
    const user = JSON.parse(localStorage.getItem("user"));

    if (!activeProject) {
        alert("Selecciona un proyecto primero.");
        return;
    }

    // Intentar inferir un target por defecto a partir de la vista previa
    const cols = Array.from(datasetPreviewHead.querySelectorAll("th")).map(th => th.textContent);
    let defaultTarget = cols.includes("Y") ? "Y" : (cols.length ? cols[cols.length - 1] : "");

    // Pedir la columna objetivo (deja vacío si no hay objetivo → no supervisado)
    let target = prompt("Columna objetivo (por ejemplo, Y). Déjalo vacío si no hay:", defaultTarget);
    if (target !== null) target = target.trim();
    if (target === "") target = null;

    try {
        const resp = await fetch("/recommend_model", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                project_id: activeProject.id,
                user: user.email,
                target
            })
        });
        const data = await resp.json();

        if (!data.success) {
            alert(data.msg || "No se pudo generar una recomendación.");
            return;
        }

        const recsHtml = (data.recommendations || [])
            .map(r => `<li><b>${r.model}</b> — score: ${r.score}<br><small>${r.why}</small></li>`)
            .join("");

        modelRecDiv.innerHTML = `
            <h3>Recomendación de modelo</h3>
            <p><b>Tarea:</b> ${data.task || "unsupervised"}${data.target ? ` — <b>Objetivo:</b> ${data.target}` : ""}</p>
            <p><b>Features numéricas:</b> ${data.numeric_features?.join(", ") || "Ninguna"}</p>
            <p><b>Features categóricas:</b> ${data.categorical_features?.join(", ") || "Ninguna"}</p>
            <ul>${recsHtml}</ul>
        `;
    } catch (err) {
        console.error("Error al recomendar modelo:", err);
        alert("Error al conectar con el servidor.");
    }
}


// ------------------- MOSTRAR RESULTADO DEL PREPROCESAMIENTO -------------------
function renderPreprocessSummary(summary, numericCols, categoricalCols) {
    preprocessResultDiv.innerHTML = `
        <h3>Resumen del preprocesamiento</h3>
        <ul>
            <li><b>Filas antes:</b> ${summary.filas_antes}</li>
            <li><b>Filas después:</b> ${summary.filas_despues}</li>
            <li><b>Columnas antes:</b> ${summary.columnas_antes}</li>
            <li><b>Columnas después:</b> ${summary.columnas_despues}</li>
            <li><b>Duplicados eliminados:</b> ${summary.duplicados_eliminados}</li>
            <li><b>Columnas eliminadas:</b> ${summary.columnas_eliminadas.join(", ") || "Ninguna"}</li>
        </ul>
        <p><b>Columnas numéricas:</b> ${numericCols.join(", ") || "Ninguna"}</p>
        <p><b>Columnas categóricas:</b> ${categoricalCols.join(", ") || "Ninguna"}</p>
    `;
    preprocessResultDiv.classList.remove("hidden");
}

// Refrescar vista previa del dataset limpio
async function refreshCleanPreview() {
    const activeProject = JSON.parse(localStorage.getItem("activeProject"));
    const user = JSON.parse(localStorage.getItem("user"));

    try {
        const resp = await fetch(`/get_clean_dataset?project_id=${activeProject.id}&user=${encodeURIComponent(user.email)}`);
        const data = await resp.json();

        if (data.success) {
            renderDatasetPreview(data.columns, data.preview);
        } else {
            console.warn("No se pudo refrescar la vista del dataset limpio:", data.msg);
        }
    } catch (err) {
        console.error("Error al refrescar dataset limpio:", err);
    }
}

// ======== Seleccionar modelo: usar TU form y TU select existentes ========

// Mapea los valores de tu <option value="..."> a la clave interna y categoría
const MODEL_MAP = {
  linear:       { key: "linear_regression",   category: "regresion" },
  logistic:     { key: "logistic_regression", category: "clasificacion" },
  perceptron:   { key: "perceptron",          category: "clasificacion" },
  decisiontree: { key: "decision_tree",       category: "clasificacion" },
  naivebayes:   { key: "naive_bayes",         category: "clasificacion" },
  mlp:          { key: "mlp",                 category: "clasificacion" },
  pca:          { key: "pca",                 category: "reduccion_dimensionalidad" },
  kmeans:       { key: "kmeans",              category: "clustering" },
};

// Toma tus elementos reales existentes
const modelForm = document.getElementById("modelForm");
const modelSelect = document.getElementById("modelSelect");
const modelStatus = document.getElementById("model-recommendations"); // lo usamos como “status”

async function saveModelSelection() {
  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  if (!activeProject?.id) return alert("Selecciona un proyecto primero.");
  if (!user?.email) return alert("No hay sesión activa.");

  const uiVal = modelSelect?.value;
  if (!uiVal) return alert("Elige un modelo.");

  const map = MODEL_MAP[uiVal];
  if (!map) return alert("Modelo no reconocido.");

  // Si en tu HTML agregas inputs de hiperparámetros, puedes leerlos aquí.
  const params = {}; // por ahora dejamos defaults del backend

  try {
    const resp = await fetch("/select_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: activeProject.id,
        user: user.email,
        category: map.category,
        algorithm_key: map.key,
        params
      })
    });
    const data = await resp.json();

    if (data.success) {
      if (modelStatus) {
        modelStatus.innerHTML = `<span style="color:green">✔ Modelo guardado: <b>${uiVal}</b> → <code>${map.key}</code></span>`;
      }
      // habilitar entrenar si tienes botón en #train-model
      const trainBtn = document.getElementById("trainButton") || document.querySelector("#train-model button");
      if (trainBtn) trainBtn.disabled = false;

      if (data.project) {
        localStorage.setItem("activeProject", JSON.stringify(data.project));
      }
    } else {
      if (modelStatus) {
        modelStatus.innerHTML = `<span style="color:#b00">✖ ${data.msg || "No se pudo guardar la selección"}</span>`;
      } else {
        alert(data.msg || "No se pudo guardar la selección");
      }
    }
  } catch (err) {
    console.error("Error guardando modelo:", err);
    if (modelStatus) modelStatus.innerHTML = `<span style="color:#b00">✖ Error de conexión</span>`;
    else alert("Error de conexión");
  }
}

// Evita recarga del form y usa nuestra función
if (modelForm) {
  modelForm.addEventListener("submit", (e) => {
    e.preventDefault();
    e.stopPropagation();
    saveModelSelection();
  });

  // Por si el botón del form dispara submit por defecto, reforzamos:
  const submitBtn = modelForm.querySelector('button[type="submit"]');
  if (submitBtn) submitBtn.setAttribute("type", "button");
  if (submitBtn) {
    submitBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      saveModelSelection();
    }, { capture: true });
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
