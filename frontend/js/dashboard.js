// dashboard.js — versión compacta, comentada y lista para sobrescribir

"use strict";

/* ========= Helpers de DOM y fetch ========= */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const apiFetch = (url, options = {}) =>
  fetch(url, options).then(async (r) => {
    let data = null;
    try { data = await r.json(); } catch (_) { /* sin cuerpo JSON */ }
    return data ?? { success: false, msg: "Respuesta inválida del servidor" };
  });

/* ========= Referencias globales ========= */
// Header / sesión
const headerTitle = $("header h1");
const logoutLink = $("a[href='index.html']");

// Secciones
const sectionCreate = $("#create-project");
const sectionProjects = $("#projects-list");
const sectionUpload = $("#upload-dataset");
const sectionSelect = $("#select-model");
const sectionTrain  = $("#train-model");
const sectionResults = $("#results");
const sectionPredict = $("#predict");

// Formularios / Controles
const projectForm = $("#projectForm");
const projectsTableBody = $("#projectsTableBody");

// Dataset
const datasetForm        = $("#datasetForm");
const datasetFileInput   = $("#datasetFile");
const datasetInfo        = $("#datasetInfo");
const datasetPreviewHead = $("#datasetPreviewHead");
const datasetPreviewBody = $("#datasetPreviewBody");

// Preprocesar
let preprocessBtn = $("#preprocessBtn");        // puede no existir en HTML; lo creamos si falta
let preprocessResultDiv = $("#preprocessResult"); // idem

// Seleccionar modelo
const modelForm = $("#modelForm");
const modelSelect = $("#modelSelect");
const modelRecDiv = $("#model-recommendations");
let recommendBtn = $("#btnRecommend"); // lo creamos si no existe

// Entrenar
const trainButton = $("#trainButton");
const trainingStatus = $("#trainingStatus");

// Resultados
const metricsBox = $("#metrics");
const chartsBox = $("#charts");
let trainingChartInstance = null;

// Predecir (nueva sección)
const predictInfo   = $("#predictInfo");
const predictForm   = $("#predictForm");
const predictInputs = $("#predictInputs");
const predictResult = $("#predictResult");

// ---- Parámetros del modelo (UI) ----
const paramsPanel   = document.querySelector("#model-params-panel");
const paramsFormEl  = document.querySelector("#paramsForm");
const modelSelectEl = document.querySelector("#modelSelect");
const saveParamsBtn = document.querySelector("#saveParamsBtn");
const selectModelWithParamsBtn = document.querySelector("#selectModelWithParamsBtn");

let DEFAULTS_CACHE = null;

// Mapeo UI -> backend (usa el mismo que el submit, pero global)
const ALG_MAP = {
  "linear":       "linear_regression",
  "logistic":     "logistic_regression",
  "perceptron":   "perceptron",
  "decisiontree": "decision_tree",
  "naivebayes":   "naive_bayes",
  "mlp":          "mlp",
  "pca":          "pca",
  "kmeans":       "kmeans"
};

// Enums para desplegar <select> en vez de inputs de texto
const PARAM_ENUMS = {
  activation: ["relu", "tanh", "sigmoid"],
  criterion: ["gini", "entropy"],
  nb_type: ["gaussian", "multinomial", "bernoulli"],
  init: ["k-means++", "random"]
};

const PARAM_RANGES = {
  learning_rate: "1e-4 a 1 (típico 0.001-0.1)",
  n_iterations: "100 a 5000",
  fit_intercept: "true/false",
  early_stopping: "true/false",
  tolerance: "1e-9 a 1e-2",
  patience: "3 a 50",
  normalize: "true/false",
  max_grad_norm: "1 a 1e6",
  verbose: "true/false",
  l2: "0 a 10",
  decision_threshold: "0 a 1",
  clip: "1e-20 a 1e-5",
  shuffle: "true/false",
  random_state: "Entero, ej. 0-10_000",
  callbacks: "Lista vacía o funciones personalizadas",
  criterion: "gini / entropy",
  max_depth: "1-50 o vacío para sin límite",
  min_samples_split: "2-50",
  min_samples_leaf: "1-20",
  max_features: "1 al total de columnas o vacío",
  nb_type: "gaussian / multinomial / bernoulli",
  var_smoothing: "1e-12 a 1e-6",
  alpha: "0 a 5",
  class_priors: "Lista de probabilidades que sumen 1",
  binarize: "0 a 1",
  hidden_layers: "Listas cortas, ej. [16], [64,32]",
  activation: "relu / tanh / sigmoid",
  batch_size: "8 a 512",
  validation_split: "0 a 0.3",
  n_clusters: "2 a 20",
  init: "k-means++ o random",
  n_init: "1 a 50",
  max_iter: "50 a 1000",
  tol: "1e-6 a 1e-2",
  n_components: "1 al número de columnas",
  whiten: "true/false",
  copy: "true/false"
};

const PARAM_TOOLTIPS = {
  learning_rate: "Qué tanto cambia el modelo cada vez que aprende. Valores altos aprenden rápido pero pueden fallar; valores bajos son más estables pero lentos.",
  n_iterations: "Cantidad de veces que el modelo repasa los datos para aprender.",
  fit_intercept: "Si está activo, el modelo calcula primero un punto de partida base antes de ajustar los datos.",
  early_stopping: "Detiene el entrenamiento cuando deja de mejorar para ahorrar tiempo y evitar errores.",
  tolerance: "Mejora mínima necesaria para considerar que el modelo sigue aprendiendo.",
  patience: "Intentos sin mejora que se permiten antes de parar cuando early_stopping está activo.",
  normalize: "Ajusta todas las columnas para que tengan escalas parecidas antes de entrenar.",
  max_grad_norm: "Límite máximo para los cambios internos en cada paso. Mantenerlo bajo evita saltos bruscos.",
  verbose: "Muestra mensajes detallados durante el entrenamiento para ver qué está ocurriendo.",
  l2: "Penalización que mantiene los valores del modelo más pequeños para evitar que se pase de optimista.",
  decision_threshold: "Número entre 0 y 1 que define a partir de qué probabilidad se elige la clase positiva.",
  clip: "Valor mínimo usado para evitar divisiones por cero o resultados infinitos durante el cálculo.",
  shuffle: "Mezcla las filas del dataset en cada pasada para que el modelo no aprenda un orden fijo.",
  random_state: "Número fijo para repetir exactamente los mismos resultados en futuros entrenamientos.",
  callbacks: "Acciones personalizadas que se ejecutan en momentos clave del entrenamiento. Déjalo vacío si no necesitas nada extra.",
  criterion: "Regla que indica cómo decide el árbol la mejor pregunta en cada rama.",
  max_depth: "Número máximo de preguntas seguidas que puede hacer el árbol antes de responder.",
  min_samples_split: "Cantidad mínima de filas necesarias para que el árbol divida una rama en dos.",
  min_samples_leaf: "Cantidad mínima de filas que debe haber en cada hoja del árbol.",
  max_features: "Número máximo de columnas que el árbol revisa para buscar la mejor pregunta. Vacío significa usar todas.",
  nb_type: "Tipo de versión de Naive Bayes según tus datos: gaussiano para números, multinomial para conteos, bernoulli para sí/no.",
  var_smoothing: "Pequeño valor que se suma para evitar divisiones por cero cuando las varianzas son muy pequeñas.",
  alpha: "Cantidad de suavizado para evitar probabilidades en cero. Valores mayores generan predicciones más cautas.",
  class_priors: "Probabilidades iniciales de cada clase si quieres forzar un sesgo. Déjalo vacío para calcularlo desde los datos.",
  binarize: "Valor límite para convertir números en 0 o 1 antes de entrenar en la versión Bernoulli.",
  hidden_layers: "Lista con la cantidad de neuronas en cada capa intermedia de la red. Ejemplo: [64, 32].",
  activation: "Función que determina cómo responde cada neurona. Cambiarla puede hacer que la red aprenda patrones distintos.",
  batch_size: "Número de filas que se usan juntas antes de ajustar el modelo. Valores pequeños hacen ajustes frecuentes; los grandes son más estables.",
  validation_split: "Porción del dataset reservada para medir cómo va el entrenamiento sin tocar esos datos.",
  n_clusters: "Cantidad de grupos que K-Means intentará formar.",
  init: "Forma inicial de ubicar los centros de los grupos antes de empezar a ajustar.",
  n_init: "Veces que K-Means se reinicia desde posiciones distintas para quedarse con el mejor resultado.",
  max_iter: "Límite máximo de ciclos de ajuste que puede hacer el modelo.",
  tol: "Mejora mínima necesaria para seguir iterando. Si los cambios son menores, el proceso se detiene.",
  n_components: "Número de columnas nuevas que PCA mantendrá después de comprimir la información.",
  whiten: "Si está activo, ajusta las nuevas columnas de PCA para que tengan la misma escala.",
  copy: "Si está activo, trabaja sobre una copia del dataset y deja intactos los datos originales."
};


/* ========= Utilidades de UI ========= */
function show(el) { el?.classList.remove("hidden"); }
function hide(el) { el?.classList.add("hidden"); }
function setText(el, text) { if (el) el.textContent = text; }

/** Limpia todo lo que depende del proyecto activo */
function clearProjectDependentUI() {
  // Dataset preview
  if (datasetPreviewHead) datasetPreviewHead.innerHTML = "";
  if (datasetPreviewBody) datasetPreviewBody.innerHTML = "";
  if (datasetInfo) hide(datasetInfo);
  setDatasetNameLabel(null);

  // Recomendaciones / selección
  if (modelRecDiv) modelRecDiv.innerHTML = "";
  if (modelForm) modelForm.reset?.();
  if (modelSelect) modelSelect.value = modelSelect.value || "";

  // Entrenamiento / resultados
  if (trainingStatus) trainingStatus.innerHTML = "";
  if (metricsBox) metricsBox.innerHTML = "";
  if (chartsBox) {
    chartsBox.innerHTML = "";
    if (trainingChartInstance) {
      trainingChartInstance.destroy();
      trainingChartInstance = null;
    }
  }

  // Predecir
  if (predictInputs) predictInputs.innerHTML = "";
  if (predictResult) { predictResult.innerHTML = ""; hide(predictResult); }
  if (predictForm) hide(predictForm);
  if (predictInfo) { setText(predictInfo, "Entrena un modelo para habilitar la predicción con nuevos datos."); show(predictInfo); }
  hide(sectionPredict);

  // Cambios Hiperatributos 
  if (paramsFormEl) paramsFormEl.innerHTML = "";
  if (paramsPanel) paramsPanel.classList.add("hidden"); 
}

/** Cambia el título del bloque de preview */
function setDatasetNameLabel(name) {
  const h3 = $("#upload-dataset h3");
  if (!h3) return;
  h3.textContent = name ? `Vista previa del dataset: ${name}` : "Vista previa del dataset";
}

/** Render de la tabla de preview */
function renderDatasetPreview(columns, rows) {
  if (!columns?.length || !rows?.length) {
    hide(datasetInfo);
    return;
  }
  datasetPreviewHead.innerHTML = "";
  datasetPreviewBody.innerHTML = "";

  const headRow = document.createElement("tr");
  columns.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    headRow.appendChild(th);
  });
  datasetPreviewHead.appendChild(headRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((c) => {
      const td = document.createElement("td");
      const v = row[c];
      td.textContent = v === null || v === undefined ? "" : v;
      tr.appendChild(td);
    });
    datasetPreviewBody.appendChild(tr);
  });

  show(datasetInfo);
}

/** Resumen bonito del preprocesamiento */
function renderPreprocessSummary(summary, numericCols, categoricalCols) {
  if (!preprocessResultDiv) {
    preprocessResultDiv = document.createElement("div");
    preprocessResultDiv.id = "preprocessResult";
    sectionUpload.appendChild(preprocessResultDiv);
  }

  preprocessResultDiv.innerHTML = `
    <div class="panel">
      <div class="panel-title">Resumen del preprocesamiento</div>
      <ul>
        <li><b>Filas antes:</b> ${summary.filas_antes}</li>
        <li><b>Filas después:</b> ${summary.filas_despues}</li>
        <li><b>Columnas antes:</b> ${summary.columnas_antes}</li>
        <li><b>Columnas después:</b> ${summary.columnas_despues}</li>
        <li><b>Duplicados eliminados:</b> ${summary.duplicados_eliminados}</li>
        <li><b>Columnas eliminadas:</b> ${Array.isArray(summary.columnas_eliminadas) && summary.columnas_eliminadas.length ? summary.columnas_eliminadas.join(", ") : "Ninguna"}</li>
      </ul>
      <p><b>Columnas numéricas:</b> ${numericCols?.length ? numericCols.join(", ") : "Ninguna"}</p>
      <p><b>Columnas categóricas:</b> ${categoricalCols?.length ? categoricalCols.join(", ") : "Ninguna"}</p>
    </div>
  `;
  show(preprocessResultDiv);
}

/** Recarga preview desde el backend (dataset limpio) */
async function refreshCleanPreview() {
  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  if (!activeProject?.id || !user?.email) return;

  const data = await apiFetch(`/get_clean_dataset?project_id=${activeProject.id}&user=${encodeURIComponent(user.email)}`);
  if (data?.success) {
    renderDatasetPreview(data.columns, data.preview);
  }
}

/** Construye UI de predicción a partir de features_used */
function enablePredictUI(features, task) {
  if (!Array.isArray(features) || !features.length) {
    hide(sectionPredict);
    return;
  }

  show(sectionPredict);
  hide(predictInfo);
  show(predictForm);
  predictInputs.innerHTML = "";
  predictResult.innerHTML = "";
  hide(predictResult);

  features.forEach((f) => {
    const wrap = document.createElement("div");
    const id = `pred-${f}`;
    wrap.innerHTML = `
      <label for="${id}" style="font-weight:600">${f}</label>
      <input id="${id}" name="${f}" type="number" step="any" placeholder="Valor para ${f}">
    `;
    predictInputs.appendChild(wrap);
  });

  predictForm.onsubmit = async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
    const user = JSON.parse(localStorage.getItem("user") || "{}");
    if (!activeProject?.id || !user?.email) {
      alert("No hay proyecto o usuario activo.");
      return;
    }

    const inputs = {};
    let hasError = false;
    features.forEach((f) => {
      const el = $(`#pred-${f}`);
      const raw = el?.value?.trim?.() ?? "";
      if (raw === "" || Number.isNaN(Number(raw))) {
        el?.classList.add("input-error");
        hasError = true;
      } else {
        el?.classList.remove("input-error");
        inputs[f] = Number(raw);
      }
    });
    if (hasError) { alert("Por favor completa todos los campos con valores numéricos válidos."); return; }

    const data = await apiFetch("/predict_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: activeProject.id, user: user.email, inputs })
    });

    if (!data?.success) { alert(data.message || "No se pudo predecir."); return; }

    let html = `<div class="panel-title">Resultado de la predicción</div>`;
    if (data.task === "regression") {
      html += `<p><b>Valor predicho:</b> <span class="chip">${data.prediction}</span></p>`;
    } else if (data.task === "classification") {
      html += `<p><b>Clase predicha:</b> <span class="chip">${data.prediction}</span></p>`;
      if (data.proba && data.classes) {
        html += `<p style="margin-top:8px;"><b>Probabilidades:</b></p><ul>`;
        data.classes.forEach((c, i) => {
          const p = data.proba[i];
          html += `<li>${c}: ${(p * 100).toFixed(2)}%</li>`;
        });
        html += `</ul>`;
      }
    } else if (data.task === "clustering") {
      html += `<p><b>Cluster asignado:</b> <span class="chip">${data.prediction}</span></p>`;
    } else if (data.task === "dimensionality_reduction" && data.components) {
      html += `<p><b>Componentes:</b> ${data.components.map((v,i)=>`PC${i+1}=${Number(v).toFixed(4)}`).join(", ")}</p>`;
    } else {
      html += `<p>Predicción completada.</p>`;
    }

    predictResult.innerHTML = html;
    show(predictResult);
  };
}

/* ========= Lógica de proyectos ========= */
function addProjectToTable(project) {
  const row = document.createElement("tr");
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
  projectsTableBody.appendChild(row);

  // Ver
  row.querySelector(".view-btn").addEventListener("click", () => viewProject(project));

  // Editar inline
  row.querySelector(".edit-btn").addEventListener("click", (e) => enableEditMode(e, project, row));

  // Eliminar
  row.querySelector(".delete-btn").addEventListener("click", async () => {
    if (!confirm(`¿Seguro que deseas eliminar "${project.name}"?`)) return;
    const user = JSON.parse(localStorage.getItem("user") || "{}");
    const data = await apiFetch("/delete_project", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: project.id, user: user.email })
    });
    if (data?.success) {
      alert("Proyecto eliminado correctamente");
      row.remove();
      hide(sectionUpload); hide(sectionSelect); hide(sectionTrain); hide(sectionResults); hide(sectionPredict);
      clearProjectDependentUI();
    } else {
      alert(data?.msg || "No se pudo eliminar el proyecto");
    }
  });
}

async function loadProjects(userEmail) {
  const data = await apiFetch(`/get_projects?user=${encodeURIComponent(userEmail)}`);
  if (data?.success) {
    projectsTableBody.innerHTML = "";
    data.projects.forEach(addProjectToTable);
  } else {
    alert(data?.msg || "No se pudieron cargar los proyectos");
  }
}

function viewProject(project) {
  localStorage.setItem("activeProject", JSON.stringify(project));

  // Limpiar UI dependiente y mostrar secciones principales
  clearProjectDependentUI();
  show(sectionUpload); show(sectionSelect); show(sectionTrain); show(sectionResults);

  // Encabezado
  setText(headerTitle, `Aivana - Bienvenido, ${JSON.parse(localStorage.getItem("user")||"{}").name || ""}`);

  // Título dataset
  setDatasetNameLabel(project.dataset_name || null);

  // Si ya hay dataset, cargar preview automáticamente
  if (project.dataset_path) {
    const user = JSON.parse(localStorage.getItem("user") || "{}");
    (async () => {
      const data = await apiFetch(`/get_clean_dataset?project_id=${project.id}&user=${encodeURIComponent(user.email)}`);
      if (data?.success) {
        renderDatasetPreview(data.columns, data.preview);
        // persistir el nombre si no está
        if (!project.dataset_name) {
          const parts = (project.dataset_path || "").split("/");
          project.dataset_name = parts[parts.length - 1] || "dataset.csv";
          localStorage.setItem("activeProject", JSON.stringify(project));
          setDatasetNameLabel(project.dataset_name);
        }
      }
    })();
  }
}

function resetEditButton(editBtn, project, row) {
  const clone = editBtn.cloneNode(true);
  clone.textContent = "Editar";
  editBtn.replaceWith(clone);
  clone.addEventListener("click", (e) => enableEditMode(e, project, row));
  return clone;
}

/* ========= Editar proyecto inline ========= */
function enableEditMode(event, project, row) {
  row = row || event.target.closest("tr");
  if (row.classList.contains("editing")) return;
  row.classList.add("editing");

  const nameCell = row.children[0];
  const descCell = row.children[1];
  const editBtn = row.querySelector(".edit-btn");

  const oldName = nameCell.textContent.trim();
  const oldDesc = descCell.textContent.trim() === "-" ? "" : descCell.textContent.trim();

  nameCell.innerHTML = `<input type="text" class="edit-name" value="${oldName}" style="width:95%; padding:6px; border:1px solid #ccc; border-radius:6px;">`;
  descCell.innerHTML = `<input type="text" class="edit-desc" value="${oldDesc}" style="width:95%; padding:6px; border:1px solid #ccc; border-radius:6px;">`;

  editBtn.textContent = "Guardar";

  const cancelBtn = document.createElement("button");
  cancelBtn.textContent = "Cancelar";
  cancelBtn.classList.add("cancel-btn");
  editBtn.after(cancelBtn);

  const finalizeEdit = (newName, newDesc) => {
    nameCell.textContent = newName;
    descCell.textContent = newDesc || "-";
    row.classList.remove("editing");
    if (cancelBtn.parentNode) cancelBtn.remove();
    resetEditButton(editBtn, project, row);
  };

  editBtn.onclick = async () => {
    const newName = row.querySelector(".edit-name").value.trim();
    const newDesc = row.querySelector(".edit-desc").value.trim();
    const user = JSON.parse(localStorage.getItem("user") || "{}");

    if (!newName) { alert("El nombre del proyecto no puede estar vacío"); return; }

    // Si no hay cambios, cancelar
    if (newName === oldName && (newDesc === oldDesc || newDesc === "-")) {
      finalizeEdit(oldName, oldDesc);
      return;
    }

    // Oculta secciones si cambias el nombre (para evitar incoherencias)
    hide(sectionUpload); hide(sectionSelect); hide(sectionTrain); hide(sectionResults); hide(sectionPredict);
    clearProjectDependentUI();

    try {
      const data = await apiFetch("/edit_project", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: project.id, name: newName, description: newDesc, user: user.email })
      });

      if (data?.success) {
        alert("Proyecto actualizado correctamente");
        project.name = newName; project.description = newDesc;
        // Actualizar localStorage si este proyecto es el activo
        const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
        if (activeProject?.id === project.id) {
          activeProject.name = newName; activeProject.description = newDesc;
          localStorage.setItem("activeProject", JSON.stringify(activeProject));
        }
        finalizeEdit(newName, newDesc);
      } else {
        alert(data?.msg || "No se pudieron guardar los cambios");
        finalizeEdit(oldName, oldDesc);
      }
    } catch (err) {
      console.error(err);
      alert("No se pudieron guardar los cambios");
      finalizeEdit(oldName, oldDesc);
    }
  };

  cancelBtn.onclick = () => finalizeEdit(oldName, oldDesc);
}

/* ========= Eventos: Crear proyecto ========= */
projectForm?.addEventListener("submit", async (e) => {
  e.preventDefault(); e.stopPropagation();

  const name = $("#projectName")?.value?.trim?.();
  const description = $("#projectDesc")?.value?.trim?.();
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  if (!name) { alert("El nombre del proyecto es obligatorio"); return; }
  if (!user?.email) { alert("No hay sesión activa"); return; }

  const data = await apiFetch("/create_project", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description, user: user.email })
  });

  if (data?.success) {
    addProjectToTable(data.project);
    projectForm.reset();
    alert(data.msg);
  } else {
    alert(data?.msg || "No se pudo crear el proyecto");
  }
});

/* ========= Eventos: Subir dataset ========= */
if (datasetForm) {
  datasetForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    e.stopPropagation();

    const file = datasetFileInput?.files?.[0];
    if (!file) { alert("Por favor selecciona un archivo CSV."); return; }

    const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
    const user = JSON.parse(localStorage.getItem("user") || "{}");
    if (!activeProject?.id) { alert("Debes tener un proyecto activo antes de subir un dataset."); return; }
    if (!user?.email) { alert("No hay sesión activa."); return; }

    const formData = new FormData();
    formData.append("project_id", activeProject.id);
    formData.append("user", user.email);
    formData.append("file", file);

    const data = await apiFetch("/upload_dataset", { method: "POST", body: formData });
    if (!data?.success) { alert(data?.msg || "Error al subir dataset"); return; }

    alert(data.msg);

    // Persistir dataset en proyecto activo
    activeProject.dataset_path = data.project?.dataset_path || activeProject.dataset_path;
    activeProject.dataset_name = file.name;
    localStorage.setItem("activeProject", JSON.stringify(activeProject));

    setDatasetNameLabel(activeProject.dataset_name);

    if (data.columns && data.preview) {
      renderDatasetPreview(data.columns, data.preview);
    } else {
      await refreshCleanPreview();
    }
  });
}

/* ========= Preprocesar datos ========= */
function ensurePreprocessControls() {
  if (!preprocessBtn) {
    preprocessBtn = document.createElement("button");
    preprocessBtn.id = "preprocessBtn";
    preprocessBtn.textContent = "Preprocesar datos";
    preprocessBtn.style.marginTop = "10px";
    sectionUpload.appendChild(preprocessBtn);
  }
  if (!preprocessResultDiv) {
    preprocessResultDiv = document.createElement("div");
    preprocessResultDiv.id = "preprocessResult";
    sectionUpload.appendChild(preprocessResultDiv);
  }

  preprocessBtn.onclick = async () => {
    const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
    const user = JSON.parse(localStorage.getItem("user") || "{}");
    if (!activeProject?.id || !activeProject?.dataset_path) { alert("Primero debes subir un dataset."); return; }
    if (!user?.email) { alert("No hay sesión activa."); return; }

    if (!confirm("¿Deseas limpiar y preparar el dataset para entrenamiento?")) return;

    const data = await apiFetch("/preprocess_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: activeProject.id, user: user.email })
    });

    if (!data?.success) { alert(data?.msg || "Error en el preprocesamiento."); return; }

    alert(data.msg);

    // Actualizar dataset_path si el backend lo devolvió
    if (data.clean_path) {
      activeProject.dataset_path = data.clean_path;
      localStorage.setItem("activeProject", JSON.stringify(activeProject));
    }

    renderPreprocessSummary(data.summary, data.numeric_cols, data.categorical_cols);

    if (data.columns && data.preview) {
      renderDatasetPreview(data.columns, data.preview);
    } else {
      await refreshCleanPreview();
    }
  };
}

// cambio parametros
modelSelect?.addEventListener("change", async () => {
  const uiVal = modelSelect.value;
  const algKey = ALG_MAP[uiVal];
  if (!algKey) { if (paramsPanel) paramsPanel.classList.add("hidden"); return; }
  try {
    const defaults = await fetchDefaultParams();
    renderParamsForm(algKey, defaults, {}); // puedes pasar existingParams si los obtienes del proyecto
    if (paramsPanel) paramsPanel.classList.remove("hidden");
  } catch (e) {
    console.error(e);
    if (paramsPanel) paramsPanel.classList.add("hidden");
  }
});


/* ========= Recomendación de modelo ========= */
function ensureRecommendButton() {
  if (!recommendBtn) {
    recommendBtn = document.createElement("button");
    recommendBtn.id = "btnRecommend";
    recommendBtn.textContent = "Sugerir modelo";
    recommendBtn.style.marginTop = "10px";
    sectionSelect.appendChild(recommendBtn);
  }
  recommendBtn.onclick = recommendModel;
}

async function recommendModel() {
  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  if (!activeProject?.id) { alert("Selecciona un proyecto primero."); return; }

  // Intento simple: usa columnas de la vista previa si están en DOM
  const cols = Array.from(datasetPreviewHead?.querySelectorAll("th") || []).map((th) => th.textContent);
  let defaultTarget = cols.includes("Y") ? "Y" : (cols.length ? cols[cols.length - 1] : "");
  let target = prompt("Columna objetivo (por ejemplo, Y). Déjalo vacío si no hay:", defaultTarget);
  if (target !== null) target = target.trim();
  if (target === "") target = null;

  const data = await apiFetch("/recommend_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: activeProject.id, user: user.email, target })
  });

  if (!data?.success) { alert(data?.msg || "No se pudo generar una recomendación."); return; }

  const recsHtml = (data.recommendations || [])
    .map((r) => `<li><b>${r.model}</b> — score: ${r.score}<br><small>${r.why}</small></li>`)
    .join("");

  modelRecDiv.innerHTML = `
    <div class="panel">
      <div class="panel-title">Recomendación de modelo</div>
      <p><b>Tarea:</b> ${data.task || "unsupervised"}${data.target ? ` — <b>Objetivo:</b> ${data.target}` : ""}</p>
      <p><b>Features numéricas:</b> ${data.numeric_features?.join(", ") || "Ninguna"}</p>
      <p><b>Features categóricas:</b> ${data.categorical_features?.join(", ") || "Ninguna"}</p>
      <ul>${recsHtml}</ul>
    </div>
  `;
}

/* ========= set/model/params ========= */
saveParamsBtn?.addEventListener("click", async () => {
  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  const uiVal = modelSelectEl?.value;
  const algorithm_key = ALG_MAP[uiVal];
  if (!activeProject?.id || !user?.email || !algorithm_key) {
    alert("Selecciona proyecto, usuario y modelo."); return;
  }

  const params = readParamsForm();
  const res = await fetch("/set_model_params", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: activeProject.id, user: user.email, algorithm_key, params })
  }).then(r=>r.json()).catch(()=>({success:false}));
  if (!res?.success) { alert(res?.msg || "Error guardando parámetros"); return; }
  alert("Parámetros guardados ✔");
});


selectModelWithParamsBtn?.addEventListener("click", async () => {
  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  const uiVal = modelSelectEl?.value;
  const algorithm_key = ALG_MAP[uiVal];
  const category = (uiVal === "pca" || uiVal === "kmeans") ? "unsupervised" : "supervised";
  if (!activeProject?.id || !user?.email || !algorithm_key) {
    alert("Selecciona proyecto, usuario y modelo."); return;
  }

  const params = readParamsForm();
  const res = await fetch("/select_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: activeProject.id, user: user.email, category, algorithm_key, params })
  }).then(r=>r.json()).catch(()=>({success:false}));

  if (!res?.success) { alert(res?.msg || "No se pudo seleccionar el modelo"); return; }
  alert("Modelo seleccionado y parámetros fijados ✔");
});

/* ========= Seleccionar modelo ========= */
modelForm?.addEventListener("submit", async (e) => {
  e.preventDefault(); 
  e.stopPropagation();

  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const value = modelSelect?.value;

  if (!activeProject?.id) { alert("Selecciona un proyecto primero."); return; }
  if (!value) { alert("Selecciona un algoritmo."); return; }

  const algorithm_key = ALG_MAP[value];
  if (!algorithm_key) { alert("Modelo no soportado."); return; }

  const category = (value === "pca" || value === "kmeans") ? "unsupervised" : "supervised";

  // Si el panel está visible, lee parámetros. Si no, manda vacío.
  const params = paramsPanel && !paramsPanel.classList.contains("hidden") ? readParamsForm() : {};

  const data = await apiFetch("/select_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: activeProject.id,
      user: user.email,
      category,
      algorithm_key,
      params
    })
  });

  if (data?.success) {
    alert("Modelo seleccionado y guardado en el proyecto.");
  } else {
    alert(data?.msg || "No se pudo seleccionar el modelo.");
  }
});



/* ========= Entrenar modelo ========= */
trainButton?.addEventListener("click", async () => {
  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");

  if (!activeProject?.id) { alert("Selecciona un proyecto primero."); return; }

  // Pedimos target
  const cols = Array.from(datasetPreviewHead?.querySelectorAll("th") || []).map((th) => th.textContent);
  let defaultTarget = cols.includes("Y") ? "Y" : (cols.length ? cols[cols.length - 1] : "");
  let target = prompt("Columna objetivo (target) para entrenar:", defaultTarget);
  if (!target) { alert("Debes indicar la columna objetivo (target)."); return; }

  setText(trainingStatus, "Entrenando modelo...");

  const data = await apiFetch("/train_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: activeProject.id, user: user.email, target })
  });

  if (!data?.success) {
    setText(trainingStatus, "");
    alert(data?.msg || "Error durante el entrenamiento.");
    return;
  }

  // Mostrar métricas en Resultados
  setText(trainingStatus, "Entrenamiento completado.");
  renderMetrics(data.metrics);
  renderTrainingCurve(data.training_curve);

  // Habilitar sección Predecir
  if (data.metrics && Array.isArray(data.metrics.features_used)) {
    enablePredictUI(data.metrics.features_used, data.metrics.task);
    $("#predict")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

/** Dibuja métricas (regresión / clasificación / clustering / reducción) */
function renderMetrics(m) {
  if (!m) { metricsBox.innerHTML = ""; return; }
  let html = `<div class="panel"><div class="panel-title">Métricas (${m.task ? capitalize(m.task) : "Modelo"})</div><ul>`;
  const safe = (v) => (typeof v === "number" ? Number(v.toFixed ? v.toFixed(6) : v) : v);

  if (m.task === "regression") {
    html += `<li><b>MSE:</b> ${safe(m.mse)}</li>`;
    html += `<li><b>MAE:</b> ${safe(m.mae)}</li>`;
    html += `<li><b>R²:</b> ${safe(m.r2)}</li>`;
    if (m.intercept !== undefined) html += `<li><b>Intercept:</b> ${safe(m.intercept)}</li>`;
    if (m.coefficients) {
      html += `<li><b>Coeficientes:</b><ul>`;
      Object.entries(m.coefficients).forEach(([k, v]) => {
        html += `<li>${k}: ${safe(v)}</li>`;
      });
      html += `</ul></li>`;
    }
  } else if (m.task === "classification") {
    html += `<li><b>Accuracy:</b> ${safe(m.accuracy)}</li>`;
    if (m.confusion_matrix) {
      const labels = Object.keys(m.confusion_matrix).map(Number).sort((a,b)=>a-b);
      html += `<li><b>Matriz de confusión:</b><table class="mini"><thead><tr><th></th>${labels.map(l=>`<th>${l}</th>`).join("")}</tr></thead><tbody>`;
      labels.forEach((i)=> {
        html += `<tr><th>${i}</th>`;
        labels.forEach((j)=> {
          html += `<td>${m.confusion_matrix[i]?.[j] ?? 0}</td>`;
        });
        html += `</tr>`;
      });
      html += `</tbody></table></li>`;
    }
  } else if (m.task === "clustering") {
    html += `<li><b>Muestras:</b> ${safe(m.n_samples)}</li>`;
    if (m.inertia !== undefined) html += `<li><b>Inercia:</b> ${safe(m.inertia)}</li>`;
  } else if (m.task === "dimensionality_reduction") {
    if (Array.isArray(m.explained_variance_ratio)) {
      html += `<li><b>Varianza explicada:</b> ${m.explained_variance_ratio.map((v)=>safe(v)).join(", ")}</li>`;
    }
  }
  if (m.n_train !== undefined) html += `<li><b>n_train:</b> ${safe(m.n_train)}</li>`;
  if (m.n_test !== undefined)  html += `<li><b>n_test:</b> ${safe(m.n_test)}</li>`;
  if (Array.isArray(m.features_used)) html += `<li><b>Features usadas:</b> ${m.features_used.join(", ")}</li>`;
  if (m.target) html += `<li><b>Target:</b> ${m.target}</li>`;

  html += `</ul></div>`;
  metricsBox.innerHTML = html;
}

function renderTrainingCurve(curve) {
  if (!chartsBox) return;

  if (trainingChartInstance) {
    trainingChartInstance.destroy();
    trainingChartInstance = null;
  }

  chartsBox.innerHTML = "";

  const panel = document.createElement("div");
  panel.className = "panel";
  const title = document.createElement("div");
  title.className = "panel-title";
  title.textContent = curve?.title || "Progreso del entrenamiento";
  panel.appendChild(title);

  if (!curve || !Array.isArray(curve.series) || !curve.series.length) {
    const msg = document.createElement("p");
    msg.textContent = "El modelo no expuso datos iterativos para graficar.";
    panel.appendChild(msg);
    chartsBox.appendChild(panel);
    return;
  }

  const wrapper = document.createElement("div");
  wrapper.className = "training-chart-wrapper";
  const canvas = document.createElement("canvas");
  wrapper.appendChild(canvas);
  panel.appendChild(wrapper);

  if (curve.notes) {
    const notes = document.createElement("p");
    notes.className = "chart-notes";
    notes.textContent = curve.notes;
    panel.appendChild(notes);
  }

  chartsBox.appendChild(panel);

  if (typeof Chart === "undefined") {
    const fallback = document.createElement("p");
    fallback.textContent = "No fue posible cargar la librería de gráficos.";
    panel.appendChild(fallback);
    return;
  }

  const palette = ["#1d3557", "#e63946", "#2a9d8f", "#ffb703", "#6d597a", "#118ab2", "#fb8500", "#8ecae6"];
  const datasets = curve.series
    .map((serie, idx) => {
      const values = Array.isArray(serie?.values) ? serie.values : [];
      const points = values
        .map((value, i) => (typeof value === "number" && Number.isFinite(value) ? { x: i + 1, y: value } : null))
        .filter(Boolean);
      return {
        label: serie?.name || `Serie ${idx + 1}`,
        data: points,
        borderColor: palette[idx % palette.length],
        backgroundColor: palette[idx % palette.length] + "33",
        fill: false,
        tension: 0.2,
        spanGaps: true
      };
    })
    .filter((d) => d.data.length);

  if (!datasets.length) {
    const msg = document.createElement("p");
    msg.textContent = "No hay valores numéricos para mostrar.";
    panel.appendChild(msg);
    return;
  }

  trainingChartInstance = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      interaction: { intersect: false, mode: "index" },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: curve.xLabel || "Iteraciones" },
          ticks: { precision: 0 }
        },
        y: {
          title: { display: true, text: curve.yLabel || "Valor" },
          beginAtZero: false
        }
      },
      plugins: {
        legend: { display: datasets.length > 1 },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.parsed.y;
              if (typeof val === "number") {
                const formatted = Number.isInteger(val) ? val : Number(val.toFixed(4));
                return `${ctx.dataset.label}: ${formatted}`;
              }
              return `${ctx.dataset.label}: ${val}`;
            }
          }
        }
      }
    }
  });
}

function capitalize(s) { return (s && s[0].toUpperCase() + s.slice(1)) || s; }

function inferInputType(val) {
  if (typeof val === "boolean") return "checkbox";
  if (typeof val === "number") return "number";
  if (val === null || val === undefined) return "text";
  if (Array.isArray(val)) return "array";
  return "text";
}

async function fetchDefaultParams() {
  if (DEFAULTS_CACHE) return DEFAULTS_CACHE;
  const r = await fetch("/get_default_params");
  const js = await r.json();
  if (!js?.success) throw new Error("No se pudieron obtener DEFAULT_PARAMS");
  DEFAULTS_CACHE = js.defaults || {};
  return DEFAULTS_CACHE;
}

function renderParamsForm(algorithmKey, defaults, existingParams = {}) {
  if (!paramsFormEl) return;
  paramsFormEl.innerHTML = "";

  const base = defaults[algorithmKey] || {};
  const params = { ...base, ...existingParams };

  Object.entries(params).forEach(([key, val]) => {
    const row  = document.createElement("div");
    row.className = "row";
    const label = document.createElement("label");
    label.textContent = key;
    const tooltipText = PARAM_TOOLTIPS[key];
    const rangeText = PARAM_RANGES[key];
    if (tooltipText) label.appendChild(createTooltipIcon(tooltipText, "info"));
    if (rangeText) label.appendChild(createTooltipIcon(`Rango sugerido: ${rangeText}`, "range"));

    if (PARAM_ENUMS[key]) {
      const select = document.createElement("select");
      select.name = key;
      PARAM_ENUMS[key].forEach(opt => {
        const o = document.createElement("option");
        o.value = opt; o.textContent = opt;
        if (String(val) === opt) o.selected = true;
        select.appendChild(o);
      });
      row.appendChild(label);
      row.appendChild(select);
    } else {
      const type = inferInputType(val);
      let input = document.createElement("input");
      input.name = key;

      if (type === "checkbox") {
        input.type = "checkbox";
        input.checked = Boolean(val);
      } else if (type === "number") {
        input.type = "number";
        input.step = "any";
        input.value = Number.isFinite(val) ? val : 0;
      } else if (type === "array") {
        input.type = "text";
        input.value = Array.isArray(val) ? JSON.stringify(val) : "[]";
        const hint = document.createElement("small");
        hint.textContent = "JSON, ej. [64,32]";
        row.appendChild(label);
        row.appendChild(input);
        row.appendChild(hint);
        paramsFormEl.appendChild(row);
        return;
      } else {
        input.type = "text";
        input.value = (val === null || val === undefined) ? "" : String(val);
      }

      row.appendChild(label);
      row.appendChild(input);
    }

    paramsFormEl.appendChild(row);
  });
}

function createTooltipIcon(text, variant = "info") {
  const wrapper = document.createElement("span");
  wrapper.className = `param-tooltip ${variant}`;
  wrapper.tabIndex = 0;
  wrapper.setAttribute("role", "img");
  wrapper.setAttribute("aria-label", text);
  const icon = document.createElement("span");
  icon.className = "param-tooltip-icon";
  icon.textContent = variant === "range" ? "↔" : "?";
  const bubble = document.createElement("span");
  bubble.className = "param-tooltip-text";
  bubble.textContent = text;
  wrapper.appendChild(icon);
  wrapper.appendChild(bubble);
  return wrapper;
}

function readParamsForm() {
  if (!paramsFormEl) return {};
  const inputs = Array.from(paramsFormEl.querySelectorAll("input,select"));
  const params = {};
  inputs.forEach(inp => {
    const name = inp.name;
    if (inp.tagName === "SELECT") {
      params[name] = inp.value;
    } else if (inp.type === "checkbox") {
      params[name] = inp.checked;
    } else if (inp.type === "number") {
      const num = Number(inp.value);
      params[name] = Number.isFinite(num) ? num : 0;
    } else {
      const v = (inp.value || "").trim();
      if ((v.startsWith("[") && v.endsWith("]")) || (v.startsWith("{") && v.endsWith("}"))) {
        try { params[name] = JSON.parse(v); } catch { params[name] = v; }
      } else if (v.toLowerCase() === "true")  { params[name] = true; }
      else if (v.toLowerCase() === "false")   { params[name] = false; }
      else if (v.toLowerCase() === "none" || v.toLowerCase() === "null" || v === "") { params[name] = null; }
      else if (!isNaN(Number(v))) { params[name] = Number(v); }
      else { params[name] = v; }
    }
  });
  return params;
}


// ====== cuando cambie el modelo, pintar TODOS los parámetros ======
modelSelectEl?.addEventListener("change", async () => {
  const uiVal = modelSelectEl.value;
  const algorithm_key = ALG_MAP[uiVal];
  if (!algorithm_key) { paramsPanel?.classList.add("hidden"); return; }
  try {
    const defaults = await fetchDefaultParams();
    renderParamsForm(algorithm_key, defaults, {});
    paramsPanel?.classList.remove("hidden");
  } catch (e) {
    console.error(e);
    paramsPanel?.classList.add("hidden");
  }
});




/* ========= Inicio ========= */
window.addEventListener("DOMContentLoaded", () => {
  const user = JSON.parse(localStorage.getItem("user") || "null");
  if (!user) {
    alert("Debes iniciar sesión primero.");
    window.location.href = "index.html";
    return;
  }

  setText(headerTitle, `Aivana - Bienvenido, ${user.name}`);
  loadProjects(user.email);

  // Asegurar controles que podrían no estar en HTML
  ensurePreprocessControls();
  ensureRecommendButton();
  fetchDefaultParams().catch(console.warn);
});

// Cerrar sesión
logoutLink?.addEventListener("click", () => {
  localStorage.removeItem("user");
});
