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

// Predecir (nueva sección)
const predictInfo   = $("#predictInfo");
const predictForm   = $("#predictForm");
const predictInputs = $("#predictInputs");
const predictResult = $("#predictResult");

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
  if (chartsBox) chartsBox.innerHTML = "";

  // Predecir
  if (predictInputs) predictInputs.innerHTML = "";
  if (predictResult) { predictResult.innerHTML = ""; hide(predictResult); }
  if (predictForm) hide(predictForm);
  if (predictInfo) { setText(predictInfo, "Entrena un modelo para habilitar la predicción con nuevos datos."); show(predictInfo); }
  hide(sectionPredict);
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

  editBtn.onclick = async () => {
    const newName = row.querySelector(".edit-name").value.trim();
    const newDesc = row.querySelector(".edit-desc").value.trim();
    const user = JSON.parse(localStorage.getItem("user") || "{}");

    if (!newName) { alert("El nombre del proyecto no puede estar vacío"); return; }

    // Si no hay cambios, cancelar
    if (newName === oldName && (newDesc === oldDesc || newDesc === "-")) {
      cancelBtn.click(); return;
    }

    // Oculta secciones si cambias el nombre (para evitar incoherencias)
    hide(sectionUpload); hide(sectionSelect); hide(sectionTrain); hide(sectionResults); hide(sectionPredict);
    clearProjectDependentUI();

    const data = await apiFetch("/edit_project", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: project.id, name: newName, description: newDesc, user: user.email })
    });
    if (data?.success) {
      alert("Proyecto actualizado correctamente");
      project.name = newName; project.description = newDesc;
      nameCell.textContent = newName;
      descCell.textContent = newDesc || "-";
      // Actualizar localStorage si este proyecto es el activo
      const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
      if (activeProject?.id === project.id) {
        activeProject.name = newName; activeProject.description = newDesc;
        localStorage.setItem("activeProject", JSON.stringify(activeProject));
      }
    } else {
      alert(data?.msg || "No se pudieron guardar los cambios");
      nameCell.textContent = oldName;
      descCell.textContent = oldDesc || "-";
    }

    row.classList.remove("editing");
    editBtn.textContent = "Editar";
    cancelBtn.remove();
    editBtn.onclick = (e) => enableEditMode(e, project, row);
  };

  cancelBtn.onclick = () => {
    nameCell.textContent = oldName;
    descCell.textContent = oldDesc || "-";
    row.classList.remove("editing");
    editBtn.textContent = "Editar";
    cancelBtn.remove();
    editBtn.onclick = (e) => enableEditMode(e, project, row);
  };
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

/* ========= Seleccionar modelo ========= */
/* ========= Seleccionar modelo ========= */
modelForm?.addEventListener("submit", async (e) => {
  e.preventDefault(); 
  e.stopPropagation();

  const activeProject = JSON.parse(localStorage.getItem("activeProject") || "{}");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const value = modelSelect?.value;

  if (!activeProject?.id) { alert("Selecciona un proyecto primero."); return; }
  if (!value) { alert("Selecciona un algoritmo."); return; }

  // Mapeo de la opción del <select> a la clave de backend
  const map = {
    "linear":       "linear_regression",
    "logistic":     "logistic_regression",
    "perceptron":   "perceptron",
    "decisiontree": "decision_tree",
    "naivebayes":   "naive_bayes",
    "mlp":          "mlp",
    "pca":          "pca",
    "kmeans":       "kmeans"
  };

  const algorithm_key = map[value];
  if (!algorithm_key) { alert("Modelo no soportado."); return; }

  // Derivar category que el backend exige
  const category = (value === "pca" || value === "kmeans") ? "unsupervised" : "supervised";

  const data = await apiFetch("/select_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_id: activeProject.id,
      user: user.email,
      category,            // <<=== NUEVO
      algorithm_key
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

function capitalize(s) { return (s && s[0].toUpperCase() + s.slice(1)) || s; }

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
});

// Cerrar sesión
logoutLink?.addEventListener("click", () => {
  localStorage.removeItem("user");
});
