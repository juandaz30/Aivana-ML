// Obtenemos referencias a los elementos del DOM
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const toggleText = document.getElementById("toggleText");

// detector de evento click para el objeto toggleText
toggleText.addEventListener("click", (e) => {
    // cuando da click en 'Regístrate'
    if (e.target.id === "showRegister") {
        e.preventDefault(); // evita saltos a la parte superior al dar clic (comportamiento por defecto HTML)
        loginForm.classList.add("hidden"); // se oculta el login
        registerForm.classList.remove("hidden"); // se muestra el formulario de registro
        toggleText.innerHTML = '¿Ya tienes cuenta? <a id="showLogin" href="#">Inicia Sesión</a>'; // cambia el texto del toogle
    }
    // cuando da click en 'Inicia Sesión'
    if (e.target.id === "showLogin") {
        e.preventDefault();
        registerForm.classList.add("hidden"); // se oculta el formulario de registro
        loginForm.classList.remove("hidden"); // se muestra el login
        toggleText.innerHTML = '¿Aún no tienes cuenta? <a id="showRegister" href="#">Regístrate</a>';
    }
});

// ------------------------------------------------------
// EVENTOS DE ENVÍO DE FORMULARIOS
// ------------------------------------------------------

// función asíncrona que se ejecuta al presionar el botón de Iniciar Sesión
// async define la función como asíncrona, es decir que devuelve una promesa y await pausa la ejecución de la función async y sigue con el hilo de ejecución principal hasta que obtiene una respuesta 
loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    // obiene el usuario y la clave del login
    const email = document.getElementById("loginEmail").value;
    const password = document.getElementById("loginPassword").value;
    // guarda la respuesta del servidor a la petición (fetch) POST de loggeo
    const resp = await fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
    });

    // respuesta del servidor en formato json
    const data = await resp.json();

    // data que devuelve el API guardado en la llave 'success'
    if (data.success) {
        // se guarda la información del usuario en localStorage del navegador
        localStorage.setItem("user", JSON.stringify(data.user));

        // se redirige al dashboard cuando el login es exitoso
        window.location.href = "dashboard.html";
    } else {
        // muestra el mensaje de desautorización
        alert(data.msg);
    }
});


// función asíncrona que se ejecuta al presionar el botón de Registrarse
registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    // obtiene los valores de los campos
    const name = document.getElementById("registerName").value;
    const email = document.getElementById("registerEmail").value;
    const password = document.getElementById("registerPassword").value;
    const confirm = document.getElementById("registerConfirm").value;

    if (password !== confirm) {
        alert("Las contraseñas no coinciden");
        return;
    }
    // guarda la respuesta del servidor a la petición (fetch) POST de registro
    const resp = await fetch("http://127.0.0.1:5000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password })
    });
    // respuesta del servidor en formato json
    const data = await resp.json();
    // muestra el mensaje de creación exitosa o correo existente
    alert(data.msg);
});
