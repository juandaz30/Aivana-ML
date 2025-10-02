// Obtenemos referencias a los elementos del DOM
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const toggleText = document.getElementById("toggleText");

// Función para mostrar el formulario de registro
function showRegisterForm() {
    loginForm.classList.add("hidden");      // Oculta login
    registerForm.classList.remove("hidden"); // Muestra registro
    toggleText.innerHTML = '¿Ya tienes cuenta? <a id="showLogin" href="#">Inicia Sesión</a>';

    // Vuelve a capturar el link "Inicia Sesión"
    document.getElementById("showLogin").addEventListener("click", showLoginForm);
}

// Función para mostrar el formulario de login
function showLoginForm() {
    registerForm.classList.add("hidden");   // Oculta registro
    loginForm.classList.remove("hidden");   // Muestra login
    toggleText.innerHTML = '¿Aún no tienes cuenta? <a id="showRegister" href="#">Regístrate</a>';

    // Vuelve a capturar el link "Regístrate"
    document.getElementById("showRegister").addEventListener("click", showRegisterForm);
}

// Inicializar: escuchar el primer enlace "Regístrate"
document.getElementById("showRegister").addEventListener("click", showRegisterForm);

// ------------------------------------------------------
// EVENTOS DE ENVÍO DE FORMULARIOS
// ------------------------------------------------------

// Aquí simulamos lo que pasará cuando el usuario haga login
loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const email = document.getElementById("loginEmail").value;
    const password = document.getElementById("loginPassword").value;

    const resp = await fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
    });

    const data = await resp.json();

    if (data.success) {
        // Guardamos el usuario en localStorage
        localStorage.setItem("user", JSON.stringify(data.user));

        // Redirigimos al dashboard
        window.location.href = "dashboard.html";
    } else {
        alert(data.msg);
    }
});


// Aquí simulamos el registro de un nuevo usuario
registerForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const name = document.getElementById("registerName").value;
    const email = document.getElementById("registerEmail").value;
    const password = document.getElementById("registerPassword").value;
    const confirm = document.getElementById("registerConfirm").value;

    if (password !== confirm) {
        alert("Las contraseñas no coinciden");
        return;
    }

    const resp = await fetch("http://127.0.0.1:5000/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password })
    });

    const data = await resp.json();
    alert(data.msg);
});
