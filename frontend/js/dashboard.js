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
    }
});

// Cerrar sesión → borrar localStorage
const logoutLink = document.querySelector("a[href='index.html']");
if (logoutLink) {
    logoutLink.addEventListener("click", (e) => {
        localStorage.removeItem("user");
    });
}
