// Navigation Logic
document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');

    if (loginForm) {
        // Login Form Handler
        loginForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value.trim();

            if (username === 'admin' && password === 'admin123') {
                window.location.href = 'features.html';
            } else {
                alert('Invalid credentials.');
            }
        });
    }

    // Logout Button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            alert('Logging out...');
            window.location.href = 'index.html';
        });
    }

    // Placeholder Button Actions
    const cameraBtn = document.getElementById('camera-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const captureBtn = document.getElementById('capture-btn');

    if (cameraBtn) {
        cameraBtn.addEventListener('click', () => alert('Opening camera...'));
    }

    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => alert('Uploading video...'));
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', () => alert('Capturing video...'));
    }
});

document.getElementById('upload-btn').addEventListener('click', function () {
    window.location.href = 'count.html';
});
