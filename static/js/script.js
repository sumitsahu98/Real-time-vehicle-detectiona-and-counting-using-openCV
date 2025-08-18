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
                // ✅ Redirect to Flask route instead of features.html
                window.location.href = '/features';
            } else {
                alert('Invalid credentials.Use username admin and password admin123');
            }
        });
    }

    // Logout Button
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            alert('Logging out...');
            // ✅ Redirect to home (Flask route "/")
            window.location.href = '/';
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
        uploadBtn.addEventListener('click', () => {
            alert('Uploading video...');
            // ✅ Redirect to Flask route instead of count.html
            window.location.href = '/count';
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', () => alert('Capturing video...'));
    }
});
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
                // ✅ Redirect to Flask route instead of features.html
                window.location.href = '/features';
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
            // ✅ Redirect to home (Flask route "/")
            window.location.href = '/';
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
        uploadBtn.addEventListener('click', () => {
            alert('Uploading video...');
            // ✅ Redirect to Flask route instead of count.html
            window.location.href = '/count';
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', () => alert('Capturing video...'));
    }
});
