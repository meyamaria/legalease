// Original UI interaction code
const btnPopup = document.querySelector('.btnLogin-popup');
const cover_box = document.querySelector('.cover_box');
const loginLink = document.querySelector('.login-link');
const registerLink = document.querySelector('.register-link');
const iconClose = document.querySelector('.icon-close');

// Get form elements
const loginForm = document.getElementById('loginForm');
const registerForm = document.getElementById('registerForm');
const messageContainer = document.querySelector('.message-container') || 
                        document.createElement('div'); // Create if it doesn't exist

// Set up message container if needed
if (!document.querySelector('.message-container')) {
    messageContainer.classList.add('message-container');
    cover_box.appendChild(messageContainer);
}

// API endpoint URL - update this with your actual server URL
const API_URL = 'http://localhost:5000/api';

// UI functions
function activateCoverBox() {
    cover_box.classList.add('active');
}

function deactivateCoverBox() {
    cover_box.classList.remove('active');
}

function activatePopup() {
    cover_box.classList.add('active-popup');
}

function deactivateCoverPopup() {
    cover_box.classList.remove('active-popup');
}

// Display messages to user
function showMessage(message, isError = false) {
    messageContainer.textContent = message;
    messageContainer.className = 'message-container';
    if (isError) {
        messageContainer.classList.add('error');
    } else {
        messageContainer.classList.add('success');
    }
    messageContainer.style.display = 'block';
    
    // Hide message after 3 seconds
    setTimeout(() => {
        messageContainer.style.display = 'none';
    }, 3000);
}

// API Functions
async function registerUser(username, email, password) {
    try {
        const response = await fetch(`${API_URL}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                email,
                password
            })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Registration failed');
        }
        
        showMessage('Registration successful! Please log in.');
        // Switch to login form
        deactivateCoverBox();
        return true;
    } catch (error) {
        showMessage(error.message, true);
        return false;
    }
}

async function loginUser(username, password) {
    try {
        const response = await fetch(`${API_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                password
            })
        });

        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Login failed');
        }
        
        // Store the token in localStorage
        localStorage.setItem('token', data.token);
        localStorage.setItem('user', JSON.stringify(data.user));
        
        showMessage('Login successful!');
        deactivateCoverPopup(); // Close the login popup
        
        // Redirect to main application page or update UI to show logged-in state
        setTimeout(() => {
            window.location.href = '/dashboard.html'; // Or update UI to show logged-in state
        }, 1000);
        
        return true;
    } catch (error) {
        showMessage(error.message, true);
        return false;
    }
}

// Check if user is already logged in
function checkAuthStatus() {
    const token = localStorage.getItem('token');
    const user = localStorage.getItem('user');
    
    if (token && user) {
        // Update UI for logged-in user
        const userObj = JSON.parse(user);
        // For example, you could display the username and hide login button
        if (document.querySelector('.user-profile')) {
            document.querySelector('.user-profile').textContent = userObj.username;
            document.querySelector('.user-profile').style.display = 'block';
        }
        if (btnPopup) {
            btnPopup.style.display = 'none';
        }
    }
}

// Event Listeners
registerLink.addEventListener('click', activateCoverBox);
loginLink.addEventListener('click', deactivateCoverBox);
btnPopup.addEventListener('click', activatePopup);
iconClose.addEventListener('click', deactivateCoverPopup);

// Handle form submissions
if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('registerUsername').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        
        await registerUser(username, email, password);
    });
}

if (loginForm) {
    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('loginUsername').value;
        const password = document.getElementById('loginPassword').value;
        
        await loginUser(username, password);
    });
}

// Log out function
function logOut() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    window.location.href = '/index.html'; // Redirect to homepage
}

// Add logout button event listener if it exists
if (document.querySelector('.logout-btn')) {
    document.querySelector('.logout-btn').addEventListener('click', logOut);
}

// Check authentication status when the page loads
document.addEventListener('DOMContentLoaded', checkAuthStatus);