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
    messageContainer.style.padding = '10px';
    messageContainer.style.margin = '10px 0';
    messageContainer.style.borderRadius = '5px';
    messageContainer.style.display = 'none';
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
    messageContainer.style.backgroundColor = isError ? '#ffebee' : '#e8f5e9';
    messageContainer.style.color = isError ? '#c62828' : '#2e7d32';
    messageContainer.style.border = `1px solid ${isError ? '#ef9a9a' : '#a5d6a7'}`;
    messageContainer.style.display = 'block';
    
    // Hide message after 3 seconds
    setTimeout(() => {
        messageContainer.style.display = 'none';
    }, 3000);
}

// API Functions
async function registerUser(username, email, password) {
    try {
        // For development/testing without a backend server:
        // Store user data in localStorage to simulate registration
        const users = JSON.parse(localStorage.getItem('users') || '[]');
        
        // Check if email already exists
        if (users.some(user => user.email === email)) {
            throw new Error('Email already registered');
        }
        
        // Add new user
        users.push({ username, email, password });
        localStorage.setItem('users', JSON.stringify(users));
        
        showMessage('Registration successful! Redirecting to search page...', false);
        
        // Redirect to index page after a short delay
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
        
        return true;
    } catch (error) {
        showMessage(error.message, true);
        return false;
    }
}

async function loginUser(email, password) {
    try {
        // For development/testing without a backend server:
        // Check credentials against localStorage
        const users = JSON.parse(localStorage.getItem('users') || '[]');
        const user = users.find(u => u.email === email && u.password === password);
        
        if (!user) {
            throw new Error('Invalid email or password');
        }
        
        // Create a simple token (in a real app, this would be a JWT from server)
        const token = btoa(`${email}:${Date.now()}`);
        
        // Store auth data
        localStorage.setItem('token', token);
        localStorage.setItem('user', JSON.stringify({
            username: user.username,
            email: user.email
        }));
        
        showMessage('Login successful! Redirecting to search page...', false);
        
        // Redirect to main application page
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 1500);
        
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
if (registerLink) registerLink.addEventListener('click', activateCoverBox);
if (loginLink) loginLink.addEventListener('click', deactivateCoverBox);
if (btnPopup) btnPopup.addEventListener('click', activatePopup);
if (iconClose) iconClose.addEventListener('click', deactivateCoverPopup);

// Handle form submissions
if (registerForm) {
    registerForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('registerUsername').value;
        const email = document.getElementById('registerEmail').value;
        const password = document.getElementById('registerPassword').value;
        
        // Use the registerUser function instead of duplicating code
        await registerUser(username, email, password);
    });
}

if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const email = document.getElementById("loginEmail").value;
        const password = document.getElementById("loginPassword").value;
    
        // Use the loginUser function instead of duplicating code
        await loginUser(email, password);
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
document.addEventListener('DOMContentLoaded', function() {
    // Get the form elements now that the DOM is loaded
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            
            await loginUser(email, password);
        });
    }
    
    if (registerForm) {
        registerForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const username = document.getElementById('registerUsername').value;
            const email = document.getElementById('registerEmail').value;
            const password = document.getElementById('registerPassword').value;
            
            await registerUser(username, email, password);
        });
    }
    
    // Check authentication status
    checkAuthStatus();
});