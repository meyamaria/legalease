const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');

// Initialize Express app
const app = express();

// Middleware
app.use(bodyParser.json()); // for parsing application/json

// MongoDB connection URI (replace with your MongoDB URI)
const dbURI = "mongodb://localhost:27017/legaLEASE"; // Use your MongoDB URI

// Connect to MongoDB
mongoose.connect(dbURI, { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => {
        console.log('Connected to MongoDB');
    })
    .catch((err) => {
        console.log('Error connecting to MongoDB:', err);
    });

// Define a User Schema (optional based on your form fields)
const userSchema = new mongoose.Schema({
    username: String,
    email: String,
    password: String,
});

const User = mongoose.model('User', userSchema);

// Example POST route to register a user
app.post('/register', async (req, res) => {
    const { username, email, password } = req.body;

    const newUser = new User({
        username,
        email,
        password,
    });

    try {
        await newUser.save();
        res.status(201).send('User registered successfully');
    } catch (error) {
        res.status(500).send('Error registering user');
    }
});

// Example POST route to login a user
app.post('/login', async (req, res) => {
    const { email, password } = req.body;

    const user = await User.findOne({ email });

    if (user && user.password === password) {
        res.status(200).send('Login successful');
    } else {
        res.status(400).send('Invalid credentials');
    }
});

// Set the port for the server
const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
