// ====== Configuration ======
const imageFolder = 'images/';
const imageList = [
   'law1.jpg',
    'law2.jpg',
    'law3.jpg',
    'law4.jpg',
    'law5.jpg',
    'law6.jpg',
    'law7.jpg'
];

// ====== DOM Elements ======
const carouselContainer = document.getElementById('carouselImages');
const searchInput = document.getElementById('searchInput');
const voiceIcon = document.getElementById('voiceIcon');
const voiceSearch = document.getElementById('voiceSearch');
const resultsContainer = document.getElementById('resultsContainer');
const legalResponse = document.getElementById('legalResponse');
const citations = document.getElementById('citations');
const closeResults = document.getElementById('closeResults');
const malayalamToggle = document.getElementById('malayalamToggle');

// Check if user is authenticated
function checkAuth() {
    const token = localStorage.getItem('token');
    if (!token) {
        // Redirect to login page if not authenticated
        window.location.href = 'start.html';
    }
}

function logOut() {
    // Remove authentication data from localStorage
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    
    // Redirect to the start page
    window.location.href = 'start.html';
}

// ====== Carousel Implementation ======
let currentIndex = 0;
let isCarouselPaused = false;
let autoSlideInterval;
let cards = [];
let touchStartX = 0;
let touchEndX = 0;

// Initialize carousel
function initCarousel() {
    // Create cards with lazy loading for non-critical images
    imageList.forEach((img, index) => {
        const card = document.createElement('div');
        card.className = 'card';
        if (index === 0) {
            card.classList.add('active');
        }
        
        const imgEl = document.createElement('img');
        imgEl.alt = `Legal concept illustration ${index + 1}`;
        
        // Only load the first two images immediately, lazy load the rest
        if (index < 2) {
            imgEl.src = `${imageFolder}${img}`;
        } else {
            imgEl.dataset.src = `${imageFolder}${img}`;
            imgEl.loading = 'lazy';
        }
        
        card.appendChild(imgEl);
        carouselContainer.appendChild(card);
    });
    
    cards = document.querySelectorAll('.card');
    
    // Start auto-slide
    startAutoSlide();
    
    // Add event listeners for carousel controls
    document.querySelector('.prev-btn').addEventListener('click', prevSlide);
    document.querySelector('.next-btn').addEventListener('click', nextSlide);
    
    // Add touch swipe support for mobile
    carouselContainer.addEventListener('touchstart', handleTouchStart, false);
    carouselContainer.addEventListener('touchend', handleTouchEnd, false);
    
    // Pause auto-slide when hovering over carousel
    carouselContainer.addEventListener('mouseenter', () => { isCarouselPaused = true; });
    carouselContainer.addEventListener('mouseleave', () => { isCarouselPaused = false; });
    
    // Load visible slides when they come into view
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target.querySelector('img');
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    delete img.dataset.src;
                }
            }
        });
    }, { threshold: 0.1 });
    
    cards.forEach(card => observer.observe(card));
}

function showSlide(index) {
    cards.forEach((card, i) => {
        if (i === index) {
            card.classList.add('active');
        } else {
            card.classList.remove('active');
        }
    });

    // Calculate the translation with proper card sizing and margins
    const cardWidth = cards[0].offsetWidth;
    const margin = parseInt(getComputedStyle(cards[0]).marginRight);
    const offset = index * (cardWidth + margin);
    carouselContainer.style.transform = `translateX(-${offset}px)`;
}

function nextSlide() {
    currentIndex = (currentIndex + 1) % cards.length;
    showSlide(currentIndex);
}

function prevSlide() {
    currentIndex = (currentIndex - 1 + cards.length) % cards.length;
    showSlide(currentIndex);
}

function startAutoSlide() {
    autoSlideInterval = setInterval(() => {
        if (!isCarouselPaused) {
            nextSlide();
        }
    }, 4000);
}

function handleTouchStart(e) {
    touchStartX = e.changedTouches[0].screenX;
}

function handleTouchEnd(e) {
    touchEndX = e.changedTouches[0].screenX;
    if (touchEndX < touchStartX - 50) {
        nextSlide();
    } else if (touchEndX > touchStartX + 50) {
        prevSlide();
    }
}

// ====== Search Functionality ======
searchInput.addEventListener('input', () => {
    // Dynamically change icon based on input content
    voiceIcon.className = searchInput.value.trim() !== '' 
        ? 'bi bi-send-fill' 
        : 'bi bi-mic-fill';
});

voiceSearch.addEventListener('click', () => {
    if (searchInput.value.trim() !== '') {
        // Send query to ML model
        processLegalQuery(searchInput.value.trim());
    } else {
        // Voice input mode
        startVoiceRecognition();
    }
});

// Close results when button is clicked
closeResults.addEventListener('click', () => {
    resultsContainer.classList.add('hidden');
});

// Toggle Malayalam input
let isInMalayalamMode = false;
malayalamToggle.addEventListener('click', () => {
    isInMalayalamMode = !isInMalayalamMode;
    malayalamToggle.style.color = isInMalayalamMode ? '#ff6b00' : '#fff';
    searchInput.style.fontFamily = isInMalayalamMode ? 'Manjari, sans-serif' : 'inherit';
    searchInput.placeholder = isInMalayalamMode ? 'നിങ്ങളുടെ നിയമ ചോദ്യം ഇവിടെ ചോദിക്കുക' : 'Ask your legal query here';
});

// Voice recognition implementation
function startVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window)) {
        alert('Voice recognition is not supported in your browser.');
        return;
    }
    
    const recognition = new webkitSpeechRecognition();
    recognition.lang = isInMalayalamMode ? 'ml-IN' : 'en-IN';
    recognition.continuous = false;
    
    // Show recording indicator
    voiceIcon.className = 'bi bi-record-fill';
    voiceIcon.style.color = 'red';
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        searchInput.value = transcript;
        voiceIcon.className = 'bi bi-send-fill';
        voiceIcon.style.color = '';
    };
    
    recognition.onerror = () => {
        voiceIcon.className = 'bi bi-mic-fill';
        voiceIcon.style.color = '';
    };
    
    recognition.onend = () => {
        if (voiceIcon.className === 'bi bi-record-fill') {
            voiceIcon.className = 'bi bi-mic-fill';
            voiceIcon.style.color = '';
        }
    };
    
    recognition.start();
}

// ====== Machine Learning Model Integration ======
// TensorFlow.js and model variables
let model;
let ipcData = [];
let tokenizer;
let isModelLoaded = false;

// Load the IPC data and ML model
async function loadModel() {
    try {
        // 1. Load IPC sections data
        const response = await fetch('ipc_sections.csv');
        const csvText = await response.text();
        ipcData = parseCSV(csvText);
        
        // 2. Load the model (using TensorFlow.js)
        model = await tf.loadLayersModel('model/model.json');
        
        // 3. Load the tokenizer/vocab (for text processing)
        const tokenizerResponse = await fetch('model/tokenizer.json');
        tokenizer = await tokenizerResponse.json();
        
        isModelLoaded = true;
        console.log('Legal ML model loaded successfully');
    } catch (error) {
        console.error('Error loading ML model:', error);
    }
}

// Parse CSV data
function parseCSV(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',');
    
    return lines.slice(1).map(line => {
        if (!line.trim()) return null;
        
        const values = line.split(',');
        const entry = {};
        
        headers.forEach((header, i) => {
            entry[header.trim()] = values[i] ? values[i].trim() : '';
        });
        
        return entry;
    }).filter(entry => entry !== null);
}

// Process legal query with ML model
async function processLegalQuery(query) {
    if (!isModelLoaded) {
        legalResponse.innerHTML = '<p>Loading legal knowledge database... Please try again in a moment.</p>';
        resultsContainer.classList.remove('hidden');
        
        // Try to load the model
        await loadModel();
        if (!isModelLoaded) {
            legalResponse.innerHTML = '<p>Could not load the legal assistant. Please check your connection and try again.</p>';
            return;
        }
    }

    // Show loading state
    legalResponse.innerHTML = '<p>Analyzing your legal query...</p>';
    citations.innerHTML = '';
    resultsContainer.classList.remove('hidden');
    
    try {
        // Preprocess the query text
        const processedQuery = preprocessText(query);
        
        // Convert text to tensor
        const queryTensor = textToTensor(processedQuery);
        
        // Get prediction from model
        const predictions = await model.predict(queryTensor);
        const relevanceScores = await predictions.data();
        
        // Get top relevant IPC sections
        const topSections = getTopSections(relevanceScores, 5);
        
        // Generate response based on query and relevant sections
        const response = generateResponse(query, topSections);
        
        // Display the results
        legalResponse.innerHTML = response.mainText;
        
        // Display citations
        citations.innerHTML = '';
        response.citations.forEach(citation => {
            const chip = document.createElement('div');
            chip.className = 'citation-chip';
            chip.textContent = citation.section;
            chip.title = citation.description;
            chip.addEventListener('click', () => {
                alert(`IPC Section ${citation.section}: ${citation.description}`);
            });
            citations.appendChild(chip);
        });
        
    } catch (error) {
        console.error('Error processing query:', error);
        legalResponse.innerHTML = '<p>Sorry, there was an error processing your query. Please try again.</p>';
    }
}

// Text preprocessing function
function preprocessText(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, '')
        .trim();
}

// Convert text to tensor for model input
function textToTensor(text) {
    // Use the tokenizer to convert text to tokens (word IDs)
    const tokens = text.split(' ')
        .map(word => tokenizer.word_index[word] || 0)
        .slice(0, 100); // Assume max sequence length of 100
    
    // Pad sequence to fixed length
    while (tokens.length < 100) {
        tokens.push(0);
    }
    
    // Create a 2D tensor [batch_size=1, sequence_length=100]
    return tf.tensor2d([tokens], [1, 100]);
}

// Get top relevant IPC sections
function getTopSections(scores, numSections) {
    // Create array of [index, score] pairs
    const indexedScores = scores.map((score, index) => [index, score]);
    
    // Sort by score in descending order
    indexedScores.sort((a, b) => b[1] - a[1]);
    
    // Get top N sections
    return indexedScores
        .slice(0, numSections)
        .filter(item => item[1] > 0.1) // Filter by minimum relevance threshold
        .map(item => {
            const sectionIndex = item[0];
            return {
                ...ipcData[sectionIndex],
                relevance: item[1]
            };
        });
}

// Generate response text and citations
function generateResponse(query, relevantSections) {
    if (relevantSections.length === 0) {
        return {
            mainText: `<p>I couldn't find specific IPC sections related to your query: "${query}".</p>
                      <p>You might want to try rephrasing your question or consulting with a legal professional for more specific guidance.</p>`,
            citations: []
        };
    }

    // Create answer text
    let responseText = `<p>Based on your query: <strong>"${query}"</strong>, I found the following information:</p><br>`;
    
    // Add main response based on top section
    const topSection = relevantSections[0];
    responseText += `<p>Your query appears to relate to <strong>${topSection.offense || 'offenses'}</strong> under the Indian Penal Code.</p>`;
    
    if (topSection.description) {
        responseText += `<p>${topSection.description}</p>`;
    }
    
    if (topSection.punishment) {
        responseText += `<p><strong>Punishment:</strong> ${topSection.punishment}</p>`;
    }
    
    // Add disclaimer
    responseText += `<br><p><em>Disclaimer: This information is provided for educational purposes only and should not be considered legal advice. Please consult with a qualified legal professional for specific guidance on your situation.</em></p>`;
    
    // Prepare citations
    const citationsList = relevantSections.map(section => ({
        section: section.article || 'Unknown Section',
        description: section.description || 'No description available',
        relevance: section.relevance
    }));
    
    return {
        mainText: responseText,
        citations: citationsList
    };
}

// ====== Initialize the application ======
document.addEventListener('DOMContentLoaded', function() {
    checkAuth();
    
    // Rest of your existing initialization code
    initCarousel();
    
    // Load ML model in background
    setTimeout(() => {
        loadModel();
    }, 1000);
    
    // Add keyboard shortcut for search (Enter key)
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (searchInput.value.trim() !== '') {
                processLegalQuery(searchInput.value.trim());
            }
        }
    });
});