const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(bodyParser.json());
app.use('/public', express.static(path.join(__dirname, 'public')));

// Serve the homepage
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'index.html'));
});

// Serve the sentiment analysis page
app.get('/sentiment_analysis', (req, res) => {
    res.sendFile(path.join(__dirname, 'templates', 'sentiment_analysis.html'));
});

// Handle sentiment analysis requests
app.post('/predict', async (req, res) => {
    const tweet = req.body.tweet;

    try {
        const response = await axios.post('http://127.0.0.1:5000/predict', { tweet });
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: 'Error calling Python model.' });
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
    console.log('Serving static files from:', path.join(__dirname, 'public'));
});
