from flask import Flask, request, jsonify, render_template
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from safetensors.torch import load_file as load_safetensors
import warnings

warnings.filterwarnings('ignore')

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model and tokenizer
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Load the model state from safetensors
model_weights = load_safetensors('sentiment_model-20240729T190638Z-001/sentiment_model/model.safetensors')
model.load_state_dict(model_weights)
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

MAX_LEN = 100

def preprocess_tweet(tweet, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        tweet,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tweet = data.get('tweet', '').strip()
    if not tweet:
        return jsonify({'error': 'Oops no input found.'}), 400

    processed_tweet = preprocess_tweet(tweet, tokenizer, MAX_LEN)
    input_ids = processed_tweet['input_ids'].to(device)
    attention_mask = processed_tweet['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    confidence_scores = probabilities[0].tolist()
    confidence, prediction = torch.max(probabilities, dim=1)

    sentiment = ['negative', 'neutral', 'positive'][prediction.item()]
    confidence_score = confidence.item()

    # Debug: Print the response being returned
    print({
        'sentiment': sentiment,
        'confidence': confidence_score,
        'confidence_scores': {
            'negative': confidence_scores[0],
            'neutral': confidence_scores[1],
            'positive': confidence_scores[2]
        }
    })

    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence_score,
        'confidence_scores': {
            'negative': confidence_scores[0],
            'neutral': confidence_scores[1],
            'positive': confidence_scores[2]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
