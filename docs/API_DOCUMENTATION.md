# RasaChatbot API Documentation

## Overview

RasaChatbot API adalah API chatbot empati yang adaptif dengan analisis sentimen real-time dan fitur penelitian. Dibangun dengan FastAPI dan menggunakan IndoBERT untuk analisis emosi.

## Base URL

- **Production**: `https://your-app-name.onrender.com`
- **Local Development**: `http://localhost:8000`

## Authentication

Saat ini API tidak memerlukan authentication. Untuk production, pertimbangkan menambahkan API key atau JWT authentication.

## Endpoints

### Health Check

#### GET `/health`

Mengecek status kesehatan API.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-01-29T10:30:00",
  "version": "1.0.0"
}
```

### Chat

#### POST `/chat`

Endpoint utama untuk berinteraksi dengan chatbot.

**Request Body:**

```json
{
  "message": "string (required)",
  "user_id": "string (optional, default: 'default')",
  "reset_session": "boolean (optional, default: false)"
}
```

**Response:**

```json
{
  "response": "Kedengeran berat ya. Mau cerita apa yang bikin sedih?",
  "sentiment": "sadness",
  "confidence": 0.87,
  "transition": null,
  "empathy_level": 1,
  "turn_count": 1,
  "timestamp": "2025-01-29T10:30:00",
  "user_id": "user123",
  "research_data": {
    "style_analysis": {
      "pronouns": "aku",
      "exclamation_level": "low",
      "uses_repetition": false,
      "uses_emoji": false,
      "slang_words": ["banget"],
      "casualness": 1,
      "message_length": "medium"
    },
    "personality_profile": {
      "preferred_pronouns": "aku",
      "formality_level": "casual",
      "emoji_usage": false,
      "exclamation_tendency": "low",
      "openness_level": 1
    },
    "short_term_memory": [],
    "special_case": false,
    "confidence_threshold": 0.65,
    "previous_sentiment": null
  }
}
```

### Reset Session

#### POST `/reset`

Reset session percakapan untuk user tertentu.

**Request Body:**

```json
{
  "user_id": "string (optional)"
}
```

**Response:**

```json
{
  "message": "Session reset successfully",
  "user_id": "user123"
}
```

### Statistics

#### GET `/stats`

Mendapatkan statistik chatbot untuk keperluan penelitian.

**Response:**

```json
{
  "turn_count": 5,
  "personality_profile": {
    "preferred_pronouns": "aku",
    "openness_level": 2
  },
  "memory_size": 3,
  "previous_sentiment": "happy",
  "confidence_threshold": 0.65,
  "available_sentiments": ["sadness", "love", "anger", "happy", "fear"]
}
```

## Data Models

### Sentiment Types

- `sadness`: Kesedihan
- `love`: Cinta/kasih sayang
- `anger`: Kemarahan
- `happy`: Kebahagiaan
- `fear`: Ketakutan
- `neutral`: Netral (fallback)

### Empathy Levels

- `1`: Simple validation (turn 1-2)
- `2`: Deeper understanding (turn 3-5)
- `3`: Meaningful connection (turn 6+)

### Style Analysis Fields

- `pronouns`: gue/aku/saya
- `exclamation_level`: low/medium/high
- `uses_repetition`: boolean
- `uses_emoji`: boolean
- `slang_words`: array of detected slang
- `message_length`: short/medium/long

## Error Codes

- `200`: Success
- `422`: Validation Error (invalid input)
- `500`: Internal Server Error

## Rate Limiting

Tidak ada rate limiting saat ini. Untuk production, pertimbangkan:

- 100 requests per minute per user
- 1000 requests per hour per IP

## Examples

### Flutter Integration

```dart
class RasaChatbotService {
  static const String baseUrl = 'https://your-app.onrender.com';

  static Future<ChatResponse> sendMessage(String message) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'message': message,
        'user_id': 'flutter_user'
      }),
    );

    if (response.statusCode == 200) {
      return ChatResponse.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to send message');
    }
  }
}
```

### JavaScript/Web Integration

```javascript
class RasaChatbotAPI {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  async sendMessage(message, userId = "web_user") {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      throw new Error("Failed to send message");
    }

    return await response.json();
  }

  async resetSession(userId = "web_user") {
    const response = await fetch(`${this.baseUrl}/reset`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ user_id: userId }),
    });

    return await response.json();
  }
}
```

### Python Integration

```python
import requests
import json

class RasaChatbotClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def send_message(self, message, user_id="python_user"):
        response = self.session.post(
            f"{self.base_url}/chat",
            headers={"Content-Type": "application/json"},
            json={
                "message": message,
                "user_id": user_id
            }
        )
        response.raise_for_status()
        return response.json()

    def reset_session(self, user_id="python_user"):
        response = self.session.post(
            f"{self.base_url}/reset",
            json={"user_id": user_id}
        )
        return response.json()

# Usage
client = RasaChatbotClient("https://your-app.onrender.com")
result = client.send_message("Hai, aku sedih banget hari ini")
print(f"Bot: {result['response']}")
print(f"Sentiment: {result['sentiment']} ({result['confidence']:.2f})")
```

## Research Data Usage

### Analyzing User Communication Patterns

```python
def analyze_conversation_patterns(chat_responses):
    """Analyze patterns from chat responses"""

    sentiments = [r['sentiment'] for r in chat_responses]
    transitions = [r['transition'] for r in chat_responses if r['transition']]

    # Sentiment distribution
    sentiment_counts = Counter(sentiments)

    # Style evolution
    style_evolution = []
    for response in chat_responses:
        style_data = response['research_data']['style_analysis']
        style_evolution.append({
            'turn': response['turn_count'],
            'pronouns': style_data.get('pronouns'),
            'exclamation_level': style_data.get('exclamation_level'),
            'casualness': style_data.get('casualness', 0)
        })

    return {
        'sentiment_distribution': sentiment_counts,
        'emotion_transitions': transitions,
        'style_evolution': style_evolution,
        'empathy_progression': [r['empathy_level'] for r in chat_responses]
    }
```

### Personality Profiling

```python
def track_personality_development(research_data_list):
    """Track how user personality profile develops over time"""

    profiles = []
    for data in research_data_list:
        profile = data['research_data']['personality_profile']
        profiles.append({
            'openness_level': profile['openness_level'],
            'emoji_usage': profile['emoji_usage'],
            'preferred_pronouns': profile['preferred_pronouns'],
            'exclamation_tendency': profile['exclamation_tendency']
        })

    return profiles
```

## Deployment Guide

### Prerequisites

1. **Python 3.9+**
2. **Git**
3. **Gemini API Key** - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Local Development

```bash
# Clone repository
git clone https://github.com/your-username/rasachatbot-api.git
cd rasachatbot-api

# Setup environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run development server
python app.py
```

### Deploy to Render

1. **Push to GitHub**

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Create Render Service**

   - Go to [render.com](https://render.com)
   - Connect GitHub repository
   - Create new Web Service
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py`

3. **Set Environment Variables**
   - `GEMINI_API_KEY`: Your actual API key
   - `MODEL_NAME`: `tartarmee/indobert-sentiment-mental-health`
   - `PORT`: `8000`

### Deploy with Docker

```bash
# Build image
docker build -t rasachatbot-api .

# Run container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_api_key \
  -e MODEL_NAME=tartarmee/indobert-sentiment-mental-health \
  rasachatbot-api
```

### Deploy with Docker Compose

```bash
# Create .env file with your variables
echo "GEMINI_API_KEY=your_api_key" > .env

# Run with docker-compose
docker-compose up -d
```

## Monitoring and Maintenance

### Health Monitoring

Set up monitoring untuk endpoint `/health` dengan interval 5 menit.

### Logs

API menggunakan Python logging. Dalam production, setup log aggregation seperti:

- Render: Built-in logs
- Docker: Volume mounting untuk log files
- Cloud platforms: CloudWatch, StackDriver

### Performance Optimization

1. **Model Caching**: Model IndoBERT di-cache otomatis oleh HuggingFace
2. **Response Time**: Target < 3 detik per request
3. **Memory Usage**: Monitor penggunaan RAM untuk model
4. **Gemini API Limits**: Monitor quota usage

### Scaling Considerations

- **Vertical Scaling**: Increase RAM/CPU untuk model processing
- **Horizontal Scaling**: Multiple instances dengan load balancer
- **Database**: Add persistent storage untuk user sessions
- **Caching**: Redis untuk session management

## Security

### Current Security Measures

- CORS middleware untuk cross-origin requests
- Input validation dengan Pydantic
- Error handling tanpa expose internal details

### Production Security Recommendations

1. **API Authentication**: Add API key atau JWT
2. **Rate Limiting**: Prevent abuse
3. **Input Sanitization**: Additional validation
4. **HTTPS**: Enforce SSL/TLS
5. **Environment Variables**: Never commit secrets

## Support and Contributing

### Getting Help

1. Check API documentation
2. Test endpoints dengan `/docs` (Swagger UI)
3. Review logs untuk error details
4. Use production test script: `python production_test.py <API_URL>`

### Contributing

1. Fork repository
2. Create feature branch
3. Add tests untuk new features
4. Submit pull request

## Changelog

### v1.0.0 (2025-01-29)

- Initial release
- Basic chat functionality
- Sentiment analysis dengan IndoBERT
- Style mirroring dan empathy progression
- Research data output
- FastAPI implementation
- Deployment ready untuk Render/Docker

---

_Generated by RasaChatbot API v1.0.0_
