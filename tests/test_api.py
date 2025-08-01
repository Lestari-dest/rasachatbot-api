import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

client = TestClient(app)

class TestAPI:
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "RasaChatbot API is running"

    @patch('app.chatbot')
    def test_chat_endpoint_success(self, mock_chatbot):
        """Test successful chat interaction"""
        # Mock chatbot response
        mock_chatbot.chat.return_value = {
            'response': 'Test response',
            'sentiment': 'happy',
            'confidence': 0.8,
            'transition': None,
            'empathy_level': 1,
            'style_analysis': {'pronouns': 'aku'},
            'special_case': False
        }
        mock_chatbot.turn_count = 1
        mock_chatbot.user_personality_profile = {'openness_level': 1}
        mock_chatbot.short_term_memory = []
        mock_chatbot.confidence_threshold = 0.65
        mock_chatbot.previous_sentiment = None

        payload = {
            "message": "Test message",
            "user_id": "test_user"
        }

        response = client.post("/chat", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert data["response"] == "Test response"
        assert data["sentiment"] == "happy"
        assert data["user_id"] == "test_user"
        assert "research_data" in data

    def test_chat_endpoint_validation(self):
        """Test chat endpoint input validation"""
        # Test empty message
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

        # Test missing message
        response = client.post("/chat", json={"user_id": "test"})
        assert response.status_code == 422

    @patch('app.chatbot')
    def test_reset_endpoint(self, mock_chatbot):
        """Test reset endpoint"""
        response = client.post("/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "user_id" in data
        mock_chatbot.reset_chat_session.assert_called_once()

    @patch('app.chatbot')
    def test_stats_endpoint(self, mock_chatbot):
        """Test stats endpoint"""
        mock_chatbot.turn_count = 5
        mock_chatbot.user_personality_profile = {'openness_level': 2}
        mock_chatbot.short_term_memory = []
        mock_chatbot.previous_sentiment = 'happy'
        mock_chatbot.confidence_threshold = 0.65
        mock_chatbot.labels = ['sadness', 'love', 'anger', 'happy', 'fear']

        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "turn_count" in data
        assert "personality_profile" in data
        assert "available_sentiments" in data

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])