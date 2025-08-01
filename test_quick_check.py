# import os
# from dotenv import load_dotenv

# def quick_test():
#     """Quick test untuk check semua working"""
    
#     print("🚀 QUICK TEST - RasaChatbot")
#     print("=" * 40)
    
#     # 1. Environment check
#     load_dotenv()
#     api_key = os.getenv('GEMINI_API_KEY')
    
#     if not api_key:
#         print("❌ Step 1 FAILED: No API key")
#         return False
#     print("✅ Step 1 PASSED: API key found")
    
#     # 2. Import check
#     try:
#         from chatbot.rasa_chatbot import RasaChatbot
#         print("✅ Step 2 PASSED: Import successful")
#     except Exception as e:
#         print(f"❌ Step 2 FAILED: Import error - {e}")
#         return False
    
#     # 3. Model loading check
#     try:
#         chatbot = RasaChatbot('tartarmee/indobert-sentiment-mental-health', api_key)
#         print("✅ Step 3 PASSED: Model loaded")
#     except Exception as e:
#         print(f"❌ Step 3 FAILED: Model loading error - {e}")
#         return False
    
#     # 4. Sentiment analysis check
#     try:
#         result = chatbot.analyze_sentiment('aku sedih banget')
#         sentiment = result['dominant_sentiment'] 
#         confidence = result['confidence']
#         print(f"✅ Step 4 PASSED: Sentiment = {sentiment} ({confidence:.2f})")
#     except Exception as e:
#         print(f"❌ Step 4 FAILED: Sentiment error - {e}")
#         return False
    
#     # 5. Chat function check
#     try:
#         chat_result = chatbot.chat('hai aku sedih')
#         response = chat_result['response']
#         print(f"✅ Step 5 PASSED: Chat working")
#         print(f"   Sample response: {response[:40]}...")
#     except Exception as e:
#         print(f"❌ Step 5 FAILED: Chat error - {e}")
#         return False
    
#     print("\n🎉 ALL TESTS PASSED!")
#     print("🚀 Your chatbot is ready for production!")
#     return True

# if __name__ == "__main__":
#     quick_test()


# sentiment test

import os
from dotenv import load_dotenv

print("🧪 Testing Sentiment Analysis...")

# Load environment
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    exit(1)

print(f"✅ API key found: {api_key[:20]}...")

# Import chatbot
try:
    from chatbot.rasa_chatbot import RasaChatbot
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Initialize chatbot
try:
    print("📥 Loading model...")
    chatbot = RasaChatbot('tartarmee/indobert-sentiment-mental-health', api_key)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    exit(1)

# Test sentiment analysis
print("\n🔍 Testing sentiment analysis:")
test_text = "aku sedih banget hari ini"

try:
    result = chatbot.analyze_sentiment(test_text)
    sentiment = result['dominant_sentiment']
    confidence = result['confidence']
    
    print(f"Input: '{test_text}'")
    print(f"✅ Sentiment: {sentiment}")
    print(f"✅ Confidence: {confidence:.3f}")
    
    if sentiment == 'sadness':
        print("🎉 CORRECT! Model detected sadness properly!")
    else:
        print(f"⚠️  Expected 'sadness', got '{sentiment}'")
        
except Exception as e:
    print(f"❌ Sentiment analysis failed: {e}")
    exit(1)

print("\n✅ Sentiment analysis test completed successfully!")