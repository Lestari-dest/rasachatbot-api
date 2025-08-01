print("🧪 Testing dependencies...")

try:
    import fastapi
    print("✅ FastAPI:", fastapi.__version__)
except ImportError:
    print("❌ FastAPI not installed")

try:
    import torch
    print("✅ PyTorch:", torch.__version__)
except ImportError:
    print("❌ PyTorch not installed")

try:
    import transformers
    print("✅ Transformers:", transformers.__version__)
except ImportError:
    print("❌ Transformers not installed")

try:
    import google.generativeai as genai
    print("✅ Google GenerativeAI: Available")
except ImportError:
    print("❌ Google GenerativeAI not installed")

try:
    from chatbot.rasa_chatbot import RasaChatbot
    print("✅ RasaChatbot: Import successful")
except ImportError as e:
    print(f"❌ RasaChatbot import failed: {e}")

print("\n✅ Setup test complete!")