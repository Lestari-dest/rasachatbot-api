import os
import requests
import json
from dotenv import load_dotenv

def validate_environment():
    """Validate environment variables"""
    load_dotenv()
    
    required_vars = ['GEMINI_API_KEY', 'MODEL_NAME']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ All required environment variables present")
    return True

def validate_dependencies():
    """Validate all dependencies are installed"""
    try:
        import torch
        import transformers
        import google.generativeai
        import fastapi
        import uvicorn
        import pydantic
        print("✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def test_local_api():
    """Test local API if running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Local API is running and healthy")
            return True
        else:
            print(f"⚠️  Local API responded with status: {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print("⚠️  Local API is not running (this is OK if not started yet)")
        return True

def validate_project_structure():
    """Validate project has correct structure"""
    required_files = [
        'app.py',
        'requirements.txt',
        'chatbot/__init__.py',
        'chatbot/rasa_chatbot.py',
        '.env',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure is correct")
    return True

def main():
    """Run all validations"""
    print("🔍 Validating deployment readiness...\n")
    
    checks = [
        ("Project Structure", validate_project_structure),
        ("Environment Variables", validate_environment),
        ("Dependencies", validate_dependencies),
        ("Local API", test_local_api)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"Checking {check_name}...")
        if not check_func():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All checks passed! Ready for deployment!")
        print("\n📝 Next steps:")
        print("1. Commit your changes: git add . && git commit -m 'Ready for deployment'")
        print("2. Push to GitHub: git push origin main")
        print("3. Deploy to Render or your preferred platform")
    else:
        print("❌ Some checks failed. Please fix the issues above before deploying.")

if __name__ == "__main__":
    main()