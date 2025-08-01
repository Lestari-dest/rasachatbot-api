import os
import subprocess
import shutil

def setup_project():
    """Setup project structure otomatis"""
    print("🚀 Setting up RasaChatbot API project...")
    
    # Create directories
    directories = [
        'chatbot',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        'chatbot/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write("# This file makes the directory a Python package\n")
        print(f"✅ Created file: {init_file}")
    
    # Check if .env exists
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            shutil.copy('.env.example', '.env')
            print("✅ Created .env from .env.example")
            print("⚠️  Remember to edit .env and add your GEMINI_API_KEY!")
        else:
            print("⚠️  .env.example not found, create it manually")
    
    print("🎉 Project setup complete!")
    print("\n📝 Next steps:")
    print("1. Edit .env and add your GEMINI_API_KEY")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python app.py")
    print("4. Test: python test_api.py")

if __name__ == "__main__":
    setup_project()