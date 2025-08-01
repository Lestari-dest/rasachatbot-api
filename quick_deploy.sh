#!/bin/bash

echo "🚀 RasaChatbot Quick Deploy Script"
echo "=================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "Please copy .env.example to .env and fill in your API keys"
    exit 1
fi

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📁 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: RasaChatbot API"
else
    echo "📁 Git repository already exists"
fi

# Validate deployment
echo "🔍 Running deployment validation..."
python validate_deployment.py

if [ $? -ne 0 ]; then
    echo "❌ Validation failed! Please fix the issues before deploying."
    exit 1
fi

# Commit changes
echo "📝 Committing changes..."
git add .
git commit -m "Ready for deployment - $(date)"

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  No git remote found!"
    echo "Please add your GitHub repository:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/rasachatbot-api.git"
    exit 1
fi

# Push to GitHub
echo "📤 Pushing to GitHub..."
git push origin main

echo "✅ Code pushed to GitHub!"
echo ""
echo "🌐 Next steps for Render deployment:"
echo "1. Go to https://render.com"
echo "2. Connect your GitHub repository"
echo "3. Create a new Web Service"
echo "4. Set environment variables (GEMINI_API_KEY, etc.)"
echo "5. Deploy!"
echo ""
echo "📖 For detailed instructions, see the deployment guide in README.md"