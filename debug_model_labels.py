import os
from dotenv import load_dotenv
import requests
import json

def comprehensive_test():
    """Single comprehensive test untuk semua fitur RasaChatbot"""
    
    print("🚀 RASACHATBOT COMPREHENSIVE TEST")
    print("=" * 60)
    print("Testing all features in one go...")
    
    # Setup
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    # Import chatbot
    try:
        from chatbot.rasa_chatbot import RasaChatbot
        chatbot = RasaChatbot('tartarmee/indobert-sentiment-mental-health', api_key)
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"❌ Chatbot initialization failed: {e}")
        return False
    
    # ==========================================
    # TEST 1: SENTIMENT ACCURACY
    # ==========================================
    print(f"\n{'='*60}")
    print("🎯 TEST 1: SENTIMENT ACCURACY")
    print("=" * 60)
    
    sentiment_tests = [
        # Sadness
        ("aku sedih banget hari ini", "sadness"),
        ("kecewa banget sama hidup", "sadness"),
        ("galau banget rasanya", "sadness"),
        
        # Anger  
        ("kesel banget sama temen", "anger"),
        ("marah banget aku", "anger"),
        ("emosi banget hari ini", "anger"),
        
        # Happy
        ("seneng banget hari ini", "happy"),
        ("bahagia banget rasanya", "happy"),
        ("excited banget aku", "happy"),
        
        # Love
        ("sayang banget sama dia", "love"),
        ("cinta banget sama pacar", "love"),
        ("rindu banget sama kamu", "love"),
        
        # Fear
        ("takut banget besok ujian", "fear"),
        ("deg-degan banget", "fear"),
        ("nervous banget aku", "fear")
    ]
    
    sentiment_correct = 0
    sentiment_total = len(sentiment_tests)
    
    print(f"Testing {sentiment_total} sentiment examples...")
    
    sentiment_results = {}
    for text, expected in sentiment_tests:
        result = chatbot.analyze_sentiment(text)
        predicted = result['dominant_sentiment']
        confidence = result['confidence']
        is_correct = predicted == expected
        
        if is_correct:
            sentiment_correct += 1
        
        # Group by expected sentiment for analysis
        if expected not in sentiment_results:
            sentiment_results[expected] = {'correct': 0, 'total': 0, 'examples': []}
        
        sentiment_results[expected]['total'] += 1
        if is_correct:
            sentiment_results[expected]['correct'] += 1
        
        sentiment_results[expected]['examples'].append({
            'text': text,
            'predicted': predicted,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Print sentiment results
    for sentiment, data in sentiment_results.items():
        accuracy = data['correct'] / data['total']
        print(f"\n{sentiment.upper()}: {accuracy:.1%} ({data['correct']}/{data['total']})")
        
        for example in data['examples']:
            status = "✅" if example['correct'] else "❌"
            print(f"   {status} '{example['text'][:30]}...' → {example['predicted']} ({example['confidence']:.2f})")
    
    sentiment_accuracy = sentiment_correct / sentiment_total
    print(f"\n📊 OVERALL SENTIMENT ACCURACY: {sentiment_accuracy:.1%} ({sentiment_correct}/{sentiment_total})")
    
    # ==========================================
    # TEST 2: STYLE ANALYSIS
    # ==========================================
    print(f"\n{'='*60}")
    print("🎭 TEST 2: STYLE ANALYSIS")
    print("=" * 60)
    
    style_tests = [
        # Pronoun variations
        ("gue sedih banget", {"pronouns": "gue"}),
        ("aku seneng banget", {"pronouns": "aku"}),
        ("saya khawatir sekali", {"pronouns": "saya"}),
        
        # Exclamation levels
        ("biasa aja", {"exclamation_level": "low"}),
        ("seru banget!", {"exclamation_level": "medium"}),
        ("asik banget!!!", {"exclamation_level": "high"}),
        
        # Repetition & emoji
        ("sedihhhh bangettt 😔", {"uses_repetition": True, "uses_emoji": True}),
        ("seneng banget", {"uses_repetition": False, "uses_emoji": False}),
        
        # Slang detection
        ("anjir kesel banget sih", {"slang_words": ["anjir", "banget", "sih"]}),
        ("wkwk lucu banget deh", {"slang_words": ["wkwk", "banget", "deh"]})
    ]
    
    style_correct = 0
    style_total = 0
    
    print(f"Testing {len(style_tests)} style analysis cases...")
    
    for text, expected_features in style_tests:
        style_analysis = chatbot.analyze_user_style(text)
        
        print(f"\nInput: '{text}'")
        
        for feature, expected_value in expected_features.items():
            actual_value = style_analysis.get(feature)
            
            if feature == "slang_words":
                expected_set = set(expected_value)
                actual_set = set(actual_value or [])
                is_correct = expected_set.issubset(actual_set)
                print(f"   Expected slang: {expected_value}")
                print(f"   Detected slang: {actual_value}")
            else:
                is_correct = actual_value == expected_value
            
            if is_correct:
                style_correct += 1
            style_total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"   {status} {feature}: {actual_value}")
    
    style_accuracy = style_correct / style_total if style_total > 0 else 0
    print(f"\n📊 STYLE ANALYSIS ACCURACY: {style_accuracy:.1%} ({style_correct}/{style_total})")
    
    # ==========================================
    # TEST 3: EMPATHY PROGRESSION & MEMORY
    # ==========================================
    print(f"\n{'='*60}")
    print("💝 TEST 3: EMPATHY PROGRESSION & MEMORY")
    print("=" * 60)
    
    # Reset for clean conversation test
    chatbot.reset_chat_session()
    
    conversation_flow = [
        "hai",
        "aku sedih banget hari ini",
        "iya rasanya berat banget",
        "kayaknya hidup ini susah banget ya",
        "tapi sekarang udah mulai baikan sih",
        "terima kasih udah dengerin aku"
    ]
    
    print("Testing conversation flow and empathy progression...")
    
    empathy_levels = []
    transitions = []
    
    for i, message in enumerate(conversation_flow, 1):
        result = chatbot.chat(message)
        
        empathy_levels.append(result['empathy_level'])
        if result.get('transition'):
            transitions.append(result['transition'])
        
        print(f"\nTurn {i}: '{message}'")
        print(f"   🤖 Bot: {result['response']}")
        print(f"   😊 Sentiment: {result['sentiment']} ({result['confidence']:.2f})")
        print(f"   💝 Empathy Level: {result['empathy_level']}/3")
        
        if result.get('transition'):
            print(f"   🔄 Transition: {result['transition']}")
        
        # Show style analysis for interesting cases
        if i in [2, 5]:  # Show for turns with interesting style
            style = result['style_analysis']
            print(f"   🎭 Style: {style.get('pronouns')} pronouns, {len(style.get('slang_words', []))} slang")
    
    # Check empathy progression
    empathy_progression = len(set(empathy_levels)) > 1
    has_transitions = len(transitions) > 0
    
    print(f"\n📊 EMPATHY & MEMORY RESULTS:")
    print(f"   ✅ Empathy progression: {empathy_progression} (levels: {empathy_levels})")
    print(f"   ✅ Emotion transitions: {has_transitions} (found: {len(transitions)})")
    print(f"   ✅ Memory system: {len(chatbot.short_term_memory)} items stored")
    
    # ==========================================
    # TEST 4: API ENDPOINTS
    # ==========================================
    print(f"\n{'='*60}")
    print("🌐 TEST 4: API ENDPOINTS")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8000"
    api_results = []
    
    # Test endpoints
    endpoints_to_test = [
        ("GET", "/", "Root endpoint"),
        ("GET", "/health", "Health check"),
        ("GET", "/stats", "Stats endpoint")
    ]
    
    print("Testing API endpoints...")
    
    for method, endpoint, description in endpoints_to_test:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ {method} {endpoint} - {description}")
                api_results.append(True)
            else:
                print(f"   ❌ {method} {endpoint} - HTTP {response.status_code}")
                api_results.append(False)
                
        except Exception as e:
            print(f"   ❌ {method} {endpoint} - Error: {str(e)[:50]}...")
            api_results.append(False)
    
    # Test chat endpoint with payload
    try:
        chat_payload = {
            "message": "aku seneng banget hari ini 😊",
            "user_id": "test_user"
        }
        
        response = requests.post(f"{base_url}/chat", json=chat_payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ POST /chat - Chat endpoint")
            print(f"      Response: {data['response'][:40]}...")
            print(f"      Sentiment: {data['sentiment']} ({data['confidence']:.2f})")
            print(f"      Research data keys: {len(data.get('research_data', {}))}")
            api_results.append(True)
        else:
            print(f"   ❌ POST /chat - HTTP {response.status_code}")
            api_results.append(False)
            
    except Exception as e:
        print(f"   ❌ POST /chat - Error: {str(e)[:50]}...")
        api_results.append(False)
    
    api_success_rate = sum(api_results) / len(api_results) if api_results else 0
    print(f"\n📊 API ENDPOINTS SUCCESS: {api_success_rate:.1%} ({sum(api_results)}/{len(api_results)})")
    
    # ==========================================
    # FINAL RESULTS & RECOMMENDATIONS
    # ==========================================
    print(f"\n{'='*60}")
    print("🎉 FINAL RESULTS & RECOMMENDATIONS")
    print("=" * 60)
    
    # Calculate overall scores
    scores = {
        'Sentiment Accuracy': sentiment_accuracy,
        'Style Analysis': style_accuracy,
        'Empathy Progression': 1.0 if empathy_progression else 0.5,
        'Memory System': 1.0 if len(chatbot.short_term_memory) > 0 else 0.0,
        'API Endpoints': api_success_rate
    }
    
    print("\n📊 COMPONENT SCORES:")
    for component, score in scores.items():
        status = "🎉" if score >= 0.8 else "👍" if score >= 0.6 else "⚠️"
        print(f"   {status} {component}: {score:.1%}")
    
    overall_score = sum(scores.values()) / len(scores)
    
    print(f"\n🎯 OVERALL SYSTEM SCORE: {overall_score:.1%}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if overall_score >= 0.8:
        print("   🚀 EXCELLENT! Ready for production deployment!")
        print("   ✅ Flutter integration: GO")
        print("   ✅ Research data collection: GO") 
        print("   ✅ User testing: GO")
        
    elif overall_score >= 0.6:
        print("   👍 GOOD! System is functional with minor improvements needed")
        print("   ✅ Development testing: GO")
        print("   ⚠️  Production: Monitor closely")
        
        # Specific recommendations
        if sentiment_accuracy < 0.7:
            print("   🔧 Consider fine-tuning sentiment model")
        if style_accuracy < 0.7:
            print("   🔧 Improve style analysis rules")
        if api_success_rate < 0.8:
            print("   🔧 Check server stability")
            
    else:
        print("   ⚠️  NEEDS WORK! Address issues before deployment")
        print("   ❌ Production: NOT READY")
        
        # Identify main issues
        problem_areas = [k for k, v in scores.items() if v < 0.5]
        if problem_areas:
            print(f"   🔧 Focus on: {', '.join(problem_areas)}")
    
    print(f"\n✨ Test completed! System is {'READY' if overall_score >= 0.7 else 'NEEDS WORK'}")
    
    return overall_score >= 0.7

if __name__ == "__main__":
    success = comprehensive_test()
    exit(0 if success else 1)