import requests
import json
import time
from datetime import datetime
import sys

class LocalAPITester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def print_header(self, title):
        """Print formatted test header"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_result(self, success, message):
        """Print formatted test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {message}")
    
    def test_server_connection(self):
        """Test if server is running"""
        self.print_header("Server Connection Test")
        
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.print_result(True, f"Server is running - {data.get('status', 'Unknown')}")
                return True
            else:
                self.print_result(False, f"Server responded with status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_result(False, "Cannot connect to server. Is it running on localhost:8000?")
            print("💡 Start server with: python app.py")
            return False
        except Exception as e:
            self.print_result(False, f"Connection error: {e}")
            return False
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        self.print_header("Health Check Test")
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.print_result(True, f"Health status: {data.get('status', 'Unknown')}")
                print(f"   📅 Timestamp: {data.get('timestamp', 'N/A')}")
                print(f"   🔢 Version: {data.get('version', 'N/A')}")
                return True
            else:
                self.print_result(False, f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_result(False, f"Health check error: {e}")
            return False
    
    def test_chat_basic(self):
        """Test basic chat functionality"""
        self.print_header("Basic Chat Test")
        
        test_cases = [
            {
                "message": "Hai!",
                "expected_sentiment": ["neutral", "happy"],
                "description": "Simple greeting"
            },
            {
                "message": "aku sedih banget hari ini",
                "expected_sentiment": ["sadness"],
                "description": "Sadness expression with Indonesian"
            },
            {
                "message": "anjirrrr kesel banget sama temen!!!",
                "expected_sentiment": ["anger"],
                "description": "Anger with slang and repetition"
            },
            {
                "message": "sekarang aku seneng sih 😊",
                "expected_sentiment": ["happy"],
                "description": "Happy with emoji"
            }
        ]
        
        all_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {test_case['description']}")
            print(f"   Input: '{test_case['message']}'")
            
            try:
                payload = {
                    "message": test_case["message"],
                    "user_id": "test_user",
                    "reset_session": i == 1  # Reset on first message
                }
                
                response = self.session.post(
                    f"{self.base_url}/chat",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check response structure
                    required_fields = ['response', 'sentiment', 'confidence', 'research_data']
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        self.print_result(False, f"Missing fields: {missing_fields}")
                        all_passed = False
                        continue
                    
                    # Check sentiment
                    sentiment = data['sentiment']
                    sentiment_correct = sentiment in test_case['expected_sentiment']
                    
                    self.print_result(sentiment_correct, 
                        f"Sentiment: {sentiment} ({'Expected' if sentiment_correct else 'Unexpected'})")
                    
                    print(f"   🤖 Response: {data['response']}")
                    print(f"   📊 Confidence: {data['confidence']:.2f}")
                    print(f"   📈 Empathy Level: {data.get('empathy_level', 'N/A')}")
                    print(f"   🔄 Turn Count: {data.get('turn_count', 'N/A')}")
                    
                    if data.get('transition'):
                        print(f"   🔄 Transition: {data['transition']}")
                    
                    if not sentiment_correct:
                        all_passed = False
                    
                else:
                    self.print_result(False, f"HTTP {response.status_code}: {response.text}")
                    all_passed = False
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.print_result(False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    def test_research_data(self):
        """Test research data completeness"""
        self.print_header("Research Data Test")
        
        try:
            payload = {
                "message": "gue bingung banget sama perasaan gue sekarang...",
                "user_id": "research_test"
            }
            
            response = self.session.post(f"{self.base_url}/chat", json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                research_data = data.get('research_data', {})
                
                # Check required research fields
                required_fields = [
                    'style_analysis',
                    'personality_profile', 
                    'short_term_memory',
                    'special_case'
                ]
                
                all_present = True
                for field in required_fields:
                    if field in research_data:
                        self.print_result(True, f"Research field '{field}' present")
                    else:
                        self.print_result(False, f"Research field '{field}' missing")
                        all_present = False
                
                # Check style analysis details
                style_analysis = research_data.get('style_analysis', {})
                style_fields = ['pronouns', 'exclamation_level', 'slang_words']
                
                print(f"\n📊 Style Analysis Details:")
                for field in style_fields:
                    value = style_analysis.get(field, 'N/A')
                    print(f"   • {field}: {value}")
                
                # Check personality profile
                personality = research_data.get('personality_profile', {})
                print(f"\n👤 Personality Profile:")
                for key, value in personality.items():
                    print(f"   • {key}: {value}")
                
                return all_present
            else:
                self.print_result(False, f"Research data test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.print_result(False, f"Research data test error: {e}")
            return False
    
    def test_stats_endpoint(self):
        """Test stats endpoint"""
        self.print_header("Stats Endpoint Test")
        
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                expected_fields = [
                    'turn_count', 'personality_profile', 'memory_size',
                    'confidence_threshold', 'available_sentiments'
                ]
                
                all_present = True
                for field in expected_fields:
                    if field in data:
                        self.print_result(True, f"Stats field '{field}': {data[field]}")
                    else:
                        self.print_result(False, f"Stats field '{field}' missing")
                        all_present = False
                
                return all_present
            else:
                self.print_result(False, f"Stats endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_result(False, f"Stats endpoint error: {e}")
            return False
    
    def test_reset_endpoint(self):
        """Test reset endpoint"""
        self.print_header("Reset Endpoint Test")
        
        try:
            payload = {"user_id": "test_user"}
            response = self.session.post(f"{self.base_url}/reset", json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.print_result(True, f"Reset successful: {data.get('message', 'N/A')}")
                return True
            else:
                self.print_result(False, f"Reset failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_result(False, f"Reset error: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        self.print_header("Error Handling Test")
        
        # Test empty message
        try:
            payload = {"message": "", "user_id": "test"}
            response = self.session.post(f"{self.base_url}/chat", json=payload, timeout=10)
            
            if response.status_code == 422:
                self.print_result(True, "Empty message properly rejected (422)")
            else:
                self.print_result(False, f"Empty message not properly handled: {response.status_code}")
                return False
        except Exception as e:
            self.print_result(False, f"Error handling test failed: {e}")
            return False
        
        # Test invalid JSON
        try:
            response = self.session.post(f"{self.base_url}/chat", data="invalid json", timeout=10)
            
            if response.status_code in [400, 422]:
                self.print_result(True, f"Invalid JSON properly rejected ({response.status_code})")
            else:
                self.print_result(False, f"Invalid JSON not properly handled: {response.status_code}")
                return False
        except Exception as e:
            self.print_result(False, f"Invalid JSON test failed: {e}")
            return False
        
        return True
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"🚀 RasaChatbot API Local Testing")
        print(f"🌐 Testing URL: {self.base_url}")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        tests = [
            ("Server Connection", self.test_server_connection),
            ("Health Check", self.test_health_endpoint),
            ("Basic Chat", self.test_chat_basic),
            ("Research Data", self.test_research_data),
            ("Stats Endpoint", self.test_stats_endpoint),
            ("Reset Endpoint", self.test_reset_endpoint),
            ("Error Handling", self.test_error_handling)
        ]
        
        passed = 0
        total = len(tests)
        failed_tests = []
        
        for test_name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed_tests.append(test_name)
            except Exception as e:
                print(f"❌ FAIL: {test_name} - Unexpected error: {e}")
                failed_tests.append(test_name)
        
        # Final results
        print(f"\n{'='*60}")
        print(f"📊 TEST RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Passed: {passed}/{total}")
        print(f"❌ Failed: {total - passed}/{total}")
        
        if failed_tests:
            print(f"\n🔍 Failed Tests:")
            for test in failed_tests:
                print(f"   • {test}")
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED! 🎉")
            print(f"✅ Your API is working correctly!")
            print(f"🚀 Ready for deployment!")
        else:
            print(f"\n⚠️  Some tests failed. Please check the issues above.")
            print(f"💡 Make sure:")
            print(f"   • Server is running: python app.py")
            print(f"   • Environment variables are set (.env file)")
            print(f"   • All dependencies are installed")
        
        print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return passed == total

def main():
    """Main function"""
    # Check if custom URL provided
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
        print(f"🌐 Using custom URL: {base_url}")
    else:
        base_url = "http://localhost:8000"
        print(f"🏠 Using default local URL: {base_url}")
    
    tester = LocalAPITester(base_url)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()