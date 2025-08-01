import requests
import json
import time
from datetime import datetime

class ProductionAPITester:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def test_health(self):
        """Test health endpoint"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_chat_basic(self):
        """Test basic chat functionality"""
        print("\n🔍 Testing basic chat...")
        
        test_messages = [
            "Hai!",
            "aku sedih banget hari ini",
            "kenapa ya aku nggak bisa bahagia",
            "tapi sekarang aku seneng sih 😊"
        ]
        
        for i, message in enumerate(test_messages, 1):
            try:
                payload = {
                    "message": message,
                    "user_id": "test_user",
                    "reset_session": i == 1  # Reset on first message
                }
                
                print(f"  Test {i}: '{message}'")
                response = self.session.post(
                    f"{self.base_url}/chat",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"    ✅ Response: {data['response']}")
                    print(f"    📊 Sentiment: {data['sentiment']} ({data['confidence']:.2f})")
                    if data.get('transition'):
                        print(f"    🔄 Transition: {data['transition']}")
                else:
                    print(f"    ❌ Failed: {response.status_code} - {response.text}")
                    return False
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                return False
        
        print("✅ Basic chat tests passed!")
        return True
    
    def test_research_data(self):
        """Test research data output"""
        print("\n🔍 Testing research data...")
        
        try:
            payload = {
                "message": "anjirrrr kesel banget sama temen gue!!!",
                "user_id": "research_test"
            }
            
            response = self.session.post(
                f"{self.base_url}/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
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
                
                for field in required_fields:
                    if field in research_data:
                        print(f"    ✅ {field}: Present")
                    else:
                        print(f"    ❌ {field}: Missing")
                        return False
                
                # Check style analysis details
                style_analysis = research_data.get('style_analysis', {})
                print(f"    📊 Style Analysis: {style_analysis}")
                
                print("✅ Research data test passed!")
                return True
            else:
                print(f"❌ Research data test failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Research data test error: {e}")
            return False
    
    def test_stats_endpoint(self):
        """Test stats endpoint"""
        print("\n🔍 Testing stats endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"    ✅ Stats retrieved: {data}")
                return True
            else:
                print(f"    ❌ Stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"    ❌ Stats error: {e}")
            return False
    
    def test_reset_endpoint(self):
        """Test reset endpoint"""
        print("\n🔍 Testing reset endpoint...")
        
        try:
            payload = {"user_id": "test_user"}
            response = self.session.post(
                f"{self.base_url}/reset",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print("    ✅ Reset successful")
                return True
            else:
                print(f"    ❌ Reset failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"    ❌ Reset error: {e}")
            return False
    
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print(f"🚀 Testing API at: {self.base_url}")
        print(f"⏰ Started at: {datetime.now()}\n")
        
        tests = [
            ("Health Check", self.test_health),
            ("Basic Chat", self.test_chat_basic),
            ("Research Data", self.test_research_data),
            ("Stats Endpoint", self.test_stats_endpoint),
            ("Reset Endpoint", self.test_reset_endpoint)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"{'='*50}")
            print(f"Running: {test_name}")
            if test_func():
                passed += 1
            print()
        
        print(f"{'='*50}")
        print(f"📊 Test Results: {passed}/{total} passed")
        
        if passed == total:
            print("🎉 All tests passed! API is working correctly!")
        else:
            print("❌ Some tests failed. Please check the issues above.")
        
        print(f"⏰ Completed at: {datetime.now()}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python production_test.py <API_URL>")
        print("Example: python production_test.py https://your-app.onrender.com")
        sys.exit(1)
    
    api_url = sys.argv[1]
    tester = ProductionAPITester(api_url)
    tester.run_all_tests()

if __name__ == "__main__":
    main()