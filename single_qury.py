import requests
import json
import time

def test_single_query():
    """Test with just one question for fastest performance"""
    
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 8c8cb782ead692a2056fd1494ecf1ee2ebd90d05d4d4dbd8e25bcfb03355bf00",
        "Content-Type": "application/json"
    }

    # Test with just ONE question
    test_data = {
        "documents": "https://hackrx.in/policies/BAJHLIP23020V012223.pdf",
        "questions": [
            "What is the grace period for premium payment?"
        ]
    }

    print("🧪 Testing Single Query Processing...")
    print("=" * 50)
    print(f"📄 Document: Policy PDF")
    print(f"❓ Single Question: {test_data['questions'][0]}")
    print("=" * 50)

    start_time = time.time()
    
    try:
        print("🚀 Sending request...")
        response = requests.post(url, headers=headers, json=test_data, timeout=30)
        processing_time = time.time() - start_time
        
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"\n🔍 QUESTION: {test_data['questions'][0]}")
            print(f"💡 ANSWER: {result['answers'][0]}")
            
            # Performance check
            if processing_time < 30:
                print(f"\n🏆 PERFORMANCE: EXCELLENT (Under 30s requirement)")
            else:
                print(f"\n⚠️ PERFORMANCE: Needs optimization ({processing_time:.1f}s)")
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT: Request took longer than 30 seconds")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")

if __name__ == "__main__":
    test_single_query()
