import requests
import json
import time

def test_pdf_processing():
    """Test PDF processing with the HackRx intelligent query system"""
    
    # Your local API endpoint
    url = "http://localhost:8000/hackrx/run"
    
    # Headers with the required authentication token
    headers = {
        "Authorization": "Bearer 8c8cb782ead692a2056fd1494ecf1ee2ebd90d05d4d4dbd8e25bcfb03355bf00",
        "Content-Type": "application/json"
    }

    # Test data with the PDF URL and sample questions
    test_data = {
        "documents": "https://hackrx.in/policies/BAJHLIP23020V012223.pdf",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }

    print("🧪 Testing PDF Processing with HackRx System...")
    print("=" * 60)
    print(f"📄 PDF Document: {test_data['documents'][:60]}...")
    print(f"❓ Number of Questions: {len(test_data['questions'])}")
    print("=" * 60)

    start_time = time.time()
    
    try:
        print("🚀 Sending request to API...")
        response = requests.post(url, headers=headers, json=test_data, timeout=120)
        processing_time = time.time() - start_time
        
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        print("-" * 60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ PDF PROCESSING SUCCESS!")
            print("\n📋 EXTRACTED ANSWERS:")
            print("=" * 80)
            
            for i, (question, answer) in enumerate(zip(test_data['questions'], result['answers']), 1):
                print(f"\n{i}. 🔍 QUESTION: {question}")
                print(f"   💡 ANSWER: {answer}")
                print("-" * 80)
                
            # Success metrics
            print(f"\n✅ SUCCESS METRICS:")
            print(f"   📊 Questions Processed: {len(result['answers'])}/{len(test_data['questions'])}")
            print(f"   ⏱️ Response Time: {processing_time:.2f}s (Target: <30s)")
            print(f"   🎯 API Response: Working correctly")
            
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"📄 Response: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                if 'detail' in error_data:
                    print(f"🔍 Error Details: {error_data['detail']}")
            except:
                pass
            
    except requests.exceptions.Timeout:
        print("⏰ REQUEST TIMED OUT")
        print("   Error: Request took longer than 35 seconds")
        print("   Suggestion: Check if your server is running and processing efficiently")
        
    except requests.exceptions.ConnectionError:
        print("🔌 CONNECTION ERROR")
        print("   Error: Could not connect to the API server")
        print("   Suggestion: Make sure your server is running on http://localhost:8000")
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {str(e)}")
        print("   Check your server logs for more details")

def test_with_custom_url(pdf_url, questions=None):
    """Test with your own PDF URL and questions"""
    
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 8c8cb782ead692a2056fd1494ecf1ee2ebd90d05d4d4dbd8e25bcfb03355bf00",
        "Content-Type": "application/json"
    }
    
    # Default questions if none provided
    if questions is None:
        questions = [
            "What is the main purpose of this document?",
            "What are the key benefits mentioned?",
            "Are there any waiting periods specified?",
            "What are the exclusions mentioned?",
            "What is the coverage amount or limit?"
        ]
    
    test_data = {
        "documents": pdf_url,
        "questions": questions
    }
    
    print(f"🧪 Testing with Custom PDF: {pdf_url[:50]}...")
    print(f"❓ Questions: {len(questions)}")
    
    try:
        response = requests.post(url, headers=headers, json=test_data, timeout=35)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            for i, (q, a) in enumerate(zip(questions, result['answers']), 1):
                print(f"\n{i}. Q: {q}")
                print(f"   A: {a[:200]}{'...' if len(a) > 200 else ''}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_health_check():
    """Test if the API server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Server is running and healthy!")
            return True
        else:
            print(f"⚠️ API Server responded with status: {response.status_code}")
            return False
    except:
        print("❌ API Server is not running or not accessible")
        return False

if __name__ == "__main__":
    print("🚀 HackRx PDF Processing Test Suite")
    print("=" * 50)
    
    # First check if server is running
    print("1. Checking API Health...")
    if test_health_check():
        print("\n2. Testing PDF Processing...")
        test_pdf_processing()
        
        # Uncomment the line below to test with your own PDF URL
        # test_with_custom_url("YOUR_PDF_URL_HERE")
        
    else:
        print("\n❌ Cannot proceed with tests. Please:")
        print("   1. Make sure your server is running: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("   2. Check that all dependencies are installed")
        print("   3. Verify your OpenRouter API key is set in .env file")
