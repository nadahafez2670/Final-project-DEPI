import google.generativeai as genai

# 1. PASTE YOUR NEW KEY HERE (Keep the quotes!)
MY_API_KEY = "AIzaSyDij_OLwuQTuG5ePUct0gqTMPmTCV9leqc"

# 2. Configure the library
genai.configure(api_key=MY_API_KEY)

print(f"Testing key: {MY_API_KEY[:10]}... (hidden)")

try:
    print("\nAttempting to list available models...")
    # 3. List models to verify connection
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f" - Found: {m.name}")
    print("\n✅ SUCCESS! The API Key is working.")

except Exception as e:
    print(f"\n❌ ERROR: {e}")