import google.generativeai as genai
 # The script uses the API key you provided earlier.
genai.configure(api_key="AIzaSyCJCok2v54_L1wL8e21SREAfWIWdL1Ilwc")

print("Available models that support 'generateContent':")
for m in genai.list_models():
     if 'generateContent' in m.supported_generation_methods:
         print(m.name)
