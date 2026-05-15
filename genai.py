import google.genai as genai
client = genai.Client(api_key="AIzaSyBqir0zCnZc8A0rLFNrYfFhi0xEr3K5P2I")
for m in client.models.list():
    # Use supported_actions instead of supported_methods
    if 'embedContent' in m.supported_actions:
        print(f"Model Name: {m.name}")