import openai
openai.api_key = "YOUR_API_KEY"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # Replace with "gpt-4" if desired and accessible
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, world!"}
    ],
    max_tokens=5
)
print(response)
