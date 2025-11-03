import os
from openai import OpenAI

# BASE_URL = os.getenv("BASE_URL", "https://b1de0f06d9bf.ngrok-free.app/v1")
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")
API_KEY  = os.getenv("API_KEY", "not-needed")  # server may ignore; SDK requires a value
MODEL    = os.getenv("MODEL", "qwen3:32b")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Write me a poem about your mom."},
    ],
    max_tokens=4096,
)
text = response.choices[0].message.content
print("Response:", text)