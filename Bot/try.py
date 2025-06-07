import os
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# 1. Read your API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# 2. Initialize the Responses API client
client = OpenAI(api_key=api_key)

# 3. Build the input prompt for an event post–knowledge-cutoff
user_query = "Tell me about the 2024 Nobel Prize in Chemistry awarded in October 2024. Do a web search to find out who won."

# 4. Call the o3 model via the Responses API
response = client.responses.create(
    model="o3",
    input=[{"role": "user", "content": user_query}],
    #mtools=[{"type": "web_search_preview"}]  # uncomment to enable real-time search
)

# 5. Print out the model’s answer
print(response.output_text)