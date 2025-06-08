import asyncio
from search import agentic_search
import numpy as np
import os
from aiohttp import ClientSession, ClientTimeout, ClientError
from openai import OpenAI

from dotenv import load_dotenv


load_dotenv()
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def call_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview", "search_context_size": "high",}],
        input= prompt
    )
    return response.output_text

async def main():
    query = """

    Please fetch all historical Virgin Galactic News page post timestamps from launch through April 2025, count posts per month, highlight months with zero posts, and identify any gaps longer than two months to inform the expected June 2025 activity. 

    """
    ans = await agentic_search(query)

    print(ans)


if __name__ == "__main__":
    asyncio.run(main())