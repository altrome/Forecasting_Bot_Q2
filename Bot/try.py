import asyncio
from search import call_perplexity

async def main():
    prompt = "What is the capital of France?"
    result = await call_perplexity(prompt)
    print("Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())