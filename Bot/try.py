import asyncio
from search import call_perplexity

async def main():
    prompt = """


Provide historical year-over-year percent changes in US federal government employees (series CES9091000001) over the past decade; evaluate how Trump’s hiring freeze and return-to-office mandates could affect the May 2025 reading

"""
    result = await call_perplexity(prompt)
    print("Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())