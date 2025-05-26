from llm_calls import call_claude
import asyncio

async def main():
    output = await call_claude('What is your name')
    print(output)

if __name__ == "__main__":
    asyncio.run(main())