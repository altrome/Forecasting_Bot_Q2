from llm_calls import call_gpt_o3
import asyncio

async def f():
    x = await call_gpt_o3('What is your specialty?')
    print(x)
    return x

if __name__ == "__main__":
    asyncio.run(f())