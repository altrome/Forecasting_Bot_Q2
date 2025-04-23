from llm_calls import call_gpt_o4_mini
import asyncio

async def f():
    ans = await call_gpt_o4_mini("Java or python?")
    print(ans)
    return ans

asyncio.run(f())