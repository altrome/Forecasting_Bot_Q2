from search import call_asknews
import asyncio

async def f():
    ans = await call_asknews("Fetch latest news article on the Pope.")
    print(ans)
    return ans

asyncio.run(f())