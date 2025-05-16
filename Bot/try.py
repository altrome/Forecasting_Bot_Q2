import asyncio
query = """

    You are a research assistant to a superforecaster and your task is to gather all relevant information needed by the forecaster.

    The question the superforecaster is currently handling is:

    How many citations will Nassim Taleb have on June 28, 2025?

    Resolution Criteria
    This question resolves as the total number of citations ("All") for Nassim Nicholas Taleb presented by Google Scholar when checked by Metaculus on or after June 28, 2025?

    Background Info
    Nassim Taleb is author of The Black Swan and Antifragile and served as an advisor to the Good Judgment Project. See work co-authored with Phil Tetlock: On the Difference between Binary Prediction and True Exposure With Implications For Forecasting Tournaments and Decision Making Research

    Be as extensive and thorough in your search for information as possible. Present all relevant information objectively, without making any forecasts of your own. Pay special attention to the following:

        - Historically relevant reference classes and data that would be useful to establish a reasonable base rate.
        - Timeframe to resolution, find information on previous rates of change pertaining to the variable of interest over similar timeframes.
        - Specific information about the current state from the resolution source.
        - All relevant current news articles pertaining to the question.
        - Statements from experts, government officials or other public figures.

    Cite your sources whenever possible.

    """

from agents import Agent, Runner
from agents import WebSearchTool

async def main():
    # 1. Define your agent with o3 and the WebSearchTool (max 5 uses)
    agent = Agent(
        name="WebSearchAgent",
        instructions=(
            "You are a helpful assistant with web search capabilities. "
            "When you need more information, perform up to 5 web searches."
        ),
        model="o3",
        tools=[WebSearchTool()]
    )

    # 2. Your single user prompt
    prompt = query

    # 3. Invoke the agent
    result = await Runner.run(agent, prompt)

    # 4. Output the final answer and tool‐usage info
    print("Agent says:\n", result.final_output)
    print(f"Web searches used: {result.tool_calls.get('web_search', 0)}")

if __name__ == "__main__":
    asyncio.run(main())

