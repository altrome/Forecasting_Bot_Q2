import asyncio
import time
import numpy as np
from numeric import get_numeric_forecast
from binary import get_binary_forecast
import os
import forecasting_tools
import requests
from aiohttp import ClientSession, ClientTimeout, ClientError
import json
import sys
import re
import io
from multiple_choice import get_multiple_choice_forecast
from asknews_sdk import AskNewsSDK
from prompts import context
from dotenv import load_dotenv
import aiohttp
from prompts import claude_context, gpt_context
"""
This file contains the main forecasting logic, question-type specific functions are abstracted.
"""

load_dotenv()

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
EXA_API_KEY = os.getenv("EXA_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # You'll also need the OpenAI API Key if you want to use the Exa Smart Searcher

async def numeric_forecast(question_details, write=print):
    return await get_numeric_forecast(question_details, write)

async def binary_forecast(question_details, write=print):
    return await get_binary_forecast(question_details, write)

async def multiple_choice_forecast(question_details, write=print):
    return await get_multiple_choice_forecast(question_details, write)


def call_exa_smart_searcher(question: str) -> str:
    if OPENAI_API_KEY is None:
        searcher = forecasting_tools.ExaSearcher(
            include_highlights=True,
            num_results=10,
        )
        highlights = asyncio.run(searcher.invoke_for_highlights_in_relevance_order(question))
        prioritized_highlights = highlights[:10]
        combined_highlights = ""
        for i, highlight in enumerate(prioritized_highlights):
            combined_highlights += f'[Highlight {i+1}]:\nTitle: {highlight.source.title}\nURL: {highlight.source.url}\nText: "{highlight.highlight_text}"\n\n'
        response = combined_highlights
    else:
        searcher = forecasting_tools.SmartSearcher(
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )
        response = asyncio.run(searcher.invoke(prompt))

    return response



