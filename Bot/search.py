
import asyncio
from typing import List, Dict
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FastContentExtractor import FastContentExtractor
import dateparser
from dotenv import load_dotenv
import json
import os
from aiohttp import ClientSession, ClientTimeout
from asknews_sdk import AskNewsSDK
from prompts import context
from dotenv import load_dotenv
import aiohttp
import re
import random
import time
from openai import OpenAI   
import traceback
load_dotenv()

SERPER_KEY = os.getenv("SERPER_KEY")
ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

assistant_prompt = """

You are an assistant to a superforecaster and your task involves high-quality information retrieval to help the forecaster make the most informed forecasts. Forecasting involves parsing through an immense trove of internet articles and web content. To make this easier for the forecaster, you read entire articles and extract the key pieces of the articles relevant to the question. The key pieces generally include:

1. Facts, statistics and other objective measurements described in the article
2. Opinions from reliable and named sources (e.g. if the article writes 'according to a 2023 poll by Gallup' or 'The 2025 presidential approval rating poll by Reuters' etc.)
3. Potentially useful opinions from less reliable/not-named sources (you explicitly document the less reliable origins of these opinions though)

Today, you're focusing on the question:

{title}

Resolution criteria:
{resolution_criteria}

Fine print:
{fine_print}

Background information:
{background}

Article to summarize:
{article}

Note: If the web content extraction is incomplete or you believe the quality of the extracted content isn't the best, feel free to add a disclaimer before your summary.

Please summarize only the article given, not injecting your own knowledge or providing a forecast. Aim to achieve a balance between a superficial summary and an overly verbose account. 

"""

def write(x):
    print(x)

def parse_date(date_str: str) -> str:
    parsed_date = dateparser.parse(date_str, settings={'STRICT_PARSING': False})
    if parsed_date:
        return parsed_date.strftime("%b %d, %Y")
    return "Unknown"

def validate_time(before_date_str, source_date_str):
    if source_date_str == "Unknown":
        return False
    before_date = dateparser.parse(before_date_str)
    source_date = dateparser.parse(source_date_str)
    return source_date <= before_date

# new helper: takes raw article text + the question_details dict
async def summarize_article(article: str, question_details: dict) -> str:
    prompt = assistant_prompt.format(
        title=question_details["title"],
        resolution_criteria=question_details["resolution_criteria"],
        fine_print=question_details["fine_print"],
        background=question_details["description"],
        article=article
    )
    return await call_gpt(prompt)


# Shared HTTP session
_http_session: ClientSession = None
async def get_session(timeout: int = 60) -> ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = ClientSession(timeout=ClientTimeout(total=timeout))
    return _http_session

# Concurrency limiter for external calls
HTTP_CONCURRENCY_SEMAPHORE = asyncio.Semaphore(5)

# Retry decorator/helper
async def with_retries(fn, *args, retries: int = 3, backoff: float = 3.0, **kwargs):
    for attempt in range(1, retries + 1):
        try:
            return await fn(*args, **kwargs)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == retries:
                raise
            wait = backoff * (2 ** (attempt - 1))
            print(f"[with_retries] Attempt {attempt} failed: {e}. Retrying in {wait:.1f}s...")
            await asyncio.sleep(wait)

# Utility functions

def parse_date(date_str: str) -> str:
    parsed = dateparser.parse(date_str, settings={'STRICT_PARSING': False})
    return parsed.strftime("%b %d, %Y") if parsed else "Unknown"


def validate_time(before_date_str: str, source_date_str: str) -> bool:
    if source_date_str == "Unknown":
        return False
    before = dateparser.parse(before_date_str)
    src = dateparser.parse(source_date_str)
    return src <= before

# Perplexity API call
async def _call_perplexity_once(prompt: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar-deep-research",
        "messages": [
            {"role": "system", "content": "Be thorough and detailed. Cite all sources with names and dates."},
            {"role": "user",   "content": prompt + " Cite all sources with names and dates."}
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }
    session = await get_session(timeout=60)
    async with HTTP_CONCURRENCY_SEMAPHORE, session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        # Stream and accumulate
        chunks = []
        async for chunk in resp.content.iter_chunked(1024):
            chunks.append(chunk.decode())
        content = ''.join(chunks)
        # Strip internal think tags
        return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

async def call_perplexity(prompt: str) -> str:
    return await with_retries(_call_perplexity_once, prompt)

# Google/Serper search with date filtering
async def _google_search_once(query: str, is_news: bool, date_before: str = None) -> List[str]:
    path = 'news' if is_news else 'search'
    url = f"https://google.serper.dev/{path}"
    payload = {"q": query, "num": 20}
    headers = {'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json'}
    session = await get_session(timeout=30)
    async with HTTP_CONCURRENCY_SEMAPHORE, session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        data = await resp.json()
        items = data.get('news' if is_news else 'organic', [])
        urls: List[str] = []
        for item in items:
            if len(urls) >= 12:
                break
            link = item.get('link')
            date_raw = item.get('date', '')
            date_str = parse_date(date_raw)
            if date_before and date_str != "Unknown":
                if not validate_time(date_before, date_str):
                    continue
            if link:
                urls.append(link)
        return urls


async def google_search(query: str, is_news: bool = False, date_before: str = None) -> List[str]:
    # Clean query
    clean_q = query.replace('"', '').replace("'", '').strip()
    return await with_retries(_google_search_once, clean_q, is_news)

# AskNews call
async def _call_asknews_once(question: str) -> str:
    ask = AskNewsSDK(client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes={'news'})
    hot = await asyncio.to_thread(ask.news.search_news, query=question, n_articles=8, return_type='both', strategy='latest news')
    hist = await asyncio.to_thread(ask.news.search_news, query=question, n_articles=8, return_type='both', strategy='news knowledge')
    def format_list(lst):
        entries = sorted(lst.as_dicts, key=lambda a: a['pub_date'], reverse=True)
        text = ''
        for art in entries:
            pd = art['pub_date'].strftime('%B %d, %Y %I:%M %p')
            text += f"**{art['eng_title']}**\n{art['summary']}\n{pd}\n[{art['source_id']}]({art['article_url']})\n\n"
        return text or 'No articles found.'
    return "Hot Articles:\n" + format_list(hot) + "\nHistorical Articles:\n" + format_list(hist)

async def call_asknews(question: str) -> str:
    return await with_retries(_call_asknews_once, question)


async def call_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.responses.create(
            model="o4-mini",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        write(f"[call_gpt] Error: {str(e)}")
        return f"Error calling OpenAI API: {str(e)}"


async def google_search_and_scrape(query, is_news, question_details, date_before=None):
    write(f"[google_search_and_scrape] Called with query='{query}', is_news={is_news}, date_before={date_before}")
    try:
        urls = await google_search(query, is_news, date_before)

        if not urls:
            write(f"[google_search_and_scrape] ❌ No URLs returned for query: '{query}'")
            return f"<Summary query=\"{query}\">No URLs returned from Google.</Summary>\n"

        async with FastContentExtractor() as extractor:
            write(f"[google_search_and_scrape] 🔍 Starting content extraction for {len(urls)} URLs")
            results = await extractor.extract_content(urls)
            write(f"[google_search_and_scrape] ✅ Finished content extraction")

        summarize_tasks = []
        no_results = 3
        valid_urls = []
        for url, data in results.items():
            if len(summarize_tasks) >= no_results:
                break  
            content = (data.get('content') or '').strip()
            if len(content.split()) < 100:
                write(f"[google_search_and_scrape] ⚠️ Skipping low-content article: {url}")
                continue
            if content:
                truncated = content[:8000]
                write(f"[google_search_and_scrape] ✂️ Truncated content for summarization: {len(truncated)} chars from {url}")
                summarize_tasks.append(
                    asyncio.create_task(summarize_article(truncated, question_details))
                )
                valid_urls.append(url)
            else:
                write(f"[google_search_and_scrape] ⚠️ No content for {url}, skipping summarization.")

        if not summarize_tasks:
            write("[google_search_and_scrape] ⚠️ Warning: No content to summarize")
            return f"<Summary query=\"{query}\">No usable content extracted from any URL.</Summary>\n"

        summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

        output = ""
        for url, summary in zip(valid_urls, summaries):
            if isinstance(summary, Exception):
                write(f"[google_search_and_scrape] ❌ Error summarizing {url}: {summary}")
                output += f"\n<Summary source=\"{url}\">\nError summarizing content: {str(summary)}\n</Summary>\n"
            else:
                output += f"\n<Summary source=\"{url}\">\n{summary}\n</Summary>\n"

        return output
    except Exception as e:
        write(f"[google_search_and_scrape] Error: {str(e)}")
        traceback_str = traceback.format_exc()
        write(f"Traceback: {traceback_str}")
        return f"<Summary query=\"{query}\">Error during search and scrape: {str(e)}</Summary>\n"


async def process_search_queries(response: str, forecaster_id: str, question_details: dict):
    """
    Parses out search queries from the forecaster's response, executes them
    (AskNews, Perplexity or Google/Google News), and returns formatted summaries.
    """
    try:
        # 1) Extract the "Search queries:" block
        search_queries_block = re.search(r'(?:Search queries:)(.*)', response, re.DOTALL | re.IGNORECASE)
        if not search_queries_block:
            write(f"Forecaster {forecaster_id}: No search queries block found")
            return ""

        queries_text = search_queries_block.group(1).strip()

        # 2) Try to find queries of the form: 1. "text" (Source)
        search_queries = re.findall(
            r'(?:\d+\.\s*)?(["\']?(.*?)["\']?)\s*\((Google|Google News|Assistant|Perplexity)\)',
            queries_text
        )
        # 3) Fallback to unquoted queries if none found
        if not search_queries:
            search_queries = re.findall(
                r'(?:\d+\.\s*)?([^(\n]+)\s*\((Google|Google News|Assistant|Perplexity)\)',
                queries_text
            )

        if not search_queries:
            write(f"Forecaster {forecaster_id}: No valid search queries found:\n{queries_text}")
            return ""

        write(f"Forecaster {forecaster_id}: Processing {len(search_queries)} search queries")

        # 4) Kick off one asyncio task per query
        tasks = []
        query_sources = []  # Track which source goes with which task
        
        for match in search_queries:
            # match can be ("\"text\"", "text", "Source") or ("text", "Source")
            if len(match) == 3:
                _, raw_query, source = match
            else:
                raw_query, source = match

            query = raw_query.strip().strip('"').strip("'")
            if not query:
                continue

            write(f"Forecaster {forecaster_id}: Query='{query}' Source={source}")
            query_sources.append((query, source))

            if source in ("Google", "Google News"):
                # pass question_details through so summarizer can fill the prompt
                tasks.append(
                    google_search_and_scrape(
                        query,
                        is_news=(source == "Google News"),
                        question_details=question_details,
                        date_before=question_details.get("resolution_date")
                    )
                )
            elif source == "Assistant":
                tasks.append(call_asknews(query))
            else:  # Perplexity
                tasks.append(call_perplexity(query))  # Now directly call as async

        if not tasks:
            write(f"Forecaster {forecaster_id}: No tasks generated")
            return ""

        # 5) Await all tasks one by one to avoid timeout issues
        formatted_results = ""
        
        # First gather with return_exceptions=True to prevent one failure from breaking everything
        results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # 6) Format the outputs
        for (query, source), result in zip(query_sources, results):
            if isinstance(result, Exception):
                write(f"[process_search_queries] ❌ Forecaster {forecaster_id}: Error for '{query}' → {str(result)}")
                # Add a message about the error in the formatted results
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\nError retrieving results: {str(result)}\n</Asknews_articles>\n"
                elif source == "Perplexity":
                    formatted_results += f"\n<Perplexity_report>\nQuery: {query}\nError retrieving results: {str(result)}\n</Perplexity_report>\n"
                else:
                    formatted_results += f"\n<Summary query=\"{query}\">\nError retrieving results: {str(result)}\n</Summary>\n"
            else:
                write(f"[process_search_queries] ✅ Forecaster {forecaster_id}: Query '{query}' processed successfully.")
                
                if source == "Assistant":
                    formatted_results += f"\n<Asknews_articles>\nQuery: {query}\n{result}</Asknews_articles>\n"
                elif source == "Perplexity":
                    formatted_results += f"\n<Perplexity_report>\nQuery: {query}\n{result}</Perplexity_report>\n"
                else:
                    # Google/Google News tasks already return <Summary> blocks
                    formatted_results += result

        return formatted_results

    except Exception as e:
        write(f"Forecaster {forecaster_id}: Error processing search queries: {str(e)}")
        write(f"Traceback: {traceback.format_exc()}")
        # Return what we have so far instead of nothing
        return "Error processing some search queries. Partial results may be available."

async def main():
    """
    Demonstrates the usage of process_search_queries with sample search queries.
    """
    print("Starting test for content extraction...")
    
    # This part won't be executed
    sample_response = """
    Search queries:
    1. "Nvidia stock price forecast 2024" (Google)
    2. "Ukraine Russia conflict latest developments" (Google News)
    3. "Middle East stability assessment Israel Hamas" (Perplexity)
    4. "Trump tariffs economic impact" (Assistant)
    """
    
    forecaster_id = "demo_forecaster"
    print(f"Processing sample search queries for forecaster: {forecaster_id}")
    
    results = await process_search_queries(sample_response, forecaster_id)
    
    print("\n=== SEARCH RESULTS ===\n")
    print(results)
    print("\n=== END OF RESULTS ===\n")

if __name__ == "__main__":
    asyncio.run(main())