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
from requests.exceptions import ConnectionError, Timeout
from http.client import RemoteDisconnected
import requests
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


async def call_asknews(question: str) -> str:
    """
    Use the AskNews `news` endpoint to get news context for your query.
    The full API reference can be found here: https://docs.asknews.app/en/reference#get-/v1/news/search
    """
    ask = AskNewsSDK(
        client_id=ASKNEWS_CLIENT_ID, client_secret=ASKNEWS_SECRET, scopes=set(["news"])
    )

    async with aiohttp.ClientSession() as session:
        # Create tasks for both API calls
        hot_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
            query=question,
            n_articles=8,
            return_type="both",
            strategy="latest news"
        ))
        historical_task = asyncio.create_task(asyncio.to_thread(ask.news.search_news,
            query=question,
            n_articles=8,
            return_type="both",
            strategy="news knowledge"
        ))

        # Wait for both tasks to complete
        hot_response, historical_response = await asyncio.gather(hot_task, historical_task)

    hot_articles = hot_response.as_dicts
    historical_articles = historical_response.as_dicts
    formatted_articles = "Here are the relevant news articles:\n\n"

    if hot_articles:
        hot_articles = [article.__dict__ for article in hot_articles]
        hot_articles = sorted(hot_articles, key=lambda x: x["pub_date"], reverse=True)

        for article in hot_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if historical_articles:
        historical_articles = [article.__dict__ for article in historical_articles]
        historical_articles = sorted(
            historical_articles, key=lambda x: x["pub_date"], reverse=True
        )

        for article in historical_articles:
            pub_date = article["pub_date"].strftime("%B %d, %Y %I:%M %p")
            formatted_articles += f"**{article['eng_title']}**\n{article['summary']}\nOriginal language: {article['language']}\nPublish date: {pub_date}\nSource:[{article['source_id']}]({article['article_url']})\n\n"

    if not hot_articles and not historical_articles:
        formatted_articles += "No articles were found.\n\n"
        return formatted_articles

    return formatted_articles

def call_perplexity(prompt: str) -> str:
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar-deep-research",
        "messages": [
            {
                "role": "system",
                "content": "Be thorough and detailed. Be objective in your analysis, proving documented facts only. Cite all sources with names and dates."
            },
            {
                "role": "user",
                "content": prompt + " Cite all sources with names and dates, compiling a list of sources at the end. Be objective in your analysis, providing documented facts only."
            }
        ]
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {PERPLEXITY_API_KEY}"
    }

    max_retries = 3
    backoff = 3  # seconds to wait between retries

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=200)
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                return content.strip()
            else:
                print(f"[Perplexity API] ❌ Error: HTTP {response.status_code}: {response.text}")
                return f"Error: {response.status_code}, {response.text}"

        except (ConnectionError, RemoteDisconnected, Timeout) as e:
            print(f"[Perplexity API] ⚠️ Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("[Perplexity API] ❌ Max retries reached. Giving up.")
                return f"Error: Connection aborted after {max_retries} retries ({str(e)})"
            else:
                wait_time = backoff * attempt
                print(f"[Perplexity API] 🔁 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)

    # Should never reach here
    return "Unexpected error in call_perplexity"



async def google_search(query, is_news=False, date_before=None):
    original_query = query
    query = query.replace('"', '').replace("'", '').strip()
    write(f"[google_search] Cleaned query: '{query}' (original: '{original_query}') | is_news={is_news}, date_before={date_before}")
    
    search_type = "news" if is_news else "search"
    url = f"https://google.serper.dev/{search_type}"
    headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "q": query,
        "num": 20
    })
    timeout = ClientTimeout(total=70)

    try:
        async with ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, data=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('news' if is_news else 'organic', [])
                    write(f"[google_search] Found {len(items)} raw results")

                    filtered_items = []
                    for item in items:
                        item_url = item.get('link')
                        item_date_str = item.get('date', '')
                        item_date = parse_date(item_date_str)
                        if date_before:
                            if item_date != "Unknown" and validate_time(date_before, item_date):
                                write(f"[google_search] ✅ Keeping: {item_url} (date: {item_date})")
                                filtered_items.append(item)
                            else:
                                write(f"[google_search] ❌ Dropped by date: {item_url} (date: {item_date})")
                        else:
                            write(f"[google_search] ✅ Keeping: {item_url}")
                            filtered_items.append(item)

                        if len(filtered_items) >=12:
                            break
                    
                    urls = [item['link'] for item in filtered_items]
                    write(f"[google_search] Returning {len(urls)} URLs: {urls}")
                    return urls
                else:
                    write(f"[google_search] Error in Serper API response: Status {response.status}")
                    response.raise_for_status()
    except Exception as e:
        write(f"[google_search] Exception: {str(e)}")
        raise


async def call_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model="o4-mini",
        input= prompt
    )
    return response.output_text


async def google_search_and_scrape(query, is_news, question_details, date_before=None):
    write(f"[google_search_and_scrape] Called with query='{query}', is_news={is_news}, date_before={date_before}")
    urls = await google_search(query, is_news, date_before)

    if not urls:
        write(f"[google_search_and_scrape] ❌ No URLs returned for query: '{query}'")
        return f"<Summary query=\"{query}\">No URLs returned from Google.</Summary>\n"

    async with FastContentExtractor() as extractor:
        write(f"[google_search_and_scrape] 🔍 Starting content extraction for {len(urls)} URLs")
        results = await extractor.extract_content(urls)
        write(f"[google_search_and_scrape] ✅ Finished content extraction")

    summarize_tasks = []
    no_results = random.choices([2, 3], weights=[0.44, 0.56], k=1)[0]
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
        else:
            write(f"[google_search_and_scrape] ⚠️ No content for {url}, skipping summarization.")

    if not summarize_tasks:
        write("[google_search_and_scrape] ⚠️ Warning: No content to summarize")
        return f"<Summary query=\"{query}\">No usable content extracted from any URL.</Summary>\n"

    summaries = await asyncio.gather(*summarize_tasks, return_exceptions=True)

    output = ""
    for url, summary in zip(results.keys(), summaries):
        if isinstance(summary, Exception):
            write(f"[google_search_and_scrape] ❌ Error summarizing {url}: {summary}")
            continue
        output += f"\n<Summary source=\"{url}\">\n{summary}\n</Summary>\n"

    return output

# question_details has the below structure:
# title = question_details["title"]
# resolution_criteria = question_details["resolution_criteria"]
# background = question_details["description"]
# fine_print = question_details["fine_print"]

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
                tasks.append(asyncio.to_thread(call_perplexity, query))

        if not tasks:
            write(f"Forecaster {forecaster_id}: No tasks generated")
            return ""

        # 5) Await all tasks in parallel
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=300)

        # 6) Format the outputs
        formatted_results = ""
        for match, result in zip(search_queries, results):
            # recover query and source for labeling
            if len(match) == 3:
                _, raw_query, source = match
            else:
                raw_query, source = match
            query = raw_query.strip().strip('"').strip("'")

            if isinstance(result, Exception):
                write(f"[process_search_queries] ❌ Forecaster {forecaster_id}: Error for '{query}' → {result}")
                continue
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
        write(f"Forecaster {forecaster_id}: Error processing search queries: {e}")
        return ""

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