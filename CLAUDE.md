# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Running the Bot
```bash
# Run the main forecasting bot on tournament questions
python Bot/main.py

# Run benchmark tests
python Bot/benchmark.py

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
Required environment variables in `.env`:
- `METACULUS_TOKEN` - Authentication for Metaculus API
- `OPENAI_API_KEY` - GPT models 
- `ANTHROPIC_API_KEY` - Claude models (via llm-proxy.metaculus.com)
- `SERPER_KEY` - Google search API
- `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` - News API
- `PERPLEXITY_API_KEY` - Research synthesis (optional)

### Key Configuration
Main configuration constants in `Bot/main.py`:
- `SUBMIT_PREDICTION = True` - Actually post to Metaculus
- `NUM_RUNS_PER_QUESTION = 5` - Ensemble size per question
- `TOURNAMENT_ID` - Target tournament (default: Q2_2025_AI_BENCHMARKING_ID)
- `USE_EXAMPLE_QUESTIONS = False` - Use test questions vs tournament

## Architecture Overview

### Multi-Agent Ensemble System
The bot implements a 5-agent forecasting committee:
- 2x Claude 3.7 Sonnet instances (robust reasoning)
- 2x GPT-o4-mini instances (complementary perspective) 
- 1x GPT-o3 instance (double-weighted for diversity)

Each agent approaches problems differently through prompt engineering directing some toward outside view (historical/reference class) reasoning and others toward inside view (mechanistic/causal) reasoning.

### Core Processing Flow
```
Question Analysis → Search Query Generation → Parallel Information Retrieval
    ↓
Content Extraction → Context Synthesis → Multi-Agent Forecast Generation
    ↓  
Ensemble Aggregation → Statistical Calibration → Final Prediction
```

### Key Components

**Search & Retrieval (`Bot/search.py`)**:
- `process_search_queries()` - Orchestrates multi-source information gathering
- Integrates Google Search, AskNews, Perplexity APIs
- `FastContentExtractor` for robust web content extraction

**Question-Type Handlers**:
- `Bot/binary.py` - Binary predictions with ensemble voting
- `Bot/numeric.py` - Numeric forecasts with 201-point CDFs  
- `Bot/multiple_choice.py` - Multi-option probability distributions

**LLM Interface (`Bot/llm_calls.py`)**:
- `call_claude()` - Claude via Metaculus proxy
- `call_gpt_o3()` / `call_gpt_o4_mini()` - OpenAI models
- Async processing with retry logic and backoff

### Forecast Generation Process
Binary questions follow this multi-stage process:
1. Generate historical vs current context searches
2. Parallel context retrieval and article summarization
3. 5-agent ensemble reasoning with weighted aggregation
4. Statistical ensemble combining forecasts (o3 gets 2x weight)

### Content Extraction
`FastContentExtractor.py` implements multi-strategy web scraping:
- BeautifulSoup with site-specific selectors
- Trafilatura structured extraction  
- Readability algorithm fallback
- Metadata and entity recognition

### Output Management
- Forecasts saved to `Q2_tournament_forecasts/` with detailed reasoning
- Each question gets comprehensive log file with multi-agent outputs
- Automatic filename sanitization for cross-platform compatibility

## Development Notes

### Async Architecture  
Heavy use of `asyncio` for concurrent API calls and processing. All major functions are async and use `asyncio.gather()` for parallel execution.

### Error Handling
Robust retry logic with exponential backoff for API calls. Graceful degradation when services unavailable.

### Benchmarking System
`benchmark.py` provides comprehensive evaluation with normalized peer scoring, Monte Carlo simulation, and performance visualization against community predictions.