import random
from search import call_perplexity
# Assuming call_perplexity is defined in the same file, or imported

# === TEST SCRIPT ===

if __name__ == "__main__":
    sample_queries = [
        "US unemployment rate April 2025",
        "Latest AI advancements in 2025",
        "Impact of climate change on agriculture",
        "SpaceX Mars mission updates",
        "Global inflation forecast 2025",
        "Deep learning breakthroughs 2025",
        "International tourism trends April 2025",
        "Electric vehicle adoption rates in Asia",
        "GDP growth rate predictions 2025",
        "Water scarcity challenges by 2030"
    ]

    random_query = random.choice(sample_queries)
    print(f"\n🎯 Selected random query: {random_query}\n")

    result = call_perplexity(random_query)

    print("\n=== RESULT ===\n")
    print(result)
    print("\n=== END ===\n")