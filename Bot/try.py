import numpy as np

from asknews_sdk import AsyncAskNewsSDK
import asyncio
import dotenv
import os

dotenv.load_dotenv()

ASKNEWS_CLIENT_ID = os.getenv("ASKNEWS_CLIENT_ID")
ASKNEWS_SECRET = os.getenv("ASKNEWS_SECRET")


"""
Example usage of the DeepNews endpoint for Metaculus Participants.

More information available here: https://docs.asknews.app/en/deepnews

pip install asknews

# Activating your account
Please reach out to @drafty on Discord to request access,
please provide the email address and bot name associated with your AskNews account.

# Limitations for Metaculus bot accounts
1. Only "asknews" source available ("google" reserved for Spelunker tier and up)
2. `fast` models only, `deepseek-basic` and `llama` variants. Rich models like claude-3.7 reserved for Spelunker tier and up.
3. 5M token limit per month for Metaculus bot accounts.
4. Search depth and max depth limited to 2. Higher values reserved for Spelunker tier and up.
"""

client_id = ASKNEWS_CLIENT_ID
client_secret = ASKNEWS_SECRET

ask = AsyncAskNewsSDK(
    client_id=client_id,
    client_secret=client_secret,
    scopes=["chat", "news", "stories", "analytics"],
)


async def deep_research(
    query, sources, model, search_depth=2, max_depth=2
):

    response = await ask.chat.get_deep_news(
        messages=[{"role": "user", "content": query}],
        search_depth=search_depth,
        max_depth=max_depth,
        sources=sources,
        stream=False,
        return_sources=False,
        model=model,
        inline_citations="numbered"
    )

    print(response)

if __name__ == "__main__":
    query = "What is the TAM of the global market for electric vehicles in 2025? With your final report, please report the TAM in USD using the tags <TAM> ... </TAM>"

    sources = ["asknews"]
    model = "deepseek-basic"
    search_depth = 2
    max_depth = 2 
    asyncio.run(
        deep_research(
            query, sources, model, search_depth, max_depth
        )
    )


exit()

def normalized_eval(
    bot, community, q_type, *,
    # ── peer‑population parameters ──────────────────────────────
    n_peers = 100,
    tau_good = 80,
    tau_bad  = 20,
    frac_bad = 0.20,
    shift_range_binary = 0.15,
    shift_range_mc     = 0.25,   # base cap; auto‑scaled by k
    # ── Monte‑Carlo control ────────────────────────────────────
    n_runs = 1000,                 # << NEW: how many peer-pop draws to average
    eps = 1e-4,
    seed = None,
):
    """
    Expected Metaculus PEER score, estimated by averaging `n_runs`
    independent simulations of a mixture (good+bad) peer population.
    """

    rng = np.random.default_rng(seed)

    # ---------- helpers to draw peers --------------------------------------
    def _draw_peers_binary(comm_p, size):
        shift = rng.uniform(-shift_range_binary, shift_range_binary, size=size)
        mu = np.clip(comm_p + shift, eps, 1 - eps)              # shape (size,)

        n_bad  = int(round(n_peers * frac_bad))
        n_good = n_peers - n_bad

        peers_good = rng.beta(mu[:,None] * tau_good,
                              (1-mu)[:,None] * tau_good,
                              size=(size, n_good))
        peers_bad  = rng.beta(mu[:,None] * tau_bad,
                              (1-mu)[:,None] * tau_bad,
                              size=(size, n_bad))

        peers = np.concatenate([peers_good, peers_bad], axis=1) # (size, n_peers)
        return np.clip(peers, eps, 1 - eps)

    def _draw_peers_mc(comm_arr, size):            # size == n_runs
        k = len(comm_arr)
        shift_cap = shift_range_mc / max(1, k / 2)

        # ----- 1. draw shifted means for every run -----------------------------
        shift = rng.uniform(-shift_cap, shift_cap, size=size)      # (size,)
        mu_vec = np.clip(comm_arr + shift[:, None], eps, 1 - eps)  # (size, k)
        mu_vec /= mu_vec.sum(axis=1, keepdims=True)

        # ----- 2. allocate arrays ----------------------------------------------
        peers = np.empty((size, n_peers, k), dtype=float)
        n_bad  = int(round(n_peers * frac_bad))
        n_good = n_peers - n_bad

        # ----- 3. draw good & bad peers run‑by‑run ------------------------------
        for i in range(size):
            good = rng.dirichlet(mu_vec[i] * tau_good, size=n_good)
            bad  = rng.dirichlet(mu_vec[i] * tau_bad,  size=n_bad)
            peers[i, :n_good]     = good
            peers[i, n_good:]     = bad

        return np.clip(peers, eps, 1 - eps)         # shape (n_runs, n_peers, k)

    # ---------- expected score, averaged over n_runs -----------------------
    if q_type == "binary":
        bot_p  = np.clip(float(bot), eps, 1 - eps)
        comm_p = np.clip(float(community), eps, 1 - eps)

        peer_p = _draw_peers_binary(comm_p, n_runs)             # (n_runs, n_peers)

        ln_gm_yes = np.log(peer_p).mean(axis=1)                 # (n_runs,)
        ln_gm_no  = np.log(1 - peer_p).mean(axis=1)

        ps_yes = 100 * (np.log(bot_p)     - ln_gm_yes)
        ps_no  = 100 * (np.log(1 - bot_p) - ln_gm_no)
        scores = comm_p * ps_yes + (1 - comm_p) * ps_no         # (n_runs,)

        return float(scores.mean())

    elif q_type == "multiple_choice":
        bot_arr  = np.asarray(bot,       dtype=float)
        comm_arr = np.asarray(community, dtype=float)

        if bot_arr.shape != comm_arr.shape:
            raise ValueError("bot and community distributions must match length")

        bot_arr  = np.clip(bot_arr,  eps, 1 - eps)
        comm_arr = np.clip(comm_arr, eps, 1 - eps)
        comm_arr /= comm_arr.sum()

        peer_mat = _draw_peers_mc(comm_arr, n_runs)             # (n_runs, n_peers, k)
        ln_gm    = np.log(peer_mat).mean(axis=1)                # (n_runs, k)

        peer_scores_each = 100 * (np.log(bot_arr) - ln_gm)      # (n_runs, k)
        scores = peer_scores_each @ comm_arr                    # (n_runs,)

        return float(scores.mean())

    else:
        raise ValueError("q_type must be 'binary' or 'multiple_choice'")
    

print(normalized_eval([0.40, 0.28, 0.15, 0.02, 0.03, 0.12], [0.34, 0.23, 0.08, 0.03, 0.06, 0.21], "multiple_choice"))