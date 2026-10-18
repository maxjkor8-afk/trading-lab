# Multi-Agent Crypto Trading — Research Basis & Architecture

Compiled 2026-10-18. Sources are SSRN + arXiv, verified by direct fetch where the
publisher allowed it. SSRN itself is behind Cloudflare bot verification, so SSRN
entries below come from SSRN's own indexed search results, not full-text reads —
flagged where that matters.

---

## 1. The single most important finding

> "Strong model or forecasting capability does not reliably translate into trading
> performance under live market conditions and reliability controls."
> — *Agentic Quantitative Trading: A Survey of Workflows, Systems, and Evaluation*,
> Hua et al., arXiv:2608.31041, 31 Aug 2026

This is the most recent and broadest survey of exactly what we're building. Its
structural finding matters more than any individual result:

- Current agentic trading systems **concentrate on signal discovery**.
- Full integration with **portfolio construction, execution, and risk control is
  uncommon.**
- Multi-agent systems "rely heavily on aggregation" despite diverse structures.

Translation: the part that is fun to build (agents debating charts) is the part
everyone has already built. The part that determines whether you make or lose money
(execution, sizing, risk) is the part almost nobody integrates. Our edge, if we have
one, is not in the debate layer.

---

## 2. Papers worth implementing from

### 2a. The architecture is already published

**TradingAgents: Multi-Agents LLM Financial Trading Framework**
Xiao, Sun, Luo, Wang (UCLA / MIT / Tauric Research) — arXiv:2412.20138, MARW @ AAAI 2025

This is, almost exactly, the system described in the request. Roles:

| Layer | Agents |
|---|---|
| Analyst team | fundamentals, sentiment, news, technical |
| Researcher team | **bullish researcher vs bearish researcher** (debate, N rounds) |
| Trader | reads reports + debate outcome, decides, writes rationale |
| Risk team | risk-seeking / neutral / **conservative** (debate, N rounds) |
| Fund manager | final risk adjustment, updates trader decision |

Design details worth copying:

- Agents communicate via **structured reports, not free-form chat**. The authors did
  this specifically to avoid the "telephone effect" where detail is lost across long
  message histories. Natural language is used *only* inside debates, and debate
  outcomes are written back as structured entries.
- **Tiered models**: cheap/fast models (gpt-4o-mini class) for retrieval and
  summarization; deep-reasoning models for analysis and decisions. This is the
  difference between a $50/mo and a $2,000/mo run.
- ReAct prompting throughout.

**Read the reported results carefully — they are a lesson in what not to trust:**

| Asset | Cum. Return | Sharpe | Max Drawdown |
|---|---|---|---|
| AAPL | 26.62% | **8.21** | 0.91% |
| GOOGL | 24.36% | 6.39 | 1.69% |
| AMZN | 23.21% | 5.60 | 2.11% |

A Sharpe ratio of 8.21 means almost never having a losing period. Renaissance
Medallion ran roughly 2.5 net over decades and is the best track record in the history
of finance. **A Sharpe of 8 on a 3-month backtest is not a result, it is a
diagnostic** — it tells you the test window was too short and too favourable.

To their credit the authors say as much: they flag that the Sharpe "exceeds their
expected range," attribute it to few pullbacks in the window, and state the backtest
was only **Jan 1 – Mar 29 2024** (a strong uptrend) because of LLM cost. The paper
does not discuss transaction costs. Equities, not crypto.

Use the architecture. Do not use the numbers as evidence of anything.

### 2b. Self-improvement, done the right way

**An Adaptive Multi Agent Bitcoin Trading System**
Aadi Singhi — arXiv:2510.08068, Oct 2025 (rev. Nov 2025), CC BY 4.0

Crypto-native, and it answers the self-improvement question directly. Agents:
technical analysis, sentiment evaluation, decision-making, and a **Reflect agent**
that issues daily and weekly natural-language critiques fed back into the other
agents' prompts.

The key mechanism: **improvement happens at the prompt level, not the weight level.**
No fine-tuning, no gradient updates. The authors frame verbal feedback as a scalable,
low-cost tuning method. Lineage is Reflexion / Self-Refine / Constitutional AI.

Reported (backtest, Jul 2024 – Apr 2025, single author, no live results):

- Quantitative agent: ~15% over buy-and-hold overall, +30% in bullish phases
- Sentiment agent: turned a small sideways-market loss into >100% gain
- **Adding weekly feedback improved total performance 31% and cut bearish-phase
  losses 10%**

That last line is the one to build on. Daily reflection underperformed weekly —
consistent with daily critique overfitting to noise.

### 2c. Quantitative + qualitative fusion (the explicit ask)

**Optimizing Cryptocurrency Trading with Machine Learning: Predictive Analytics with
Limit Order Book and Sentiment Data** — Diana Gorsh, SSRN 4867340

Combines order book depth + **order flow imbalance** + sentiment via LSTM and VAE
ensembles. This is the quant/qual combination, already specified for crypto. Note it
is capstone-level work — treat the feature set as the contribution, not the
performance claims.

**Order flow imbalance is the feature to steal.** It is the best-documented
short-horizon predictor in microstructure and it is computable from free exchange
websocket data.

### 2d. Why the chart is moving — and why the book is lying

**Learning the Spoofability of Limit Order Books With Interpretable Probabilistic
Neural Networks** — Fabre & Challet, SSRN 5230047

Real-time-capable detection of spoofing in crypto order books. Directly relevant to
the "why is it moving" agent: **a large share of visible crypto book depth is not
real.** An agent that reads depth naively is reading a fiction written to be read.

### 2e. More agents is not better

**Multi-Agent Large Language Models with Reinforcement Learning for Quant Trading**
Chen & Kawashima, SSRN 6682038 *(from SSRN index summary — full text not read)*

Finding: the multi-agent LLM approach beat equal-weighted benchmarks on cumulative
return **but carried more short-term risk**. Single-agent PPO delivered more stable,
risk-controlled performance with **smaller drawdowns**.

So the multi-agent design buys headline return and pays for it in risk. If the
objective is risk-adjusted return, this paper is evidence *against* the many-agent
architecture. Worth knowing before building five agents.

### 2f. How we will fool ourselves

**OpenPM: Auditable Point-in-Time Evaluation for LLM Portfolio-Management Agents**
Cai, Guo, Liu, Han, Guo, Long, Chen, Wu, Metaxas, Li — SSRN 7262158
*(from SSRN index summary — full text not read)*

Argues reported results for LLM trading agents are inflated by three specific things:

1. **Look-ahead leakage** — the LLM knows what happened. Near-unavoidable with a
   pretrained model backtested on dates inside its training data. Backtest BTC over
   2024 with a model trained through 2026 and the model has *read the news*. This one
   issue invalidates most LLM trading backtests, arguably including both papers above.
2. **Optimistic execution** — fills assumed at prices you would not get.
3. **Risk mandates described but not enforced** — the prompt says "be conservative";
   nothing in the code makes it so.

Proposes point-in-time evaluation. **Adopt this as the test harness before building
the agents.** It is the difference between a system and a story.

---

## 3. The build stack (all verified live, 2026-10-18)

| Repo | Stars | License | Role |
|---|---|---|---|
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | 111.8k | Apache-2.0 | The agent layer. v0.7.0, Oct 2026. Python 3.11+. Handles `BTC-USD`/`ETH-USD` via Yahoo. |
| [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader) | 30.3k | LGPL-3.0 | Execution + backtest engine. Rust core, Python control plane. |
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 55.5k | open source | Fastest path to something running. Dry-run, hyperopt, FreqAI. |
| ccxt/ccxt | — | MIT | Unified exchange API. The boring dependency everything rests on. |
| AI4Finance-Foundation/FinRL, FinGPT, FinRobot | ~7k each | — | RL + LLM finance reference implementations. Authors state: academic use, not for live trading. |

**Critical fact about TradingAgents:** approved orders go to a **simulated exchange**.
It does not ship live brokerage execution. The repo explicitly says it is "designed
for research purposes" and "not intended as financial, investment, or trading advice."
Live execution is the piece you would have to write — which is, per the Hua survey,
exactly the piece the field under-builds.

**NautilusTrader is the right execution layer** because strategy and execution code
run on the *same code path* in backtest and live. That property kills an entire class
of bug where backtest and live bot quietly differ. Crypto venues marked stable:
Binance, Bybit, Coinbase, Deribit, Kraken, OKX, plus Hyperliquid, dYdX, Derive and
Lighter on the DEX side. Maintainer caveat: the v2 release candidates should **not**
be used for live trading with real capital, and breaking changes are expected.

Recommended split: **TradingAgents for reasoning → NautilusTrader for execution.**
Freqtrade if the goal is something running this week.

---

## 4. Architecture

The requested four agents, mapped onto what the literature supports — plus the layer
missing from the request.

```
                        ┌─────────────────────────────┐
   market data ───────► │  INGEST (no LLM)            │
   (ccxt / websockets)  │  OHLCV, order book, OFI,    │
                        │  funding rate, open interest│
                        └──────────────┬──────────────┘
                                       │ structured features
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
        ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
        │ CHART AGENT   │      │ CAUSE AGENT   │      │ FLOW AGENT    │
        │ technical     │      │ "why moving"  │      │ OFI, spoof    │
        │ structure     │      │ news/sentiment│      │ detection     │
        └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
                └──────────────────────┼──────────────────────┘
                                       ▼
                        ┌─────────────────────────────┐
                        │  BULL  ⟷  BEAR  (N rounds)  │ ◄── the skeptic
                        │  facilitator picks outcome   │
                        └──────────────┬──────────────┘
                                       ▼
                        ┌─────────────────────────────┐
                        │  TRADER: proposes an order   │
                        └──────────────┬──────────────┘
                                       ▼
        ╔═══════════════════════════════════════════════════════════╗
        ║  RISK GATE — PLAIN CODE, NO LLM, NOT PROMPT-EDITABLE      ║
        ║  • max position size        • max leverage (hard cap)     ║
        ║  • max daily loss -> halt   • max orders/hour             ║
        ║  • no averaging down        • kill switch                 ║
        ║  REJECTS the order outright. Cannot be argued with.       ║
        ╚═══════════════════════════┬═══════════════════════════════╝
                                    ▼
                        ┌─────────────────────────────┐
                        │  EXECUTION (Nautilus)        │
                        │  paper -> testnet -> live    │
                        └──────────────┬──────────────┘
                                       ▼
                        ┌─────────────────────────────┐
                        │  REFLECT AGENT (weekly)      │
                        │  critiques -> prompt deltas  │
                        │  WRITE ACCESS: prompts only  │
                        └─────────────────────────────┘
```

### The two things this adds to the request

**1. The risk gate is not an agent.** In TradingAgents the risk team is itself a set
of LLM agents that debate, and the fund manager "makes final risk adjustments." That
is precisely OpenPM failure mode #3: a risk mandate *described* rather than *enforced*.
An LLM that can be persuaded is not a risk control. The hard limits must be ordinary
code that no agent can reach, reason with, or rewrite.

**2. The skeptic needs to be genuinely independent.** A bear agent that is the same
base model with a different system prompt is not an independent check — it shares the
same training data, the same blind spots and the same hallucination modes. Its errors
are correlated with the bull's. Two options that actually buy independence:

- a **different model family** for the skeptic, so errors decorrelate, and/or
- a **non-LLM statistical gate** — the trade is rejected unless the setup clears a
  pre-registered quantitative filter. A linear model that says no is worth more than
  an eloquent LLM that says no.

---

## 5. Self-improvement: what is safe and what is not

The "self-improving" instinct is correct, and 2510.08068 shows the safe version. The
distinction that matters:

| Safe | Not safe |
|---|---|
| Reflect agent edits a **bounded region of a prompt** | Agent edits its own risk limits |
| **Weekly** critique cadence (daily overfit noise in the paper) | Continuous online adaptation |
| Proposes strategy changes -> **you approve** -> applied | Auto-applies to the live book |
| Reward = risk-adjusted return, out-of-sample | Reward = PnL, measured live |

**The failure mode to design against, concretely:** any optimizer rewarded on
short-horizon PnL with write access to its own position sizing will discover that
leverage and averaging-down raise the reward for a while. Martingale has a beautiful
equity curve right up until it goes to zero. This is not a hypothetical about rogue
AI — it is the ordinary consequence of optimising a metric that is maximised by hidden
tail risk. The mitigation is structural, not a prompt: **sizing and leverage live
behind the risk gate, where nothing in the agent loop can reach them.**

---

## 6. On "trading the FOMO" — what the research actually says

Asked as a strategy question; answering as a *signal* question, because that is what
the literature addresses.

**The dominant finding is reverse causality.** Across the attention studies, the
robust, repeatedly-significant relationship runs **returns → attention**, not
attention → returns. Studies using Twitter, Reddit and Google Trends find the
attention index does not explain crypto returns, "however, the inverse relationship is
found to be highly significant." Multiple theses find past returns drive the following
week's attention for BTC, ETH and XRP.

Results in the other direction are mixed and regime-dependent. A 2019–2022 study of
154 coins found attention positively affects returns, volatility and volume, robust to
controls; an earlier top-20 study found bi-directional Granger causality. But another
found the attention–return link **disappears post-bubble** for BTC and ETH.

So retail-visible FOMO metrics are substantially a **lagging** indicator. Building
entries on them means trading a signal that fires after the move. If attention has
tradeable content, the literature points to it being in **volatility** — where the
finding is more consistent — rather than direction.

**The cleanest real-time crowding proxy is the perpetual funding rate.** It is
literally the price leveraged longs are paying to stay long, published by the venue,
no NLP required. The qualitative industry view is that extended elevated funding
"often precedes sharp liquidation-driven reversals," and that liquidation cascades
reset or invert funding. Two honest caveats: that reversal claim is qualitative, not
an empirical test I could verify, and a BitMEX study cited secondhand says extreme
funding events have fallen ~90% since 2016 — whatever edge existed has decayed as the
market matured.

Note the implication, since it inverts the premise: the researched way to engage a
FOMO spike is to be the one **providing liquidity to it**, not joining it. That is a
mean-reversion / market-making posture, not a momentum entry.

**This is signal research, not a recommendation.** What to trade and whether to trade
is your call — I'm not a licensed advisor.

---

## 7. Staged build, sized for ~2 hrs/day

Each stage has a gate. The point of the gates is that most systems die at one of them,
and dying at stage 2 costs a weekend instead of a bankroll.

**Stage 0 — harness first (1 week).** Before any agent. Build the OpenPM-style
point-in-time evaluator: fee model, slippage model, no look-ahead. Run buy-and-hold
through it. *Gate: the harness reproduces a known buy-and-hold return correctly.*

**Stage 1 — ingest (1 week).** ccxt + websockets. OHLCV, order book, OFI, funding
rate, open interest. Persist it. *Gate: 7 unbroken days of stored tick data.*

**Stage 2 — one dumb baseline (1 week).** No LLM. One rule, e.g. a funding-rate
threshold. Run it through the stage-0 harness with real fees. *Gate: does it beat
buy-and-hold after costs? Usually no.* **This is where most ideas correctly die**, and
finding that out before building five agents is the entire value of the stage.

**Stage 3 — agent layer (2–3 weeks).** Clone TradingAgents, point it at BTC-USD.
Structured reports between agents, tiered models. Same harness. *Gate: does the agent
layer beat the stage-2 dumb baseline after costs?* If not, the agents are decoration —
a real and common outcome, not a failure of the build.

**Stage 4 — risk gate (1 week).** Plain code, outside the agent loop. Write the kill
switch before the live keys exist. *Gate: unit tests prove the gate rejects oversized,
over-leveraged and post-daily-loss orders.*

**Stage 5 — paper, then testnet (6–8 weeks minimum).** Binance/Bybit testnet via
Nautilus. *Gate: paper results within a sane distance of backtest. If paper diverges
hard from backtest, the backtest was leaking — go back to stage 0.*

**Stage 6 — Reflect agent (ongoing).** Weekly cadence, prompt-only write access, your
approval before changes apply.

**Stage 7 — live, trivially small.** An amount whose total loss changes nothing. Scale
only on out-of-sample evidence, never on a good week.

Realistically stages 0–5 are a couple of months of evenings. Anyone promising faster is
skipping the harness, which is the only part that tells you the truth.

---

## 8. Sources

- Hua et al., *Agentic Quantitative Trading: A Survey* — arXiv:2608.31041 (Aug 2026)
- Xiao, Sun, Luo, Wang, *TradingAgents* — arXiv:2412.20138 (AAAI 2025 MARW)
- Singhi, *An Adaptive Multi Agent Bitcoin Trading System* — arXiv:2510.08068
- Gorsh, *Optimizing Cryptocurrency Trading with ML: LOB and Sentiment* — SSRN 4867340
- Fabre & Challet, *Learning the Spoofability of Limit Order Books* — SSRN 5230047
- Chen & Kawashima, *Multi-Agent LLMs with RL for Quant Trading* — SSRN 6682038
- Cai et al., *OpenPM: Auditable Point-in-Time Evaluation* — SSRN 7262158
- Saha et al., *LLM Agents for Investment Management* — SSRN 5447274 (reading list)
- Nagy, Calliess, Zohren, deep Q-learning for LOB execution — PMC10561243 (equities)
- *Deep RL in Cryptocurrency Market Making* — arXiv:1911.08647, arXiv:2004.06985
- Quantopian 888-strategy study (via QuantInsti): backtest Sharpe explained <2.5% of
  out-of-sample Sharpe variance

SSRN entries marked above were read from SSRN's indexed search results; SSRN blocks
automated full-text access, so those abstracts are unverified against the PDFs. Worth
a manual read before implementing from them — particularly OpenPM 7262158, which is
the most load-bearing of the set.
