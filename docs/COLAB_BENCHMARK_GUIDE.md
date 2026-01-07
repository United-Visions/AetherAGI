# 🧠 AetherMind Google Colab Benchmark Runner

Run AetherMind benchmarks concurrently on Google Colab with GPU acceleration.

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Recommended)

1. **Upload to Colab**: Open [Google Colab](https://colab.research.google.com) and upload:
   ```
   notebooks/AetherMind_Benchmark_Colab.ipynb
   ```

2. **Set your API key** (optional but recommended):
   - Click the 🔑 **Secrets** icon in the left sidebar
   - Add a secret named `AETHER_API_KEY` with your key

3. **Run all cells** - benchmarks will execute concurrently!

### Option 2: Python Script

```bash
# In Colab, run:
!pip install httpx aiohttp datasets tqdm nest_asyncio

# Upload colab_runner.py then:
!python benchmarks/colab_runner.py \
    --api-base https://aetheragi.onrender.com \
    --questions 20 \
    --key YOUR_API_KEY
```

---

## 📊 Available Benchmark Families

| Family | Type | Description |
|--------|------|-------------|
| `gsm8k` | Math Reasoning | Grade school math word problems |
| `mmlu` | Knowledge | 57 academic subjects |
| `arc_challenge` | Logical Reasoning | AI2 science questions |
| `hellaswag` | Logical Reasoning | Commonsense completion |
| `winogrande` | Language | Pronoun resolution |
| `truthfulqa` | Knowledge | Truthfulness testing |

---

## ⚙️ Configuration

### API Endpoints

| Environment | Endpoint |
|-------------|----------|
| **Production** | `https://aetheragi.onrender.com` |
| **Local** | `http://localhost:8000` |

### Command Line Options

```bash
python colab_runner.py \
    --api-base URL           # API endpoint (default: production)
    --key API_KEY            # Your AetherMind API key
    --questions 0            # 0 = ALL questions (default), or set a number for quick tests
    --concurrency 4          # Max parallel API calls (default: 4)
    --timeout 120            # Per-question timeout (default: 120s)
    --families gsm8k mmlu    # Specific families to run
    --output results.json    # Output file path
```

### Full Dataset Sizes

| Family | Questions | Split |
|--------|-----------|-------|
| GSM8K | 1,319 | test |
| MMLU | 14,042 | test |
| ARC-Challenge | 1,172 | test |
| HellaSwag | 10,042 | validation |
| WinoGrande | 1,267 | validation |
| TruthfulQA | 817 | validation |
| **TOTAL** | **~28,659** | |

---

## 🔄 Concurrent Execution

The runner executes **all benchmark families in parallel**:

```
Timeline:
  t=0   ├── GSM8K starts ──────────────────────────┤
        ├── MMLU starts ───────────────────────────┤
        ├── ARC starts ────────────────────────────┤
        ├── HellaSwag starts ──────────────────────┤
  t=N   └── All complete, results aggregated ──────┘
```

Rate limiting via semaphore prevents API overload while maximizing throughput.

---

## 📈 Sample Output

```
============================================================
🧠 AetherMind Concurrent Benchmark Runner
🌐 API: https://aetheragi.onrender.com
📊 Families: 6
❓ Questions per family: 20
🔄 Max concurrent calls: 4
============================================================

📥 Loading datasets...
   ✅ GSM-8K: 20 questions loaded
   ✅ MMLU: 20 questions loaded
   ✅ ARC-Challenge: 20 questions loaded
   ...

🏃 Running benchmarks concurrently...

============================================================
📊 BENCHMARK RESULTS SUMMARY
============================================================

Family               Score        Correct      Latency
--------------------------------------------------------
GSM-8K                85.0%       17/20          2100ms
MMLU                  75.0%       15/20          1800ms
ARC-Challenge         80.0%       16/20          1950ms
HellaSwag             70.0%       14/20          1700ms
WinoGrande            65.0%       13/20          1600ms
TruthfulQA            60.0%       12/20          1850ms
--------------------------------------------------------
OVERALL               72.5%       87/120

⏱️ Total time: 45.3s
📅 Timestamp: 2026-01-07T12:34:56Z
```

---

## ️ Troubleshooting

### "API connection failed"
- Render free tier sleeps after 15 mins of inactivity
- Visit https://aetheragi.onrender.com/health to wake it up
- Wait 30s and retry

### "Rate limited"
- Reduce `--concurrency` to 2
- Increase `--timeout` to 180

### "Dataset loading slow"
- First run downloads from HuggingFace (cached after)
- Colab's network is fast, but initial download takes ~1-2 min

---

## 📁 Files

```
notebooks/
  └── AetherMind_Benchmark_Colab.ipynb  # Full Colab notebook

benchmarks/
  └── colab_runner.py                    # Standalone Python script
```

---

## 🔐 API Key

Get your API key:
1. Visit https://aetheragi.onrender.com
2. Login with GitHub
3. Navigate to Settings → API Keys
4. Generate a new key (starts with `am_live_`)

Store securely:
- **Colab**: Use Secrets (🔑 sidebar)
- **Local**: Set `AETHER_API_KEY` env var
- **Script**: Pass via `--key` flag
