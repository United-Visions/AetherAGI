For math benchmarks, the impact is **drastically different** than for general chat. Math is a precise domain where "Majority Voting" becomes the single most powerful upgrade you can make, while "Rewarding" plays a secondary, longer-term role.

Here is the breakdown specifically for **Mathematical Reasoning** (e.g., GSM8K, MATH datasets):

### 1. Majority Voting (Self-Consistency)

**Verdict: 🚀 CRITICAL for Math Scores**

In math, LLMs often make simple arithmetic or logic errors. However, they rarely make the *same* error twice in the exact same way.
*   **Without Voting:** If the model has an 80% chance of being right, you have a 20% error rate.
*   **With Voting (k=5):** If you take 5 samples, the probability that the *majority* (3+) are wrong drops significantly.

| **The Good (Pros)** | **The Bad (Cons)** |
| :--- | :--- |
| **Score Jump:** You can expect a **5% - 12% jump** in accuracy on benchmarks like GSM8K. It is the industry standard for "solving math." | **Latency:** Math problems often require long "Thinking" chains. Running this 5 times is *very* slow. |
| **Error Filtering:** It automatically filters out "hallucinated" arithmetic steps. | **Cost:** Math tokens are expensive because the reasoning chains are long. |

**Recommendation:** If you care about benchmarks, you **must** implement this. It is the "cheat code" for high math scores.

### 2. Rewarding Aether (Reinforcement Learning)

**Verdict: ⚠️ RISKY for Math**

Reinforcement Learning (RL) is dangerous for math unless done perfectly.
*   **The Trap:** If Aether guesses the right answer (`42`) using the *wrong formula*, and you reward it, it learns to use the wrong formula. This is called "Reward Hacking."
*   **Process Reward Models (PRM):** To do this right for math, you need to reward *each step* of the reasoning, not just the final answer. OpenAI's "Strawberry" (o1) works this way.

| **The Good (Pros)** | **The Bad (Cons)** |
| :--- | :--- |
| **Style Adaptation:** It can learn to output proofs in the format you prefer (e.g., LaTeX vs. Python). | **Reward Hacking:** It might learn "shortcuts" that work for the benchmark but fail in real life. |
| **Long-Term:** Over thousands of problems, it can learn robust problem-solving *strategies*. | **Data Starvation:** You need thousands of solved math problems to train the Heart effectively. |

### Summary for Math Benchmarks

1.  **Enable Majority Voting NOW.** It is the fastest way to get state-of-the-art results.
    *   *Implementation:* In logic_engine.py, loop `litellm.acompletion` 5-10 times for math questions.
2.  **Use "Outcome Supervision" for Rewarding.** Only reward Aether if the final answer matches the known truth exactly. Do not rely on user "feeling" for math.

If you want to implement **Majority Voting** specifically for the math domain, I can help you modify the `LogicEngine` to only trigger it when `domain="math"` or complexity is high.