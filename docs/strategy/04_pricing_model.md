# Pricing Model: Maximizing Value Capture

**Date:** January 4, 2026  
**Philosophy:** Value-based pricing (not cost-plus)  
**Structure:** Good-Better-Best with usage-based components  
**Goal:** $300 LTV/CAC ratio by Year 3

---

## 🎯 Executive Summary

**Pricing Strategy:** Tier-based (3 tiers) + Usage-based (optional add-ons)

**Three Tiers:**
1. **Free (Open Core):** Self-hosted, 70% of features, unlimited usage
2. **Pro ($1,200-5,000/year):** Managed cloud, 90% of features, team collaboration
3. **Enterprise ($50,000-500,000/year):** Everything + dedicated support + SLAs + custom development

**Value-Based Anchor:** Customer saves $10K-1M/year, we charge 10-30% of savings

**Usage Components:**
- Compute (inference calls) - Optional for Pro, included in Enterprise
- Storage (memory/vectors) - Unlimited in all paid tiers
- Seats (users) - Pro starts at 1, Enterprise starts at 10

**Pricing Principles:**
1. **Transparent:** No hidden fees, public pricing
2. **Fair:** Pay for value received, not arbitrary limits
3. **Simple:** Easy to understand, predict costs
4. **Scalable:** Grows with customer usage

---

## 💰 Tier Breakdown

### Tier 1: Free (Open Core)

**Price:** $0/month

**What's Included:**
- Self-hosted deployment (your infra)
- 70% of codebase (Brain, Mind basics, Body basics, Orchestrator core)
- Unlimited users
- Unlimited messages
- Unlimited storage (you pay your Pinecone bill)
- Community support (Discord, GitHub)

**What's NOT Included:**
- Meta-controller (proactive intelligence)
- Self-modification (continuous improvement)
- ToolForge (autonomous tool generation)
- Heart (moral reasoning)
- Differentiable memory (learnable retrieval)
- Imagination engine (planning)
- Enterprise features (SSO, RBAC, audit logs)
- Managed hosting
- Support SLA

**Target Users:**
- Individual developers
- Researchers
- Students
- Startups (< 10 people)
- Non-commercial projects

**Economics:**
```
Cost to us: $0 (no hosting, no support)
Revenue: $0 direct

Indirect value:
- Brand awareness (GitHub stars, blog posts)
- User feedback (bug reports, feature requests)
- Hiring pipeline (contribute → hire)
- Lead generation (upgrade to Pro or bring to Enterprise)

Estimated users: 100K by Year 3
Conversion rate: 1% → 1,000 Pro customers
Value: $2.5M ARR from conversions
```

**Upgrade Trigger:**
```
User: "Can you proactively research new papers for me?"
Agent: "I can't do that automatically (need meta-controller). 
        Upgrade to Pro for autonomous research?"
[CTA: Upgrade now]
```

---

### Tier 2: Pro ($1,200-5,000/year)

**Pricing:** 
- **Starter:** $1,200/year (up to 1,000 calls/day)
- **Growth:** $2,400/year (up to 5,000 calls/day)
- **Scale:** $5,000/year (up to 20,000 calls/day)

**What's Included:**
- Everything in Free
- **Plus:**
  - Managed cloud hosting (we handle infra)
  - Meta-controller (proactive intelligence)
  - Self-modification (continuous improvement)
  - ToolForge (autonomous tool generation)
  - Team collaboration (up to 10 users)
  - Email support (48-hour response time)
  - Dashboard analytics (usage, performance)

**What's NOT Included:**
- Heart (moral reasoning) - Enterprise only
- Differentiable memory (learnable retrieval) - Enterprise only
- Imagination engine (planning) - Enterprise only
- SSO (Okta, Azure AD) - Enterprise only
- RBAC (role-based access) - Enterprise only
- Audit logs - Enterprise only
- On-premise deployment - Enterprise only
- Dedicated support - Enterprise only
- SLA (uptime guarantee) - Enterprise only

**Target Users:**
- Small teams (5-50 people)
- Freelancers/consultants
- Agencies
- Early-stage startups ($1M-10M revenue)

**Value Proposition:**
```
Without AetherMind:
- Research tools: $500/month (Elicit, Consensus, etc.)
- Productivity tools: $500/month (Notion, Slack, etc.)
- Development tools: $100/month (GitHub Copilot, etc.)
- Time saved: 20 hours/month @ $100/hour = $2,000/month
Total: $3,100/month = $37,200/year

With AetherMind Pro: $2,400/year

Savings: $34,800/year (93% cost reduction)
ROI: 14.5× return on investment
```

**Usage Limits:**
```
Starter ($1,200/year):
- 1,000 calls/day = 30K/month = 365K/year
- Cost per call: $0.0033
- Effective inference price: $0.50 per 1M tokens (60× cheaper than GPT-4)

Growth ($2,400/year):
- 5,000 calls/day = 150K/month = 1.8M/year
- Cost per call: $0.0013
- Effective: $0.40 per 1M tokens

Scale ($5,000/year):
- 20,000 calls/day = 600K/month = 7.3M/year
- Cost per call: $0.00068
- Effective: $0.35 per 1M tokens
```

**Overage Pricing:**
```
If user exceeds daily limit:
- Option 1: Wait until next day (rate limited)
- Option 2: Upgrade to next tier (in-app prompt)
- Option 3: Pay overage ($0.01 per call above limit)

Example:
User on Starter plan (1,000 calls/day)
Uses 1,200 calls today
Overage: 200 calls × $0.01 = $2

If this happens frequently → Prompt to upgrade to Growth
```

**Upgrade Trigger (Pro → Enterprise):**
```
User: "Can we add SSO for our team?"
Agent: "SSO is an Enterprise feature. You'll also get:
        - RBAC (role permissions)
        - Audit logs (compliance)
        - Dedicated support (CSM assigned)
        - 99.9% uptime SLA
        
        Would you like to talk to our sales team?"
[CTA: Request demo]
```

---

### Tier 3: Enterprise ($50,000-500,000/year)

**Pricing:** Custom (starts at $50K/year)

**Pricing Factors:**
1. **Seats:** Number of users
2. **Usage:** Inference volume (calls/month)
3. **Storage:** Vector storage (GB)
4. **Features:** On-prem, custom integrations, white-label
5. **Support:** Dedicated CSM, custom SLA

**Typical Pricing Examples:**

**Small Enterprise (100 users):**
```
Base: $50K/year (10 users included)
Additional users: 90 × $500/year = $45K
Usage: 1M calls/month included
Support: Standard CSM
Total: $95K/year
```

**Mid Enterprise (500 users):**
```
Base: $100K/year (50 users included)
Additional users: 450 × $300/year = $135K
Usage: 10M calls/month included
Support: Dedicated CSM + Slack channel
Total: $235K/year
```

**Large Enterprise (5,000 users):**
```
Base: $200K/year (500 users included)
Additional users: 4,500 × $100/year = $450K
Usage: 100M calls/month included
Storage: 10TB included
Support: Dedicated CSM + quarterly business reviews
Custom: On-premise deployment + white-label
Total: $850K/year (discounted to $650K)
```

**What's Included:**
- Everything in Pro
- **Plus:**
  - Heart (moral reasoning)
  - Differentiable memory (learnable retrieval)
  - Imagination engine (multi-step planning)
  - SSO (Okta, Azure AD, Google Workspace)
  - RBAC (role-based access control)
  - Audit logs (full compliance trail)
  - On-premise deployment option
  - White-label option (your branding)
  - Dedicated CSM (Customer Success Manager)
  - Priority support (4-hour response time)
  - 99.9% uptime SLA
  - Custom integrations (Salesforce, SAP, etc.)
  - Quarterly business reviews
  - Custom development (hourly rate)

**Value Proposition:**
```
Without AetherMind:
- SaaS tools: $1,000/user/year × 500 = $500K/year
- Consulting: $500K/year (implementations)
- Time saved: $2M/year (productivity gains)
Total value: $3M/year

With AetherMind Enterprise: $235K/year

Savings: $2.765M/year (92% cost reduction)
ROI: 11.8× return on investment
```

**Contract Terms:**
- Minimum: 1-year contract
- Payment: Annual (discount) or monthly
- Auto-renew: Yes (with 30-day notice to cancel)
- Price increases: Max 10% per year

---

## 📊 Value-Based Pricing Calculator

### Customer Savings Analysis

**Step 1: Calculate Current Costs**
```
Tool subscriptions:
- Research tools (Elicit, etc.): $X/month
- Productivity tools (Notion, etc.): $Y/month
- Development tools (Copilot, etc.): $Z/month
- Industry-specific tools: $W/month
Total tool costs: $(X+Y+Z+W) × 12 = $A/year

Time savings:
- Hours saved per user per month: H hours
- Hourly rate: $R/hour
Total time value: (H × $R × Users × 12) = $B/year

Total annual value: $A + $B = $V/year
```

**Step 2: Calculate AetherMind Price**
```
Value-capture rate: 10-30% of savings
AetherMind price: $V × 0.10 to 0.30

Example:
Company: 100 users
Tool costs: $100K/year
Time savings: 10 hours/user/month @ $100/hour = $1.2M/year
Total value: $1.3M/year

AetherMind price: $1.3M × 0.15 = $195K/year
(Customer saves $1.105M/year, 85% cost reduction)
```

**Step 3: Validate Against Tiers**
```
$195K/year for 100 users = $1,950/user/year

This is Enterprise tier pricing (makes sense for 100 users)
Standard Enterprise: $95K/year (base case)
Negotiated: $195K/year (justified by $1.3M value)

Discount from value: $1,300K value → $195K price = 85% discount
Customer perception: "This is a steal"
```

---

## 💡 Pricing Psychology

### Anchoring: Start High

**Website Pricing Page:**
```
Option A (Enterprise): $50,000/year ← ANCHOR (show first)
Option B (Pro): $2,400/year ← RELATIVE BARGAIN
Option C (Free): $0/year ← ENTRY POINT

Psychology: $2,400 feels cheap compared to $50,000
```

### Good-Better-Best

**Always show 3 options:**
```
Free: Basic features (anchors at $0)
Pro: Most popular (highlight this) ← 80% choose this
Enterprise: Premium (anchors at high price)

Result: Most users choose middle option (Pro)
```

### Decoy Pricing

**Make Pro look attractive:**
```
Pro: $2,400/year (10 users, 5K calls/day)
Enterprise: $50,000/year (10 users, unlimited calls) ← DECOY

Enterprise offers only 2× the calls but 20× the price
→ Pro looks like amazing value
→ Most choose Pro
→ Large companies still choose Enterprise (need SSO, etc.)
```

### Loss Aversion

**Free trial messaging:**
```
Don't say: "Start free trial"
Do say: "Try Pro features free for 30 days"

Don't say: "Trial ended, upgrade now"
Do say: "You're about to lose these features: [list]. Keep them for $2,400/year?"

Psychology: Fear of losing features > desire to gain features
```

---

## 🔄 Pricing Experiments

### Test 1: Annual vs Monthly

**Hypothesis:** Annual prepay gets 30%+ adoption if discounted

**Test:**
```
Monthly: $249/month × 12 = $2,988/year
Annual: $2,400/year (save $588 = 20% discount)

Expectation:
- 70% choose monthly (cash flow flexibility)
- 30% choose annual (cost savings)

Reality (run for 90 days):
- If >30% choose annual → Keep discount
- If <30% choose annual → Increase discount to 25%
```

### Test 2: Usage-Based vs Flat

**Hypothesis:** Predictable flat pricing beats usage-based for SMB

**Test:**
```
Group A: Flat $2,400/year (5K calls/day)
Group B: Usage-based $0.001 per call (no limit)

For user with 3K calls/day:
- Flat: $2,400/year
- Usage: 3K × 365 × $0.001 = $1,095/year (cheaper)

Expectation:
- Group A has higher LTV (predictable, less churn)
- Group B has lower LTV (bill shock, churn)

Reality (run for 180 days):
- Measure: LTV, churn, NPS
- Choose winner
```

### Test 3: Per-Seat vs Per-Agent

**Hypothesis:** Per-agent pricing better for teams

**Test:**
```
Group A: $2,400/year (10 users on 1 agent)
Group B: $240/user/year (10 users × $240 = $2,400)

Expectation:
- Group A: Higher NPS (predictable, no seat audits)
- Group B: Lower NPS (seat counting friction)

Reality (run for 90 days):
- Measure: NPS, expansion rate
- Choose winner
```

---

## 🚀 Pricing Evolution Roadmap

### Year 1 (2026): Simple, Low-Friction

**Strategy:** Maximize adoption, worry about monetization later

**Pricing:**
```
Free: Unlimited (just self-host)
Pro: $1,200/year flat (no usage limits)
Enterprise: Custom (starting $50K)

Goal: 100 Pro customers, 20 Enterprise
Revenue: $1.2K × 100 + $200K × 20 = $4.12M
```

---

### Year 2 (2027): Optimize Tiers

**Strategy:** Increase ARPU, introduce usage tiers

**Pricing:**
```
Free: Unlimited (self-host)
Pro: $1,200 (Starter), $2,400 (Growth), $5,000 (Scale)
Enterprise: Starting $50K, up to $500K

Goal: 5,000 Pro customers ($2.5K avg), 200 Enterprise ($200K avg)
Revenue: $2.5K × 5K + $200K × 200 = $52.5M
```

---

### Year 3 (2028): Value-Based Pricing

**Strategy:** Charge based on customer savings

**Pricing:**
```
Free: Unlimited (self-host)
Pro: $1,200-5,000 (usage-based)
Enterprise: 10-30% of customer savings (custom)

Example Enterprise deal:
Customer saves $2M/year → We charge $400K/year (20%)
Customer perception: $1.6M net savings (80% cost reduction)

Goal: 20,000 Pro ($3K avg), 1,000 Enterprise ($250K avg)
Revenue: $3K × 20K + $250K × 1K = $310M
```

---

### Year 4 (2029): Platform Fees

**Strategy:** Take % of marketplace transactions

**Pricing:**
```
Core Product: Same as Year 3
Marketplace: 30% of adapter/knowledge pack sales

Example:
Developer sells "Legal Knowledge Pack" for $499/month
1,000 customers buy it
GMV: $499 × 1K × 12 = $5.99M/year
Our cut: $5.99M × 30% = $1.8M/year

Goal: $35M marketplace revenue
Total: $310M core + $35M marketplace = $345M
```

---

### Year 5 (2030): Multi-Product Expansion

**Strategy:** Add new products with separate pricing

**Products:**
```
1. AetherMind Core: $1.2K-5K/year (Pro), $50K-500K (Enterprise)
2. AetherMind Vertical (Legal/Finance/Health): $10K-50K/year
3. AetherMind Platform (for ISVs to build on): $100K-1M/year
4. AetherMind Enterprise (on-prem, unlimited): $1M-5M/year

Total: $4B+ ARR across all products
```

---

## 📊 Pricing Comparison Matrix

### AetherMind vs Competitors

| Provider | Model | Price | What You Get |
|----------|-------|-------|--------------|
| **OpenAI** | API | $10-30 per 1M tokens | GPT-4 access, no memory, no learning |
| **Anthropic** | API | $15 per 1M tokens | Claude access, 200K context, no learning |
| **Google** | API | $7 per 1M tokens | Gemini access, 1M context, no learning |
| **LangChain** | Framework | Free (OSS) + $500/month (LangSmith) | DIY framework, you build and host |
| **Harvey** | Vertical | $5K-10K/month | Legal-specific, static model |
| **AetherMind Free** | Self-host | $0 | 70% features, unlimited usage, community support |
| **AetherMind Pro** | Managed | $1.2K-5K/year | 90% features, continuous learning, team collaboration |
| **AetherMind Enterprise** | Managed/On-prem | $50K-500K/year | 100% features, SSO, SLA, dedicated support |

**Price Comparison (Same Usage):**
```
Task: 1M inference calls over 1 year

OpenAI (GPT-4):
- 1M calls × 200 tokens avg × $0.03 per 1K = $6,000

AetherMind Pro:
- 1M calls / 365 = 2,740 calls/day
- Fits in Growth tier: $2,400/year
- Savings: $3,600 (60% cheaper)

Plus: AetherMind learns and improves, OpenAI stays static
```

---

## 🎯 Pricing Decision Framework

### Should You Price Higher or Lower?

**Price Higher If:**
- High switching costs (integrated into workflow)
- Clear ROI (customer saves $X, you charge $0.1X)
- Enterprise buyers (have budget)
- Mission-critical (uptime matters)

**Price Lower If:**
- Low switching costs (easy to churn)
- Unclear ROI (nice-to-have, not must-have)
- SMB buyers (price-sensitive)
- Competitive market (alternatives exist)

**AetherMind Position:**
```
Year 1-2: Price lower (gain market share, prove value)
Year 3+: Price higher (customers locked in through learning, clear ROI)

Rationale: Learning creates switching cost over time
After 1 year of learning, customer cannot switch (too much accumulated intelligence)
```

---

## 🎯 Conclusion

### Pricing Principles

1. **Value-Based:** Charge % of customer savings (10-30%)
2. **Transparent:** Public pricing, no hidden fees
3. **Fair:** Pay for value, not arbitrary limits
4. **Simple:** Easy to understand and predict
5. **Scalable:** Grows with customer success

### The Pricing Journey

**Year 1:** Low prices to gain adoption ($1.2K-5K Pro, $50K-200K Enterprise)  
**Year 2:** Optimize tiers based on usage data ($1.2K-5K Pro, $50K-500K Enterprise)  
**Year 3:** Value-based pricing (10-30% of customer savings)  
**Year 4:** Platform fees (30% marketplace revenue)  
**Year 5:** Multi-product expansion ($1M+ for enterprise platform)

### Success Metrics

**Key Metrics:**
- ARPU (Average Revenue Per User): $2.5K (Year 1) → $10K (Year 5)
- LTV (Lifetime Value): $12.5K (Year 1) → $50K (Year 5)
- LTV/CAC: 50× (Year 1) → 150× (Year 5)
- NDR (Net Dollar Retention): 110% (Year 1) → 130% (Year 5)

**Goal:** Capture 10-30% of value created while ensuring 70-90% savings for customers.

---

**Document Date:** January 4, 2026  
**Pricing Status:** Approved for Year 1  
**Next Review:** Q4 2026 (optimize based on data)
