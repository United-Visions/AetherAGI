# Open Core Strategy for AetherMind

**Date:** January 4, 2026  
**Decision:** 70% Open Source, 30% Proprietary  
**Model:** MongoDB-style Open Core (not full open source)  
**Timeline:** Open source in Month 6 (after Series A)

---

## 🎯 Executive Summary

**Decision:** AetherMind will use an **Open Core** business model.

**Rationale:**
1. **Defensibility:** Proprietary moat in meta-controller, self-mod, ToolForge
2. **Innovation:** Community can extend Brain, Mind, Body
3. **Trust:** Transparency in reasoning layer (safety-critical)
4. **Revenue:** Enterprise features remain proprietary

**Open Source (70%):**
- Brain (logic_engine.py, safety_inhibitor.py, core_knowledge_priors.py)
- Mind (episodic_memory.py, vector_store.py, promoter.py)
- Body (adapter_base.py, chat_ui.py, basic adapters)
- Orchestrator core (router.py, session_manager.py, active_inference.py)

**Proprietary (30%):**
- Meta-controller (UCB bandit, RL training data)
- Self-modification system (git-based sandboxing)
- ToolForge (autonomous tool generation)
- Heart (moral reasoning)
- Differentiable Memory (learnable retrieval)
- Imagination Engine (multi-step planning)
- Enterprise features (SSO, RBAC, audit logs)

---

## 📊 Open Core Precedents (Proven Model)

### MongoDB (IPO 2017, Market Cap $30B+)

**Open Source:**
- Core database engine
- Query language
- Basic replication

**Proprietary:**
- Atlas (managed cloud)
- Enterprise security
- Advanced monitoring
- Ops Manager

**Outcome:**
- $1.3B revenue (2023)
- 80% from proprietary Atlas
- Community built ecosystem (drivers, tools, content)
- Enterprise buyers trust open core (can self-host if needed)

**Lesson:** Open the engine, charge for operations

---

### Elastic (IPO 2018, Market Cap $8B+)

**Open Source:**
- Elasticsearch (core search)
- Logstash (data ingestion)
- Kibana (visualization)

**Proprietary:**
- Security features (SSO, RBAC)
- Machine learning
- Alerting
- Cloud (Elastic Cloud)

**Outcome:**
- $1B+ revenue (2023)
- 70% from proprietary cloud
- 500K+ community deployments
- Network effects from open source

**Lesson:** Open the search, charge for security & ML

---

### GitLab (IPO 2021, Market Cap $6B+)

**Open Source:**
- Core Git platform
- Basic CI/CD
- Issue tracking

**Proprietary:**
- Advanced CI/CD
- Security scanning
- Compliance features
- Enterprise support

**Outcome:**
- $500M+ revenue (2023)
- 30,000+ paying customers
- Self-hosted builds trust
- Cloud option for convenience

**Lesson:** Open the core, charge for compliance & scale

---

### Redis Labs (Acquired $2B+)

**Open Source:**
- Redis core (in-memory database)
- Basic data structures

**Proprietary:**
- Redis Enterprise (clustering)
- Active-active geo-replication
- Advanced modules (search, graph, timeseries)

**Outcome:**
- Acquired by private equity at $2B+ valuation
- Most popular in-memory database
- Community contributed modules

**Lesson:** Open the core, charge for scale & modules

---

## 🏗️ AetherMind Open Core Architecture

### Open Source Components (70%)

#### 1. Brain (Reasoning Core)

**Files:**
```
brain/
├── __init__.py ✅ Open
├── logic_engine.py ✅ Open
├── safety_inhibitor.py ✅ Open (critical for trust)
├── core_knowledge_priors.py ✅ Open
└── README.md ✅ Open
```

**Rationale:**
- **Trust:** Safety layer must be transparent (auditable)
- **Innovation:** Community can improve reasoning
- **Differentiation:** Not our moat (architecture is moat, not code)

**What stays proprietary:**
- Training data for priors
- Fine-tuning parameters
- Production model weights

---

#### 2. Mind (Memory Systems - Partial)

**Open:**
```
mind/
├── __init__.py ✅ Open
├── episodic_memory.py ✅ Open
├── vector_store.py ✅ Open (interface only)
└── promoter.py ✅ Open
```

**Proprietary:**
```
mind/
├── differentiable_store.py ❌ Proprietary (learnable retrieval)
└── ingestion/ ❌ Proprietary (optimized pipelines)
```

**Rationale:**
- **Open:** Basic memory (anyone can build chatbot with memory)
- **Proprietary:** Learnable retrieval (competitive advantage)

---

#### 3. Body (Interface Layer - Partial)

**Open:**
```
body/
├── __init__.py ✅ Open
├── adapter_base.py ✅ Open (interface)
├── adapters/
│   ├── chat_ui.py ✅ Open
│   ├── automotive.py ✅ Open (example)
│   └── smart_home.py ✅ Open (example)
└── README.md ✅ Open
```

**Proprietary:**
```
body/adapters/
├── toolforge_adapter.py ❌ Proprietary (UNIQUE CAPABILITY)
├── practice_adapter.py ❌ Proprietary (skill learning)
└── enterprise_adapters/ ❌ Proprietary (Salesforce, SAP, etc.)
```

**Rationale:**
- **Open:** Interface + basic adapters (developers can extend)
- **Proprietary:** Autonomous tool generation (our secret sauce)

---

#### 4. Orchestrator (Coordination - Partial)

**Open:**
```
orchestrator/
├── __init__.py ✅ Open
├── router.py ✅ Open
├── session_manager.py ✅ Open
├── active_inference.py ✅ Open
└── README.md ✅ Open
```

**Proprietary:**
```
orchestrator/
├── meta_controller.py ❌ Proprietary (UCB bandit, RL)
├── self_mod.py ❌ Proprietary (self-modification)
├── planning_scheduler.py ❌ Proprietary (long-horizon)
└── auth_manager.py ❌ Proprietary (enterprise SSO)
```

**Rationale:**
- **Open:** Basic orchestration (shows architecture)
- **Proprietary:** Intelligence (meta-controller), self-improvement, security

---

#### 5. What's 100% Proprietary

```
heart/ ❌ ALL PROPRIETARY
├── heart_orchestrator.py (moral reasoning)
├── moral_emotion.py (virtue ethics)
├── reward_model.py (flourishing prediction)
└── virtue_memory.py (character development)

curiosity/ ❌ ALL PROPRIETARY
├── surprise_detector.py (novelty detection)
├── research_scheduler.py (autonomous learning)
└── solo_ingestor.py (self-directed knowledge)

perception/ ❌ ALL PROPRIETARY
├── eye.py (vision)
├── transcriber.py (audio)
└── mcp_client.py (multimodal)

monitoring/ ❌ ALL PROPRIETARY
├── dashboard.py (observability)
└── kill_switch.py (safety critical)
```

**Rationale:** These are **differentiators** that justify enterprise pricing.

---

## 💰 Business Model

### Free Tier (Open Source)

**What You Get:**
- All open source code (70% of system)
- Self-hosted deployment
- Community support (GitHub, Discord)
- Basic adapters (chat, simple tools)

**Limitations:**
- No meta-controller (agent is reactive, not proactive)
- No self-modification (cannot improve itself)
- No ToolForge (cannot generate new tools)
- No Heart (no moral reasoning)
- No enterprise features (SSO, RBAC, audit)

**Who Uses This:**
- Developers learning the system
- Researchers building on architecture
- Startups with engineering resources
- Non-commercial projects

**Estimated Users:** 10,000+ by Year 2

**Revenue Impact:** $0 direct, but:
- Builds community
- Generates content (tutorials, blog posts)
- Finds bugs (crowdsourced testing)
- Creates hiring pipeline (best contributors → employees)

---

### Pro Tier (Proprietary + Managed)

**Price:** $1,200-5,000/year per agent

**What You Get:**
- Everything in Free tier
- Meta-controller (proactive agent)
- ToolForge (autonomous tool generation)
- Self-modification (continuous improvement)
- Cloud hosting (no ops required)
- Email support

**Who Uses This:**
- Solo developers
- Small teams (1-10 people)
- Startups needing autonomy
- Consultants building for clients

**Estimated Users:** 5,000 by Year 2

**Revenue:** 5,000 × $2,500 avg = $12.5M/year

---

### Enterprise Tier (Full Platform)

**Price:** $50,000-500,000/year

**What You Get:**
- Everything in Pro tier
- Heart (moral reasoning)
- Differentiable Memory (learnable retrieval)
- Imagination Engine (multi-step planning)
- Enterprise adapters (Salesforce, SAP, etc.)
- SSO (Okta, Azure AD)
- RBAC (role-based access)
- Audit logs (compliance)
- SLA (99.9% uptime)
- Dedicated support
- Custom development

**Who Uses This:**
- Fortune 500 companies
- Government agencies
- Healthcare systems
- Financial institutions

**Estimated Users:** 200 by Year 2

**Revenue:** 200 × $200K avg = $40M/year

---

### Total Addressable Revenue (Year 2)

```
Free:       10,000 users × $0 = $0
Pro:        5,000 users × $2.5K = $12.5M
Enterprise: 200 users × $200K = $40M
-------------------------------------------
Total:                           $52.5M/year
```

**Comparison (Fully Proprietary):**
```
Starter:   5,000 × $1K = $5M (lost free users)
Pro:       2,000 × $5K = $10M (higher friction)
Enterprise: 150 × $250K = $37.5M (same)
-------------------------------------------
Total:                    $52.5M/year (SAME)
```

**But with Open Core:**
- 10,000 community evangelists (vs 0)
- 500+ GitHub contributors (vs 0)
- 100+ blog posts/tutorials (vs 0)
- 10× more brand awareness
- 5× more enterprise trust (can audit code)

**Conclusion:** Same revenue, 10× more moat

---

## 🛡️ Competitive Moat Analysis

### Why Open Core Strengthens Moat

#### 1. Community Contributions

**Example: Adapter Ecosystem**
```
Open source body/adapter_base.py
→ Community builds 100+ adapters
→ We don't have to build them
→ Network effects (more adapters = more valuable)
→ We offer proprietary "Enterprise Adapter Pack" ($10K/year)
```

**Outcome:** 
- Community does 80% of adapter work
- We monetize the 20% enterprises need

---

#### 2. Security Through Transparency

**Example: Safety Inhibitor**
```
Problem: "How do we trust your AI won't go rogue?"

Closed source: "Trust us."
Open source: "Audit our code. Here's safety_inhibitor.py."

Result:
- Red team finds bug in Month 2
- We fix it
- Enterprise buyer now trusts us (saw our response)
```

**Outcome:**
- Higher enterprise close rates (80% vs 40%)
- Faster sales cycles (3 months vs 9 months)

---

#### 3. Talent Acquisition

**Example: Hiring from Community**
```
Open source → 500 contributors
→ Identify top 10 (best code, most commits)
→ Hire them (already know system)
→ Save 6 months onboarding

vs

Closed source → Job post → Interviews → Hire
→ 6 months to productivity
```

**Outcome:**
- Hire faster (1 month vs 6 months)
- Hire better (proven track record)
- Lower risk (already contributed)

---

#### 4. Distribution via Self-Hosting

**Example: Large Enterprise**
```
Concern: "We can't send data to your cloud (compliance)"

Closed source: "Sorry, we don't do on-prem."
Lost deal: $500K/year

Open core: "Deploy in your VPC. Here's the code."
Closed deal: $500K/year
Later: "Actually, cloud is easier. Migrate?"
Upsell: $750K/year (managed)
```

**Outcome:**
- Land deal with self-hosted
- Upsell to managed (higher margin)

---

### Why Keeping 30% Proprietary Protects Moat

#### 1. Cannot Fork into Competitor

**Scenario: VC-backed competitor tries to fork**

```
They get (70% open):
✓ Brain (reasoning)
✓ Mind (basic memory)
✓ Body (basic adapters)
✓ Orchestrator (basic routing)

They DON'T get (30% proprietary):
✗ Meta-controller (proactive intelligence)
✗ Self-modification (continuous improvement)
✗ ToolForge (autonomous tool generation)
✗ Heart (moral reasoning)
✗ Differentiable Memory (learnable retrieval)
✗ Imagination Engine (planning)

Result:
- They have a chatbot (commodity)
- We have an autonomous agent (differentiated)
- They cannot compete on features
```

**Protection:** 30% proprietary makes fork uncompetitive

---

#### 2. Cannot Replicate Proprietary Components

**Why Meta-Controller is Hard:**
```
Not just code (that's easy to copy)
Requires:
- RL training data (6 months of agent interactions)
- Hyperparameter tuning (100+ experiments)
- Domain expertise (PhD-level RL)
- Compute (10,000 GPU hours)

Even if they see the code, they can't replicate performance without data.
```

**Moat:** Data moat > Code moat

---

#### 3. Network Effects from Proprietary Data

**Example: ToolForge**
```
Month 1: Generated 50 tools
Month 6: Generated 500 tools
Month 12: Generated 2,000 tools

Each tool used by all customers (network effect)
Each tool failure improves algorithm (data flywheel)

Competitor starting from scratch:
- Has 0 tools
- Has no failure data
- Has no improvement loop

Time to parity: 1+ year
```

**Moat:** Accumulated learning

---

## 🚀 Rollout Strategy

### Phase 1: Internal Alpha (Months 1-3, Private)

**Status:** Complete stealth development

**What's Built:**
- All core components (Brain, Mind, Body, Orchestrator)
- Proprietary features (Meta-controller, ToolForge, etc.)
- Basic documentation

**Who Uses It:**
- Internal team only
- 5-10 pilot customers (NDA)

**Goal:** Prove product-market fit

---

### Phase 2: Private Beta (Months 3-6, Invite-Only)

**Expand Pilot:**
- 50 beta customers
- Collect feedback
- Iterate rapidly

**Prepare for Open Source:**
- Refactor code (separate open from proprietary)
- Write comprehensive docs
- Create contribution guidelines
- Set up governance (steering committee)

**Goal:** Ready for public launch

---

### Phase 3: Open Core Launch (Month 6, Public)

**Simultaneous Announcements:**
1. **Open Source Release** (GitHub, HN, Reddit)
   - 70% of codebase open sourced
   - Apache 2.0 license (permissive)
   - Contribution guidelines
   
2. **Commercial Launch** (Press release, website)
   - Pro tier: $1,200-5,000/year
   - Enterprise tier: $50K-500K/year
   - Managed cloud option
   
3. **Series A Announcement** (TechCrunch, etc.)
   - $5-10M raise
   - Open core strategy validated by investors

**Messaging:**
- "Open Core AI for the Enterprise"
- "Transparency you can audit, intelligence you can trust"
- "Self-host or cloud, you choose"

---

### Phase 4: Community Growth (Months 6-12)

**Objectives:**
- 1,000+ GitHub stars (Month 6)
- 5,000+ GitHub stars (Month 9)
- 10,000+ GitHub stars (Month 12)
- 100+ contributors
- 500+ adapters in ecosystem

**Programs:**
- Hacktoberfest participation
- Adapter bounty program ($500 per accepted adapter)
- Annual conference (AetherMind Summit)
- Developer grants ($10K to best projects)

**Goal:** Become default open source agent framework

---

## 📜 Licensing Strategy

### Open Source License: Apache 2.0

**Why Apache 2.0 (not GPL):**
- **Permissive:** Companies can use without fear
- **Commercial-friendly:** Can mix with proprietary code
- **Patent protection:** Explicit patent grant
- **Standard:** Most businesses already approved it

**What This Allows:**
- Use in commercial products ✓
- Modification ✓
- Distribution ✓
- Sublicensing ✓
- Patent use ✓

**What This Requires:**
- Include license text
- Include copyright notice
- Note any changes made

**What This DOESN'T Require:**
- Open source derivative works (vs GPL)
- Share improvements (vs AGPL)
- Copyleft (vs GPL)

---

### Proprietary License: Custom Commercial

**For Closed Source Components:**
```
AetherMind Enterprise License Agreement (ELA)

1. Grant: Limited, non-exclusive, non-transferable license
2. Restrictions: No reverse engineering, no redistribution
3. Fees: Annual subscription per agent
4. Support: Email support (Pro), dedicated (Enterprise)
5. Updates: Included in subscription
6. Warranty: Limited (standard SaaS)
7. Liability: Capped at fees paid
8. Termination: 30 days notice
```

**Key Points:**
- Cannot share proprietary code
- Cannot build competitor
- Can self-host (Enterprise tier)
- Can view source (for audit, not modification)

---

## 🎯 Competitive Landscape

### Fully Open Source Competitors

**Example: Langchain, AutoGPT, OpenDevin**

**Advantages:**
- Free
- Community-driven
- Transparent

**Disadvantages:**
- No business model → No resources → Slow development
- No enterprise features (SSO, RBAC, support)
- No optimization (no incentive)

**AetherMind Position:**
- Match on transparency (Brain is open)
- Beat on features (proprietary components)
- Beat on quality (funded team, not volunteers)

---

### Fully Proprietary Competitors

**Example: OpenAI, Anthropic, Google**

**Advantages:**
- Huge resources
- Large models
- Brand recognition

**Disadvantages:**
- Black box (cannot audit)
- Cloud-only (compliance issues)
- Expensive (20-60× our price)

**AetherMind Position:**
- Match on capabilities (Gen 2 architecture)
- Beat on trust (open core, auditable)
- Beat on cost (95% cheaper)
- Beat on flexibility (self-host option)

---

### Other Open Core Competitors

**Example: Hugging Face (not exactly same, but similar model)**

**Advantages:**
- Brand recognition
- Large community
- Investor backing

**Disadvantages:**
- Focused on models, not agents
- No autonomous capabilities
- No continuous learning

**AetherMind Position:**
- Different market (agents vs models)
- Complementary (can use HF models in AetherMind)
- Potential partnership

---

## 🎓 Lessons from Open Core Failures

### Case Study 1: Cockroach Labs (struggled with open core)

**Mistake:**
- Open sourced too much (including scale features)
- Companies used open version, never upgraded

**Lesson:**
- Keep enterprise features proprietary from Day 1
- Don't open source what enterprises need most

**Applied to AetherMind:**
- ToolForge (scale feature) = Proprietary ✓
- Meta-controller (intelligence) = Proprietary ✓
- Heart (compliance) = Proprietary ✓

---

### Case Study 2: HashiCorp (changed license 2023)

**Mistake:**
- Apache 2.0 → BSL (Business Source License)
- Community backlash
- Fork created (OpenTofu)

**Why They Changed:**
- Cloud providers (AWS) competing with managed Terraform
- Not making enough on cloud version

**Lesson:**
- Choose right license from Day 1
- Don't change license later (breaks trust)

**Applied to AetherMind:**
- Apache 2.0 for open parts (not changing) ✓
- Proprietary from Day 1 for closed parts ✓
- Clear delineation (no confusion) ✓

---

### Case Study 3: Redis (also changed license 2024)

**Mistake:**
- BSD → SSPL (Server Side Public License)
- Reason: AWS ElastiCache competing

**Lesson:**
- Cloud providers will use your open source
- Charge for managed service, not license

**Applied to AetherMind:**
- Expect AWS/Azure/GCP to offer managed AetherMind
- Beat them on:
  - Feature velocity (we innovate faster)
  - Specialized expertise (we know the system)
  - Proprietary enhancements (they can't replicate)

---

## 📊 Financial Projections

### Revenue (Open Core Model)

**Year 1:**
```
Free: 1,000 users × $0 = $0
Pro: 500 users × $2.5K = $1.25M
Enterprise: 20 users × $200K = $4M
Total: $5.25M
```

**Year 2:**
```
Free: 10,000 users × $0 = $0
Pro: 5,000 users × $2.5K = $12.5M
Enterprise: 200 users × $200K = $40M
Total: $52.5M
```

**Year 3:**
```
Free: 50,000 users × $0 = $0
Pro: 20,000 users × $3K = $60M
Enterprise: 1,000 users × $250K = $250M
Total: $310M
```

**Year 5:**
```
Free: 200,000 users × $0 = $0
Pro: 50,000 users × $3.5K = $175M
Enterprise: 5,000 users × $300K = $1.5B
Total: $1.675B
```

**IPO Readiness:** Year 4-5 at $1B+ ARR

---

### Costs (Open Core Model)

**Additional Costs vs Closed Source:**
- Community management: $500K/year (forum, Discord, events)
- Open source program office: $300K/year (governance, legal)
- Documentation: $200K/year (more extensive than closed)
Total extra: $1M/year

**Cost Savings vs Closed Source:**
- Marketing: -$2M/year (community evangelism)
- Sales: -$1M/year (self-serve adoption)
- Development: -$1M/year (community contributions)
Total savings: $4M/year

**Net Impact:** +$3M/year savings with open core

---

## ✅ Decision Matrix

| Factor | Fully Open | Open Core | Fully Closed |
|--------|-----------|-----------|--------------|
| Trust | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Revenue | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Community | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Defensibility | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Enterprise Sales | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Talent Acquisition | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed to Market | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **TOTAL** | **26** | **36** | **24** |

**Winner: Open Core** (36/40 stars)

---

## 🚀 Conclusion

### Why Open Core Wins for AetherMind

1. **Trust:** Enterprises can audit safety-critical code
2. **Moat:** Proprietary features (30%) are true differentiators
3. **Community:** 10,000+ evangelists by Year 2
4. **Revenue:** Same as closed source, but with network effects
5. **Talent:** Hire from community (faster, better, cheaper)
6. **Distribution:** Self-host option unlocks compliance-heavy customers

### What to Open (70%)

- Brain (trust in reasoning)
- Mind (basic memory)
- Body (interface + basic adapters)
- Orchestrator (core routing)

### What to Keep Closed (30%)

- Meta-controller (intelligence)
- ToolForge (autonomous capability)
- Self-modification (improvement)
- Heart (morality)
- Enterprise features (SSO, RBAC, audit)

### The Path Forward

**Month 6:** Open source launch (after Series A)  
**Year 2:** 10,000 community users, $52M revenue  
**Year 5:** 200,000 community users, $1.6B revenue, IPO-ready

**The vision:** *Build the most trusted and capable AI agent platform through open core transparency and proprietary innovation.*

---

**Document Date:** January 4, 2026  
**Decision:** APPROVED - Open Core (70/30 split)  
**Timeline:** Open source in Month 6 (Q2 2026)
