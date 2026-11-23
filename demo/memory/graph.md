# Temporal Knowledge Graph — Alex Rivera (Synthetic Benchmark Dataset)

## C1 · Daily Life & Personal Routines
- **G-01** `[V]` Alex Rivera lives with a rescue golden retriever named Luna who loves morning trail walks | sem: dog, pet, companion, furry friend, canine | eco: warm gratitude every morning
- **G-02** `[V]` Alex brews Ethiopian pour-over coffee using a V60 dripper at 07:30 before opening any IDE | sem: caffeine, morning ritual, espresso, java, brew
- **G-03** `[V]` Alex plays fingerstyle acoustic guitar to unwind after intense debugging sessions | sem: instrument, strings, melody, music hobby, chords
- **G-04** `[S]` Alex used to commute 90 minutes daily into downtown offices before switching to full remote indie work | sem: office job, subway, transit, old routine
- **G-05** `[V]` Alex trains for half-marathons on Tuesdays and Thursdays along the riverfront path | sem: running, jogging, cardio, exercise, endurance
- **G-06** `[V]` Alex cooks spicy Thai basil stir-fry and green curry from scratch on Sunday evenings | sem: dinner, recipe, asian food, culinary, wok
- **G-07** `[V]` Alex avoids running Docker Desktop on his 8 GB RAM ultrabook after repeated OOM freezes | sem: containers, memory bloat, low ram laptop, lightweight tooling

## C2 · Execution & Production Products
- **G-08** `[V]` PulseBill is Alex's indie micro-SaaS for automated B2B invoice reminders and Stripe reconciliation | sem: billing app, invoicing startup, saas revenue, payments product
- **G-09** `[V]` TrailMap is Alex's offline-first hiking GPS companion built with PWA and OPFS SQLite storage | sem: hiking app, mountain navigation, offline maps, trail tracker
- **G-10** `[V]` PulseBill production runs on Cloudflare Workers and D1 SQLite to stay under $5/month fixed cost | sem: edge hosting, cheap infra, serverless stack, cloud budget
- **G-11** `[V]` Maya Chen is the freelance UX designer collaborating part-time on TrailMap onboarding screens | sem: designer, ui partner, collaborator, figma
- **G-12** `[V]` Stripe webhook duplicate deliveries caused double-charged receipts until idempotency keys were enforced | sem: payment bug, double billing, duplicate charge, race condition
- **G-13** `[V]` TrailMap mountain users lose cellular signal above 2000m elevation requiring zero-network draft writes | sem: no signal, backcountry, airplane mode, dead zone, offline sync
- **G-14** `[S]` PulseBill v1 used a monolithic Postgres instance on a $40 VPS before migrating to edge SQLite | sem: legacy database, old server, expensive hosting
- **G-15** `[V]` Aggressive scraper bots from datacenter IPs spiked PulseBill D1 row reads by 40x in October | sem: crawler attack, ddos, traffic spike, scraping wave, waf

## C3 · Exploration, Architecture & AI Agents
- **G-16** `[V]` Alex researches zero-bloat local memory engines for coding agents without external vector databases | sem: rag alternative, lightweight memory, agent recall, no pinecone
- **G-17** `[V]` Separating planning agents (Thinkers) from file-scanning workers (Doers) cuts LLM input token spend sharply | sem: multi agent hierarchy, cost reduction, subagent pattern, token budget
- **G-18** `[V]` Inline semantic seeds attached at write time solve lexical mismatch without 100 MB ONNX embedding models | sem: synonym tagging, vocabulary gap, zero embedding, metaphor search
- **G-19** `[V]` Small 3B local models collapse into repetition loops when overloaded with multi-persona system prompts | sem: slm failure, tiny llm crash, prompt overload, attention sink
- **G-20** `[V]` Weekly episodic summaries need an anti-super-hub cap or they steal top retrieval slots from precise lessons | sem: hub dominance, retrieval noise, weekly recap, memory pollution
- **G-21** `[V]` Context pollution degrades frontend layout quality even when using the exact same foundation model | sem: dirty context, role isolation, same model variance, ui degradation
- **G-22** `[V]` SQLite WAL mode enables concurrent reads from a 3D visualization server while Hebbian edge updates run in O(1) | sem: concurrent db, fast writes, non blocking storage, graph persistence

## C4 · Support, Focus & Resilience
- **G-23** `[V]` Alex prefers direct, blunt technical feedback over polite filler or unsolicited wellness advice | sem: communication style, no fluff, straight talk, autonomy
- **G-24** `[V]` Financial runway pressure peaks at month-end when cloud renewals and rent align simultaneously | sem: money stress, tight budget, cashflow anxiety, bills | eco: quiet tension and determination to ship revenue
- **G-25** `[V]` Deep hyperfocus sprints last 4 to 6 hours and are broken immediately by noisy notifications | sem: flow state, concentration, deep work, distraction
- **G-26** `[V]` Seeing real paying customers renew PulseBill restores Alex's confidence after lonely build weeks | sem: validation, first revenue, motivation boost, milestone joy | eco: genuine relief and quiet pride
- **G-27** `[V]` Alex strictly separates private journal reflections from any repository or external API payload | sem: data privacy, local only, security boundary, zero leaks
- **G-28** `[S]` Alex used to try fixing every architecture debt item in a single sleepless weekend | sem: perfectionism, overworking, old habit, burnout trap
- **G-29** `[V]` Shipping small verifiable increments beats polishing theoretical docs for weeks without user feedback | sem: pragmatism, ship fast, lean startup, execution bias
- **G-30** `[V]` Writing short post-mortems after production incidents turns panic into durable engineering checklists | sem: incident review, blameless learning, resilience, playbook

## C1 · Extended Daily & Creative Roots
- **G-31** `[V]` Alex keeps a pocket notebook for sudden architecture diagrams drawn while drinking coffee away from screens | sem: analog notes, sketching, paper diagrams, offline thinking
- **G-32** `[V]` Rainy autumn afternoons in Valdivia are Alex's favorite atmosphere for writing long-form technical essays | sem: rain mood, writing weather, quiet focus, southern chile
- **G-33** `[V]` Alex repairs old mechanical keyboards and prefers tactile brown switches for long coding marathons | sem: hardware hobby, mechanical keyboard, tactile typing, desk setup
- **G-34** `[V]` Luna always nudges Alex's elbow around 18:30 reminding him to step outside for the evening walk | sem: dog routine, evening walk, pet signal, daily break | eco: warm smile every single evening

## C2 · Extended Production & Business Reality
- **G-35** `[V]` PulseBill targets small architecture and software studios frustrated by enterprise ERP complexity | sem: target audience, b2b niche, smb invoicing, product positioning
- **G-36** `[V]` Automated Cloudflare WAF rules cut PulseBill D1 billing exposure by 94% during bot waves | sem: cost shield, bot mitigation, serverless protection, waf savings
- **G-37** `[V]` TrailMap roadmap prioritizes offline emergency waypoints and battery-saving GPS polling intervals | sem: battery life, mountain safety, trail feature, mobile optimization
- **G-38** `[V]` Every public repository ships with an automated DLP privacy check before pushing to GitHub | sem: release checklist, privacy gate, safe publishing, security habit

## C3 · Extended Research & Open-Core Ecosystem
- **G-39** `[V]` HebbMesh and TokenScalpel form a complementary pair: local synaptic memory plus empirical token economics | sem: sister projects, open core duo, ecosystem strategy, github portfolio
- **G-40** `[V]` Publishing timestamped Zenodo preprints establishes verifiable authorship before social launches on Hacker News and Reddit | sem: doi protection, academic credibility, launch sequence, authorship proof
- **G-41** `[V]` Iris Starter Kit packages turnkey IDE hooks, nightly REM consolidation, and specialist agent factories as a paid developer kit | sem: commercial kit, gumroad funnel, turnkey pae, revenue model
- **G-42** `[V]` Deterministic FNV-1a spherical layouts make 3D memory maps immediately recognizable across sessions | sem: spatial memory, 3d coordinates, visual landmark, consistent sphere

## C4 · Extended Support & Sustainable Pace
- **G-43** `[V]` Alex treats every production mistake as fuel for a concise negative rule rather than self-blame | sem: growth mindset, reflexion habit, emotional resilience, constructive postmortem
- **G-44** `[V]` Sharing early breakthroughs with close builder friends helps counter the isolation of solo indie work | sem: peer support, founder community, honest pride, camaraderie | eco: uplifting connection with trusted friends
- **G-45** `[V]` Protecting quiet morning hours without Slack or email doubles Alex's creative output on hard problems | sem: morning deep work, distraction free, creative energy, maker schedule
