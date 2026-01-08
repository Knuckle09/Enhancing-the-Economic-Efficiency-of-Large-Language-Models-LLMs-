# Enhancing-the-Economic-Efficiency-of-Large-Language-Models-LLMs-
Nimbus AI is a reinforcement-learning powered framework that reduces LLM token usage, preserves semantic meaning, and intelligently routes prompts across cloud and local models. It delivers up to 30% cost savings while maintaining >90% similarity, with full analytics and a modern UI.

![IMG-20250523-WA0014 1](https://github.com/user-attachments/assets/ea81d353-4ed2-409b-a317-45a019ceeaef)

<div align="center"> <img src="NIMBUS AI" width="200">
Nimbus AI
Enhancing the Economic Efficiency of Large Language Models (LLMs)
Reinforcement-Learning Powered Prompt Optimization + Multi-LLM Routing
<br>










</div>
📌 Project Overview

Nimbus AI is an advanced LLM Optimization Framework that significantly reduces operational costs of Large Language Models by:

✔️ Compressing prompts using summarization + heuristic pruning

✔️ Preserving semantics using embedding-based similarity

✔️ Selecting the most economical LLM using reinforcement learning

✔️ Supporting both cloud models (Gemini, OpenAI, Claude) and local models (LLaMA, Phi-3, CodeLlama, Qwen)

✔️ Providing a full analytics dashboard for transparency

According to the published research paper in IJSREM (Vol. 09 Issue 12) :

Nimbus AI reduces tokens by ~30% while maintaining >90% semantic similarity
Cost savings reach up to ~50% when combined with multi-model routing.

This repository includes source code, research paper, dashboards, UI, evaluation charts, and publication certificates.

🏆 Publication Certificates
<img width="200" height="200" alt="e- Certificate IJSREM Prof  Farzana Nadaf" src="https://github.com/user-attachments/assets/258a74cf-6011-4a67-bb5c-73f9aacbf32e" />
<img width="200" height="200" alt="e- Certificate IJSREM Sai Samarth Budihal" src="https://github.com/user-attachments/assets/ef668644-d20a-4232-95a7-9aa29fd1f43b" />
<img width="200" height="200" alt="e- Certificate IJSREM Sughnva Chappar" src="https://github.com/user-attachments/assets/3488039a-c026-4bd2-9147-3162c8fbe605" />
<img width="200" height="200" alt="e- Certificate IJSREM Suprit Mundagod" src="https://github.com/user-attachments/assets/e4280999-26c1-43de-9926-c28734e6e78b" />
<img width="200" height="200" alt="e- Certificate IJSREM Vishwanath Kotyal" src="https://github.com/user-attachments/assets/f964ad9b-71a9-4e7a-9a48-f8e6bdea92b1" />


📚 Research Paper

📄 Full Published Paper (IJSREM 2025)
Enhancing the Economic Efficiency of Large Language Models (LLMs)
👉 Available in repository: /docs/Enhancing_the_Economic_Efficiency.pdf

🧠 System Architecture
![final architecture](https://github.com/user-attachments/assets/f9c1c8e1-c95f-402a-8c32-beb89d5c23e4)

The architecture includes:

Input Pre-Processing

Summarization + Heuristic Pruning

Token Cost Estimation

RL Training Loop

Prompt Optimizer

Multi-Model Router

Response Analyzer

Feedback Engine

🖥️ User Interface (Frontend)
🔹 Auto Mode
<img width="200" height="200" alt="Frontend" src="https://github.com/user-attachments/assets/879e2bbe-27a9-4d76-bdf2-cd5920b58ed5" />

🔹 Manual Model Selection
<img width="200" height="200" alt="Frontend_2" src="https://github.com/user-attachments/assets/79a9f36a-f008-4d73-8c43-9c47968a6142" />

📊 Evaluation & Results

All results are derived from your published paper and dashboard screenshots.

🔹 Token Reduction vs Similarity
<img width="200" height="200" alt="token_reduction_vs_similarity" src="https://github.com/user-attachments/assets/26158ae0-ddf2-4488-9477-54db99f0cab9" />

🔹 LLM Response Metrics
<img width="200" height="200" alt="llm_response_metrics" src="https://github.com/user-attachments/assets/cb32e54e-0302-4a1b-a20d-816a0e0f8fb5" />

🔹 Performance by Prompt Type
<img width="200" height="200" alt="metrics_by_prompt_type" src="https://github.com/user-attachments/assets/a3dc465c-c202-4083-a0d9-183edf3d59c8" />

🔹 Analytics Dashboard
<img width="200" height="200" alt="Results_1" src="https://github.com/user-attachments/assets/e09abd96-542b-4a10-8c07-c759e3649c8c" />

🔹 Cost Analysis
<img width="200" height="200" alt="Results_2" src="https://github.com/user-attachments/assets/c6004f39-2455-4d4b-adc9-b83a606a46c7" />

🔹 Detailed Prompt Analysis
<img width="200" height="200" alt="Results_3" src="https://github.com/user-attachments/assets/37339098-d92a-4d9e-ae90-dbd58b0bec9a" />

📈 Comparison with Existing Systems

(From IJSREM Paper Table 1 & 2)

Comparison Table
Feature	Existing Tools	RL Systems	Summarizers	Nimbus AI
Token Reduction	Low	Very Low	Medium	~30%
Meaning Preservation	Low	Medium	Medium	> 0.90
Reinforcement Learning	No	Yes	No	Yes
Multi-LLM Routing	No	No	No	Yes
Cost Reduction	<10%	Minimal	~15%	~50%
Dashboard	No	No	No	Yes
Novelty Summary
Contribution	Description
RL-Based Optimization	Reward-driven rewriting ensures both compression + quality
Multi-LLM Cost Routing	Selects cheapest + best model automatically
Semantic Validator	Ensures ≥ 90% similarity
Token Cost Estimator	Estimates cost before inference
Visual Dashboard	Full transparency in savings
🔧 Installation
Backend
cd backend
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
python api.py

Frontend
cd frontend
npm install
npm run dev


Then visit:

http://localhost:5173

📁 Repository Structure
Nimbus-AI/
│
├── backend/
├── frontend/
├── docs/
│   └── Enhancing_the_Economic_Efficiency.pdf
├── certificates/
├── results/
├── architecture/
├── README.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── .github/
│   ├── ISSUE_TEMPLATE.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│        └── ci.yml
└── LICENSE
