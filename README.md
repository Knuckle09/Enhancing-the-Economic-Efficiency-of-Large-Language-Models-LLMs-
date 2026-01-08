# Enhancing-the-Economic-Efficiency-of-Large-Language-Models-LLMs-
Nimbus AI is a reinforcement-learning powered framework that reduces LLM token usage, preserves semantic meaning, and intelligently routes prompts across cloud and local models. It delivers up to 30% cost savings while maintaining >90% similarity, with full analytics and a modern UI.

![IMG-20250523-WA0014 1](https://github.com/user-attachments/assets/ea81d353-4ed2-409b-a317-45a019ceeaef)

<div align="center"> <img src="NIMBUS AI" width="280">
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
<img width="2000" height="1415" alt="e- Certificate IJSREM Prof  Farzana Nadaf" src="https://github.com/user-attachments/assets/258a74cf-6011-4a67-bb5c-73f9aacbf32e" />

<p align="center"> <img src="certificate_farzana.png" width="70%"> <br><i>Prof. Farzana Nadaf – Published Certificate</i> </p> <p align="center"> <img src="certificate_samarth.png" width="70%"> <br><i>Sai Samarth Budihal – Published Certificate</i> </p> <p align="center"> <img src="certificate_sughnva.png" width="70%"> <br><i>Sughnva Chappar – Published Certificate</i> </p> <p align="center"> <img src="certificate_suprit.png" width="70%"> <br><i>Suprit Mundagod – Published Certificate</i> </p> <p align="center"> <img src="certificate_vishwanath.png" width="70%"> <br><i>Vishwanath Kotyal – Published Certificate</i> </p>
📚 Research Paper

📄 Full Published Paper (IJSREM 2025)
Enhancing the Economic Efficiency of Large Language Models (LLMs)
👉 Available in repository: /docs/Enhancing_the_Economic_Efficiency.pdf

🧠 System Architecture
![final architecture](https://github.com/user-attachments/assets/b22b72df-ff6d-4fab-85d7-4633945caa9f)

<p align="center"> <img src="architecture_final.jpg" width="75%"> </p>

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
<p align="center"> <img src="Frontend.png" width="85%"> </p>
🔹 Manual Model Selection
<p align="center"> <img src="Frontend_2.png" width="85%"> </p>
📊 Evaluation & Results

All results are derived from your published paper and dashboard screenshots.

🔹 Token Reduction vs Similarity
<p align="center"> <img src="token_reduction_vs_similarity.png" width="85%"> </p>
🔹 LLM Response Metrics
<p align="center"> <img src="llm_response_metrics.png" width="85%"> </p>
🔹 Performance by Prompt Type
<p align="center"> <img src="metrics_by_prompt_type.png" width="85%"> </p>
🔹 Analytics Dashboard
<p align="center"> <img src="Results_1.png" width="85%"> </p>
🔹 Cost Analysis
<p align="center"> <img src="Results_2.png" width="85%"> </p>
🔹 Detailed Prompt Analysis
<p align="center"> <img src="Results_3.png" width="85%"> </p>
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
