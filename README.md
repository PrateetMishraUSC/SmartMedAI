#SmartMedAI
##Fine-tuning a Medical QA LLM using Reinforcement Learning from AI Feedback
SmartMedAI is a zero-human-annotation alignment pipeline that transforms a 1.5B-parameter language model into a medical question-answering assistant. It combines Low-Rank Adaptation (LoRA) fine-tuning with Reinforcement Learning from AI Feedback (RLAIF) via Direct Preference Optimization (DPO) — achieving measurable gains in accuracy, semantic similarity, and text quality without any human labeling.
---

#Motivation
Off-the-shelf LLMs can produce unsafe or inaccurate medical advice without domain-specific tuning. While expert annotation is effective, it's expensive ($4–$6 per QA pair) and doesn't scale. SmartMedAI addresses this by replacing human feedback with AI-generated preference signals, enabling continuous, cost-effective alignment with medical best practices.
---

#Pipeline Overview
The framework operates in four sequential phases:
##Phase 1 — Corpus Preparation & Supervised Fine-Tuning

Dataset: 30,000 curated samples from the UltraMedical Literature split.
Preprocessing: Dialogues are formatted into system/user/assistant templates. Sequences exceeding the 95th-percentile token length are pruned, and prompt tokens are masked so the loss applies only to assistant responses.
Fine-Tuning: LoRA adapters (rank 16, α = 32) are applied to the q_proj and v_proj layers of the Qwen-2.5-1.5B-Instruct GPTQ model. Training runs for 3 epochs with AdamW (lr = 2e-5), FP16, and early stopping (patience 3).

##Phase 2 — Answer Generation & Automated Evaluation

For each question in the UltraMedical-Preference split, the Phase 1 model generates two candidate responses via nucleus sampling (top-p = 0.95, T = 0.7).
An evaluator LLM (Qwen-2.5-3B-Instruct) scores each response on a 0–1 scale against the reference answer.
3,750 valid preference triples (question, chosen answer, rejected answer) are retained.

##Phase 3 — DPO Training

The Phase 2 preference triples are combined with 52,500 records from the UltraMedical-Preference dataset (total ~56,250 examples, 90/10 train/val split).
The Phase 1 model is further trained using DPOTrainer with β = 0.4 for preference regularization, lr = 3e-5, bf16 precision, and early stopping after 4 non-improving evaluations.

##Phase 4 — Evaluation

200 held-out medical QA triples are scored by DeepSeek-R1-70B using a nine-tier alignment rubric and seven-item error checklist.
Additional automated metrics: BERTScore F1 (via DeBERTa-large-MNLI) and BLEURT.
---

#Results
MetricBase ModelDPO-Tuned ModelImprovementDeepSeek-R1 Score0.7690.805+3.6 pp (+4.7%)BERTScore F10.6210.651+3.0 pp (+4.8%)BLEURT0.5270.568+4.1 pp (+7.8%)
The DPO-tuned model shows consistent improvements across all metrics on the 200-question held-out evaluation set.
---

#Tech Stack

Base Model: Qwen-2.5-1.5B-Instruct GPTQ
Evaluator Model: Qwen-2.5-3B-Instruct
Scoring Model: DeepSeek-R1-Distill-Llama-70B (via Together.ai API)
Fine-Tuning: LoRA via PEFT (Hugging Face)
Alignment: DPOTrainer (TRL library)
Dataset: UltraMedical (400K+ biomedical instructions, 100K+ preference-annotated)
Metrics: BERTScore, Sentence-BERT cosine similarity, BLEURT
---

#Key Design Choices

LoRA over full fine-tuning — parameter-efficient adaptation keeps compute costs low while preserving the base model's general capabilities.
RLAIF over RLHF — eliminates the bottleneck of human annotation; AI-generated preferences provide a scalable reward signal.
DPO over PPO — frames preference learning as classification, avoiding the instability and large variance of vanilla policy-gradient methods.
Two-stage preference data — combining model-generated triples with UltraMedical's existing preference annotations provides richer training signal.
---

#Future Work

Scale to larger base and evaluator models for higher-quality generation and scoring.
Expand the supervised training set beyond 30K samples.
Increase the volume of preference-labeled examples for DPO.
Experiment with QLoRA and DoRA as alternatives to standard LoRA.
---
