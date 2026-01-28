# 📘 Call Driver Generator

This project uses **Azure OpenAI (GPT-4o)** to analyze customer communications related to an **insurance company acquisition** and automatically generate:

- 📞 Top **customer call driver categories**
- ❓ Realistic **customer questions** per category
- 🎧 Professional **agent response scripts**

It is designed for **call center training**, **FAQ generation**, and **agent-assist knowledge base creation**.

---

## 🧠 Problem This Solves

When **Insurance Company B acquires Company A**, customers receive:

- Emails  
- Announcements  
- Policy servicing notices  
- Training & support materials  

These changes often lead to **confusion and high call volumes**.

This system reads *all* those materials and predicts:

> “What are customers most likely to call about, and how should agents respond?”

---

## 🏗️ Architecture Overview

This pipeline uses a **Map → Reduce → Answer** LLM workflow.

