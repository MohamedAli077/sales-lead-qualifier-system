# Sales Lead Qualifier System

A machine learning-based system to classify and prioritize sales leads, predict conversion probability, and provide actionable insights.

---

## 🔍 Problem

Businesses generate thousands of leads, but most CRM systems do not prioritize them effectively.  
This leads to wasted time on low-quality leads and missed high-potential opportunities.

---

## 💡 Solution

This system:

- Classifies leads into High / Medium / Low priority  
- Predicts conversion probability  
- Compares 8 ML models and selects the best  
- Provides lead ranking and dashboards  
- Suggests follow-up actions and timelines  
- Allows CSV upload and export  

---

## ⚙️ Tech Stack

- Python  
- Scikit-learn (ML models)  
- Streamlit (UI)  
- Pandas, NumPy  
- Plotly (visualizations)  

---

## 📊 Features

- Multi-model comparison (8 algorithms)  
- Automatic best model selection  
- Out-of-fold prediction (realistic scoring)  
- Feature importance & model insights  
- Interactive dashboard  
- Real-time new lead prediction  
- Action plan recommendation system  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py