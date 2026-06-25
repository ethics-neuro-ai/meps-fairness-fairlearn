# 🩺 Algorithmic Fairness & Diagnostic Delay: Auditing Structural Violence via IPUMS MEPS

Welcome to the **IPUMS MEPS Fairness Audit** project! This repository contains a comprehensive socio-technical case study demonstrating how systemic socioeconomic inequalities encode themselves into machine learning models, and how to actively mitigate these disparities using the [Fairlearn](https://fairlearn.org) framework.

---

## 📊 Project Vision & Socio-Technical Context

In public health management systems, diagnostic and therapeutic delays are not merely operational bottlenecks. When individuals are forced to postpone necessary medical diagnoses due to financial distress, it constitutes a form of **systemic economic violence against low-income populations**. 

Algorithms trained on historical healthcare data risk capturing and reinforcing these deep socioeconomic boundaries. This project targets **allocative harms**: if a predictive system misclassifies an impoverished individual as "not at risk of delaying care" (simply because they interact less with healthcare infrastructure due to cost), it denies them proactive outreach and targeted financial subsidies. 

---

## 🧬 Dataset & Data Schema

The workflow uses an anonymous, curated subset of the **IPUMS Medical Expenditure Panel Survey (MEPS)**. To preserve full scientific validity, all data operations strictly incorporate national survey sampling weights (`PERWEIGHT`).

*   **Target Attribute (`TARGET_DELAY`)**: Derived from the original `DELAYCOST` variable. Maps whether a patient was forced to delay care in the past 12 months due to worry about costs (`1 = Yes`, `0 = No`).
*   **Protected Attribute (`LOW_INCOME`)**: Identifies patients living below the Federal Poverty Threshold (`1 = Below Poverty Line`, `0 = Others`).
*   **Features Matrix**: General demographic indicators including `AGE`, `SEX`, and Total Family Income (`FTOTVAL`).

---

## 📈 Key Empirical Insights

Our Exploratory Data Analysis (EDA) mapped via a **Nightingale Rose Chart** surfaces an alarming baseline social cleavage:

*   🔴 **Low-Income Patients (Young & Older)**: Experience an aggressive **~29% baseline healthcare delay rate**. Nearly 1 out of 3 individuals must compromise their health due to costs.
*   🟢 **Mid/High-Income Patients**: Suffer a baseline delay rate of only **3% to 4%**.

---

## 🧬 Architectural Pipeline (The 5 Experiments)

This project avoids "abstract fairness fixes" by putting the data scientist through a real-world multi-model audit pipeline:

1.  **Experiment 1: The Unaware Classifier** 🧼  
    *   *Result*: Deceptively high global accuracy (~95%) and close-to-zero disparity. 
    *   *Insight*: A textbook case of **Fairwashing**. The model achieves high accuracy on an imbalanced dataset by defaulting to negative predictions for everyone, rendering the poor statistically invisible.
2.  **Experiment 2: Exposing Disparities via Class Balancing** 🚨  
    *   *Result*: High False Negative Rate (FNR) disparities emerge (~98% disparity margin).
    *   *Insight*: Forcing the model to recognize the minority class uncovers the underlying structural polarization.
3.  **Experiment 3: Post-Processing Limitations** ⏳  
    *   *Result*: Prediction utility collapses entirely (`selection_rate` drops to zero).
    *   *Insight*: Post-processing tools (`ThresholdOptimizer`) are too rigid for highly disparate healthcare features, shutting down model operations completely to satisfy equity.
4.  **Experiment 4: In-Processing GridSearch (Demographic Parity)** ⚙️  
    *   *Result*: Disparity drops, but FNR remains high across both cohorts.
5.  **Experiment 5: In-Processing GridSearch (Equalized Odds)** 🎯  
    *   *Result*: **The Optimal Deployment Framework**. Disparity drops to ~11% while shrinking the low-income FNR to **31.1%**. The algorithm successfully identifies nearly **69%** of marginalized patients requiring care.

---

## 🎨 Visualized Analytics Included

The compiled notebook generates advanced, publication-grade analytical charts:
*   🌹 **Nightingale Rose Chart (Flower Plot)**: Mapping pre-modeling systemic poverty gaps.
*   📊 **Comparative Bar Charts**: Tracking the False Negative Rate (FNR) collapse across optimization checkpoints.
*   📉 **Weighted Confusion Matrices**: Viewing the exact distribution of False Positives and Negatives across income groups.
*   🌟 **Fairness-Accuracy Pareto Frontier**: Mapping the multi-objective optimization candidate space.

---

## 🚀 Installation & Quickstart

To clone this repository and explore the interactive Jupyter Notebook locally, execute the following commands in your terminal:

```bash
# Clone the repository
git clone https://github.com
cd meps-fairness-fairlearn

# Install requirements
pip install fairlearn scikit-learn pandas numpy matplotlib requests jupytext

# (Optional) Regenerate the Jupyter notebook with local outputs
jupytext --to notebook plot_meps_healthcare_delay.py
jupyter nbconvert --to notebook --execute plot_meps_healthcare_delay.ipynb --inplace
```

---

## ⚖️ Ethical License Notice
Data redistributed in this repository is heavily sampled and anonymized in strict compliance with the IPUMS Microdata Dissemination Policy. For academic or clinical research pipelines, please fetch the original full-scale survey distributions directly from the [IPUMS MEPS portal](https://ipums.org).
