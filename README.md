# Advanced Customer Segmentation Analytics Suite 🛍️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.x-brightgreen.svg)
![NLP](https://img.shields.io/badge/NLP-SentenceTransformers-orange.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blueviolet.svg)

---

## 🌐 Live Website

You can explore the deployed project here:
👉 [Customer Segmentation Analytics Suite (Live Demo)](https://customer-segmentation-analytics-main-final.streamlit.app/)

---

## 📖 Overview

This project is an **end-to-end machine learning pipeline** for analyzing customer transaction data, identifying distinct customer segments, and generating **actionable business strategies**.

It moves beyond simple analysis to "frame" a core business problem—how to stop wasting marketing money—as a data problem. It then solves it using advanced feature engineering, predictive analytics, and an interactive dashboard to deliver a complete "full-stack" consulting solution.

---

## 📊 Case Study: Driving E-Commerce Strategy with a Predictive Segmentation Pipeline

### 1. The Business Problem (The "Framing")

A (hypothetical) mid-sized e-commerce client faced declining marketing ROI and rising customer churn. Their "one-size-fits-all" marketing strategy was wasting money on loyal customers and failing to save at-risk ones.

The core business questions were:
* **Who are our most valuable customers (our "Whales")?**
* **Who is most likely to churn in the next 30 days?**
* **Are we wasting money by sending 20% discounts to loyal customers who would have bought anyway?**
* **How can we re-engage high-value customers who are about to leave?**

### 2. My Analytical Process (The "Structured Thought")

I designed a 4-step pipeline to move from raw data to a predictive tool.

1.  **Data Ingestion & Pipeline (Pandas & NLP)**
    * Aggregated and cleaned raw transaction history, product details, and customer reviews.
    * Used **Pandas** to build a robust ETL pipeline to handle missing values and merge disparate sources into a single master "customer" table.
    * Applied **NLP (Sentence Transformers)** to product descriptions to engineer features based on semantic meaning, not just keywords.

2.  **Advanced Feature Engineering**
    * To build a 360-degree view of each customer, I engineered features beyond simple sales:
        * **Recency, Frequency, Monetary (RFM) Scores:** The classic model for customer value.
        * **Predictive Features:** `Days_Since_Last_Purchase`, `Avg_Order_Value`, `Product_Diversity`.
        * **NLP-Driven Feature:** `Preferred_Product_Vector` (from Step 1).

3.  **Predictive Modeling & Segmentation (LightGBM, Scikit-learn)**
    * I trained two high-performance models:
        * **Model 1: Churn Prediction (Classification):** A `LightGBMClassifier` that predicts the *probability* (0.0 to 1.0) of a customer churning.
        * **Model 2: CLV Prediction (Regression):** A `LightGBMRegressor` that predicts the *total dollar value* a customer will spend.
    * I used an **Ensemble Clustering** approach (**K-Means, HDBSCAN, GMM**) on the RFM and behavioral features to group customers into stable, meaningful segments.

4.  **Segmentation & Visualization (Plotly & Streamlit)**
    * The true power came from combining these two model outputs. I grouped customers based on their CLV and Churn scores to create four distinct, actionable segments.
    * I built an interactive **Plotly dashboard** (including 3D cluster plots) and deployed the entire pipeline as a **Streamlit web app**, giving the marketing team a simple tool to get on-the-fly analysis.

### 3. Actionable Recommendations (The "Deck")

The analysis revealed four key customer personas. Based on the data, I delivered a clear, data-driven strategy:

| Persona | Recommendation | Business Impact |
| :--- | :--- | :--- |
| **Loyal Champions** | **Action:** **Stop sending discounts.** Move them to a new "VIP Loyalty Program" focused on early access and exclusive community perks. | **Impact:** Immediately increases profit margins. Builds a long-term "brand moat" that competitors can't beat with price. |
| **At-Risk Whales** | **Action:** **Deploy an aggressive, automated "Win-Back" campaign.** Proactively send a personal "we miss you" email with a 25% retention offer. | **Impact:** Directly targets and saves the most valuable, at-risk customers, providing the highest possible ROI on marketing spend. |
| **Price-Sensitive Shoppers** | **Action:** **Minimize marketing spend.** Remove them from primary campaigns and only send them low-cost, automated notifications for clearance sales. | **Impact:** Frees up budget by not over-investing in low-value, low-loyalty customers. |
| **New & Promising** | **Action:** **Implement a 30-day "Welcome" email series.** Focus on product education, brand story, and driving a second purchase. | **Impact:** Nurtures new users to convert them into high-value "Loyal Champions," increasing the long-term health of the customer base. |

---

## 🚀 Key Technical Features

* **🔎 Data Profiling:** Automated data quality and characteristics analysis.
* **⚙️ Advanced Feature Engineering:** RFM metrics (Recency, Frequency, Monetary), product diversity, and purchasing patterns.
* **🧠 NLP Analysis:** Uses **Sentence Transformers** to extract semantic meaning from product descriptions.
* **📊 Ensemble Clustering:** Combines **K-Means, HDBSCAN, and GMM** for stable, meaningful customer segmentation.
* **📈 Predictive Analytics:** Models customer **churn** and **Customer Lifetime Value (CLV)**.
* **🎨 Interactive Visualizations:** Executive dashboards and **3D cluster plots** for exploration.

---

## 📈 Results and Visualizations

**Customer Segmentation 3D Visualization**
![Segmentation Plot]()

**Executive Dashboard**
![Dashboard](https://github.com/rishabh9843/Customer-Segmentation-Analytics-main-final/blob/main/image.png)

**Churn & CLV Predictions**
![Churn CLV](images/churn_clv_predictions.png)

---

## ⚙️ How to Run

### 1️⃣ Clone the repository

```bash
git clone [https://github.com/YOUR_USERNAME/customer-segmentation-pipeline.git](https://github.com/YOUR_USERNAME/customer-segmentation-pipeline.git)
cd customer-segmentation-pipeline


