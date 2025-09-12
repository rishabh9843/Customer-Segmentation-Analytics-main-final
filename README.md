# Advanced Customer Segmentation Analytics Suite

This project is an end-to-end machine learning pipeline for analyzing customer transaction data, identifying distinct customer segments, and generating actionable business strategies.



## 🚀 Features

- **Data Profiling:** Automatically analyzes the input data for quality and characteristics.
- **Advanced Feature Engineering:** Creates sophisticated features related to customer behavior (Recency, Frequency, Monetary), product diversity, and purchasing patterns.
- **NLP Analysis:** Uses Sentence Transformers to understand the semantics of product descriptions.
- **Ensemble Clustering:** Combines multiple clustering algorithms (K-Means, HDBSCAN, GMM) to find the most stable and meaningful customer segments.
- **Predictive Analytics:** Builds models to predict customer churn and future Customer Lifetime Value (CLV).
- **Interactive Visualizations:** Generates an executive dashboard and 3D visualizations to explore the customer segments.

## 🛠️ Technologies Used

- **Python:** The core programming language.
- **Pandas & NumPy:** For data manipulation and numerical operations.
- **Scikit-learn:** For machine learning models and preprocessing.
- **Sentence Transformers:** For Natural Language Processing.
- **Plotly:** For creating interactive dashboards and visualizations.
- **UMAP & HDBSCAN:** For advanced dimensionality reduction and clustering.

## ⚙️ How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/customer-segmentation-pipeline.git](https://github.com/YOUR_USERNAME/customer-segmentation-pipeline.git)
    cd customer-segmentation-pipeline
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place your data:** Add your transaction data file (e.g., `int_online_tx.csv`) into the `/data` directory.
4.  **Run the analysis:** Open and run the `customer_segmentation.ipynb` notebook in a Jupyter or Google Colab environment.

## 📊 Sample Output

*(Here you can add a screenshot of the final dashboard or the 3D visualization)*