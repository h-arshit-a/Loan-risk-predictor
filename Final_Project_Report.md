# AI CreditPath: Complete End-to-End Project Report

## 1. Executive Summary
**AI CreditPath** is an end-to-end machine learning system designed to predict loan default probabilities and provide actionable business recommendations. It bridges the gap between raw financial data and user-friendly risk intelligence. The system features a robust data pipeline, an advanced gradient-boosted machine learning model, a high-performance FastAPI backend, and an interactive frontend dashboard.

This report details the entire journey of building the system from scratch, broken down into its core components and architectural decisions.

---

## 2. Data Engineering & Database (PostgreSQL)
The foundation of the project was built on a robust data ingestion and cleaning pipeline.

### The Raw Data & Database
* **Database Strategy**: The raw dataset (comprising hundreds of thousands of loan records) was initially stored in a **PostgreSQL** database to simulate a real-world enterprise environment.
* **Extraction**: We used Python (`psycopg2` and `SQLAlchemy`) to connect to the database and pull the raw data into Pandas DataFrames for processing.

### Data Cleaning & Preprocessing (Milestone 2)
To prepare the raw data for machine learning, we built a modular pipeline (`data_cleaning.py`, `encoder.py`, `scaler.py`):
1. **Structural Checks**: Handled null values and verified data types. Removed non-predictive features like `loanid`.
2. **Feature Engineering**: Created powerful derived features to capture borrower financial health, such as:
   * `debt_to_income_score` (DTI Ratio × Loan Amount)
   * `loan_per_month` (Loan Amount / Loan Term)
   * `credit_income_ratio` (Credit Score / Income)
3. **Encoding**: Categorical variables (e.g., employment type, education) were transformed into numerical format using One-Hot Encoding (`pd.get_dummies`).
4. **Scaling**: Numerical features were standardized using `StandardScaler` so that high-magnitude features (like Income) wouldn't overwhelm the model. The exact scaler state was saved to ensure future data is scaled identically.

---

## 3. Machine Learning Modeling (Milestones 3 & 4)
The core intelligence of CreditPath AI is its predictive model. We took an iterative approach to find the best algorithm.

### Baseline Model (Logistic Regression)
* We started with a basic Logistic Regression model to establish a performance baseline. While interpretable, it struggled to capture complex, non-linear relationships in the financial data.

### Advanced Models (XGBoost & LightGBM)
* To maximize predictive power, we upgraded to gradient boosting frameworks: **XGBoost** and **LightGBM**.
* **Why XGBoost?**: Gradient boosting builds multiple decision trees sequentially, with each tree correcting the errors of the previous ones. XGBoost is particularly excellent at handling tabular data, capturing non-linear interactions between features (like how age and income interact with loan term), and dealing with mild class imbalances.
* **Tuning & Evaluation**: We utilized hyperparameter tuning (GridSearch/RandomSearch) to optimize the trees. The models were evaluated using the **AUC-ROC** metric (Area Under the Receiver Operating Characteristic Curve), which is ideal for binary classification tasks like default prediction.
* **Serialization**: The winning XGBoost model and the fitted `StandardScaler` were exported as `.pkl` files using `joblib` so they could be loaded into a production environment without needing to retrain.

---

## 4. Backend API & Recommendation Engine (FastAPI)
To make the model accessible, we wrapped it in a high-performance REST API.

### Technology Stack
* **Framework**: **FastAPI** (Python). Chosen for its incredible speed, automatic documentation generation (Swagger UI), and native asynchronous support.
* **Validation**: Used **Pydantic** to strictly validate all incoming data from the frontend. It ensures that an API request cannot crash the model by passing invalid strings or out-of-bounds numbers.

### The Prediction Pipeline
When the API receives a request (`POST /predict`), it executes the exact same steps used in training:
1. Validates the Pydantic schema.
2. Applies the derived feature engineering logic.
3. Aligns the columns strictly to the `feature_names.pkl` layout.
4. Scales the data using the saved `scaler.pkl`.
5. Passes the data to the XGBoost model to generate a **Default Probability (0.0 to 1.0)**.

### Rule-Based Recommendation Engine
To make the prediction actionable for business users, the API passes the probability through a rule-based engine:
* **Low Risk (< 30%)**: Recommend standard automated reminders.
* **Medium Risk (30% - 59%)**: Trigger human-reviewed alert messages.
* **High Risk (≥ 60%)**: Recommend immediate debt recovery actions.

### API Database Integration
* We integrated **SQLAlchemy** directly into the FastAPI application to log every prediction request (including the input profile, risk category, and timestamp) into a PostgreSQL database table (`predictions`). This creates an audit trail for compliance and future model retraining.

### Deployment: Render
* The backend API was containerized and deployed to **Render** (`https://loan-risk-predictor.onrender.com`). Render provides 24/7 uptime, handles environment variables securely, and manages the Gunicorn/Uvicorn ASGI servers necessary to run FastAPI in production.

---

## 5. Frontend Dashboard (HTML/CSS/JS)
To provide a non-technical interface for the system, we built a beautiful, responsive, and dynamic web application.

### Design & Features
* **Vanilla Web Stack**: Built using standard HTML, vanilla CSS (with a modern glassmorphism design, dark mode toggles, and CSS variables), and vanilla JavaScript.
* **Interactive Form**: Users can manually input borrower details or use "Quick Fill" buttons to load demo personas (Low, Medium, High risk).
* **Real-time Feedback**: The frontend sends an async `fetch` request to the Render API. The UI displays loading states while waiting.
* **Visualizing Results**: The response is rendered instantly. A dynamic progress bar visualizes the probability, and a dedicated action card displays the recommended collection strategy.
* **Session History**: A real-time history sidebar tracks all predictions made during the session.

### Deployment: GitHub Pages
* The frontend was deployed using **GitHub Pages** (served directly from the `/docs` directory in the repository).
* **CORS & Integration**: The FastAPI backend was configured with CORS middleware to accept incoming API requests from the GitHub Pages domain. The frontend's `app.js` is permanently pointed to the live Render URL.

---

## 6. End-to-End System Flow
Here is the lifecycle of a single prediction in the finished product:
1. **User Interaction**: A loan officer inputs applicant data on the **GitHub Pages** dashboard and clicks "Analyse Risk".
2. **Network Request**: The browser sends a JSON payload to the **Render FastAPI** server.
3. **Validation & Logging**: FastAPI validates the data via Pydantic.
4. **Processing**: The input is transformed and scaled to match the training environment.
5. **Inference**: The **XGBoost** model predicts the default likelihood.
6. **Business Logic**: The recommendation engine assigns a risk tier and action.
7. **Storage**: The prediction record is saved to the **PostgreSQL** tracking database.
8. **Response & Render**: The API returns the JSON result to the frontend, which animates the results onto the dashboard instantly.

## Conclusion
The AI CreditPath project successfully evolved from a static Jupyter Notebook environment into a fully deployed, full-stack machine learning application. It demonstrates industry best practices in data engineering, predictive modeling, API design, and production deployment architectures.
