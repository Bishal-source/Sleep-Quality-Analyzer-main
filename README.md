# Sleep Quality Analyzer 🌙

A machine learning-based web application that analyzes sleep quality using physiological and lifestyle factors. The system predicts sleep quality using an XGBoost model and provides personalized recommendations, visual analytics, and downloadable reports.

---

# Features

- Sleep quality prediction using XGBoost
- Personalized sleep score generation
- Interactive dashboard
- Sleep score trend visualization
- PDF report generation
- User authentication system
- Dark mode support
- Input validation and error handling
- Personalized recommendations
- Historical sleep record tracking

---

# Technologies Used

## Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- Chart.js

## Backend
- Python
- Flask

## Machine Learning
- XGBoost
- Scikit-learn
- NumPy
- Pandas

## Database
- SQLite

## Report Generation
- ReportLab
- Matplotlib

---

# Machine Learning Features

The model analyzes:

- Sleep duration
- Stress level
- Physical activity
- Caffeine intake
- Screen time
- Heart rate changes
- SpO₂ levels

The application uses an XGBoost classifier with probability-based scoring to generate stable and consistent sleep quality predictions.

---

# Setup Instructions

## Clone the Repository

```bash
git clone https://github.com/Bishal-source/Sleep-Quality-Analyzer-main.git
```

---

## GitHub Repository

https://github.com/Bishal-source/Sleep-Quality-Analyzer-main

# Create Virtual Environment

```bash
python -m venv .venv
```

---

# Activate Virtual Environment

## Windows

```bash
.venv\Scripts\activate
```

## Linux / Mac

```bash
source .venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
python app.py
```

---

# Open in Browser

```text
http://127.0.0.1:5000
```

---

# Project Structure

```text
Sleep-Quality-Analyzer-main/
│
├── app.py
├── train_new_xgboost.py
├── feature_info.pkl
├── xgboost_model.json
├── requirements.txt
├── README.md
├── .gitignore
├── sleep.db
│
├── dataset/
│
├── static/
│   ├── style.css
│   ├── sleep_graph.png
│   └── sleep_report.pdf
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── history.html
│   ├── login.html
│   ├── register.html
│   └── tips.html
│
└── .venv/
```

---

# Security Improvements

- Password hashing using Werkzeug
- Inline form validation
- Improved authentication handling
- Environment-based secret key support

---

# Future Improvements

- Better ML explainability
- Advanced data visualization
- Improved analytics dashboard
- Better responsive optimization
- Enhanced recommendation system

---


---

# Author

Bishal Sunar

---

# License

This project is developed for educational and academic purposes.