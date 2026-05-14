from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import pickle
from datetime import datetime, timedelta
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from xgboost import XGBClassifier

app = Flask(__name__)
app.secret_key = "sleep_ai_secret"

# Load ML model
model = XGBClassifier()
model.load_model("xgboost_model.json")

with open("feature_info.pkl", "rb") as f:
    feature_info = pickle.load(f)

features = feature_info['features']
quality_mapping = feature_info['mapping']  # {'poor': 0, 'good': 1, 'best': 2}
reverse_mapping = {v: k for k, v in quality_mapping.items()}


# DATABASE INIT
def init_db():

    conn = sqlite3.connect("sleep.db")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS sleep_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        duration REAL,
        stress REAL,
        activity REAL,
        caffeine REAL,
        screen REAL,
        spo2_before REAL,
        spo2_after REAL,
        hr_before REAL,
        hr_after REAL,
        quality TEXT,
        score INTEGER,
        date TEXT
    )
    """)

    conn.close()


init_db()


# ML PREDICTION FUNCTION
def calculate_score(duration, stress, activity,
                    caffeine, screen,
                    spo2_before, spo2_after,
                    hr_before, hr_after):

    # Calculate HR change (feature used by model)
    hr_change = hr_after - hr_before

    # Prepare input for ML model
    # Features: ['Sleep Duration', 'Stress Level', 'Physical Activity',
    #            'Caffeine Cups', 'Screen Time', 'HR_Change']
    input_data = np.array([[duration, stress, activity, caffeine, screen, hr_change]])

    # Predict quality class: 0=poor, 1=good, 2=best
    prediction = model.predict(input_data)[0]

    # Convert to score (0-33 = poor, 34-66 = good, 67-100 = best)
    if prediction == 0:  # poor
        score = np.random.randint(20, 34)
    elif prediction == 1:  # good
        score = np.random.randint(45, 66)
    else:  # best
        score = np.random.randint(70, 95)

    # Also factor in SpO2 for a more personalized score
    if spo2_after >= 95:
        score = min(100, score + 5)
    elif spo2_after < 90:
        score = max(0, score - 10)

    return int(score), reverse_mapping[prediction]


# REGISTER
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("sleep.db")

        try:

            conn.execute(
                "INSERT INTO users (username,password) VALUES (?,?)",
                (username,password)
            )

            conn.commit()
            conn.close()

            return redirect("/login")

        except:

            return render_template(
                "register.html",
                error="User already exists"
            )

    return render_template("register.html")


# LOGIN
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("sleep.db")

        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username,password)
        ).fetchone()

        conn.close()

        if user:

            session["user"] = username
            return redirect("/")

        else:

            return render_template(
                "login.html",
                error="Invalid login"
            )

    return render_template("login.html")


# LOGOUT
@app.route("/logout")
def logout():

    session.clear()
    return redirect("/login")


# DARK MODE TOGGLE
@app.route("/toggle_dark_mode")
def toggle_dark_mode():
    if "user" not in session:
        return redirect("/login")

    session["dark_mode"] = not session.get("dark_mode", False)
    return redirect(request.referrer or "/")


# SLEEP HISTORY
@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    username = session["user"]

    quality_filter = request.args.get("quality", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    query = "SELECT * FROM sleep_records WHERE username=?"
    params = [username]

    if quality_filter:
        query += " AND quality=?"
        params.append(quality_filter)

    if date_from:
        query += " AND date>=?"
        params.append(date_from)

    if date_to:
        query += " AND date<=?"
        params.append(date_to)

    query += " ORDER BY date DESC"

    conn = sqlite3.connect("sleep.db")
    records = conn.execute(query, params).fetchall()
    conn.close()

    return render_template(
        "history.html",
        user=username,
        records=records,
        quality_filter=quality_filter,
        date_from=date_from,
        date_to=date_to
    )


# SLEEP TIPS
@app.route("/tips")
def tips():

    if "user" not in session:
        return redirect("/login")

    username = session["user"]

    conn = sqlite3.connect("sleep.db")

    recent = conn.execute("""
        SELECT * FROM sleep_records
        WHERE username=?
        ORDER BY date DESC
        LIMIT 30
    """, (username,)).fetchall()

    stats = conn.execute("""
        SELECT
            AVG(duration) as avg_duration,
            AVG(stress) as avg_stress,
            AVG(activity) as avg_activity,
            AVG(caffeine) as avg_caffeine,
            AVG(screen) as avg_screen,
            AVG(spo2_after) as avg_spo2,
            AVG(hr_after) as avg_hr,
            AVG(score) as avg_score
        FROM sleep_records WHERE username=?
    """, (username,)).fetchone()

    conn.close()

    tips = []

    if not recent:
        tips.append({"icon": "fa-bed", "title": "Start Tracking", "text": "Log your first sleep record to get personalized tips!", "type": "info"})
    else:
        if stats[0] < 7:
            tips.append({"icon": "fa-clock", "title": "Sleep More", "text": f"You're averaging {stats[0]:.1f}h sleep. Aim for 7-9 hours for optimal health.", "type": "warning"})
        elif stats[0] >= 7 and stats[0] <= 9:
            tips.append({"icon": "fa-check-circle", "title": "Great Sleep Duration", "text": "Your sleep duration is in the healthy range!", "type": "success"})
        else:
            tips.append({"icon": "fa-moon", "title": "Avoid Oversleeping", "text": f"You're averaging {stats[0]:.1f}h. Too much sleep can also affect quality.", "type": "info"})

        if stats[1] > 5:
            tips.append({"icon": "fa-brain", "title": "Reduce Stress", "text": f"Your stress level is {stats[1]:.1f}/10. Try meditation or deep breathing exercises.", "type": "warning"})
        else:
            tips.append({"icon": "fa-smile", "title": "Manage Stress Well", "text": "Your stress levels are well controlled. Keep it up!", "type": "success"})

        if stats[2] < 30:
            tips.append({"icon": "fa-running", "title": "More Activity", "text": f"You're averaging {stats[2]:.0f}min activity. Try to get at least 30 minutes daily.", "type": "warning"})
        else:
            tips.append({"icon": "fa-heartbeat", "title": "Active Lifestyle", "text": f"Great! {stats[2]:.0f}min daily activity helps sleep quality.", "type": "success"})

        if stats[3] > 2:
            tips.append({"icon": "fa-coffee", "title": "Limit Caffeine", "text": f"You're having {stats[3]:.1f} cups daily. Try to reduce, especially after 2pm.", "type": "warning"})
        else:
            tips.append({"icon": "fa-check", "title": "Healthy Caffeine", "text": "Your caffeine intake is at a healthy level.", "type": "success"})

        if stats[4] > 2:
            tips.append({"icon": "fa-mobile-alt", "title": "Reduce Screen Time", "text": f"You're on screens {stats[4]:.1f}h before bed. Try to stop 1-2 hours before sleep.", "type": "warning"})
        else:
            tips.append({"icon": "fa-book", "title": "Good Screen Habits", "text": "You're limiting screen time before bed. Great job!", "type": "success"})

        if stats[5] < 95:
            tips.append({"icon": "fa-user-md", "title": "Check SpO2", "text": f"Your avg SpO2 is {stats[5]:.1f}%. Consult a doctor if this persists.", "type": "warning"})
        else:
            tips.append({"icon": "fa-lungs", "title": "Healthy SpO2", "text": f"Your SpO2 levels are great at {stats[5]:.1f}%!", "type": "success"})

        if stats[7]:
            if stats[7] >= 75:
                tips.append({"icon": "fa-trophy", "title": "Excellent Sleep!", "text": f"Your average score is {stats[7]:.0f}/100. Keep up the great work!", "type": "success"})
            elif stats[7] >= 50:
                tips.append({"icon": "fa-chart-line", "title": "Room for Improvement", "text": f"Your score is {stats[7]:.0f}/100. Try following the tips above to improve!", "type": "info"})
            else:
                tips.append({"icon": "fa-exclamation-triangle", "title": "Needs Attention", "text": f"Your score is {stats[7]:.0f}/100. Consider consulting a sleep specialist.", "type": "warning"})

    return render_template("tips.html", user=username, tips=tips, stats=stats, dark_mode=session.get("dark_mode", False))


# HOME
@app.route("/", methods=["GET","POST"])
def home():

    if "user" not in session:
        return redirect("/login")

    report = None
    error = None

    if request.method == "POST":

        try:

            # Input validation
            try:
                duration = float(request.form["duration"])
                stress = float(request.form["stress"])
                activity = float(request.form["activity"])
                caffeine = float(request.form["caffeine"])
                screen = float(request.form["screen"])
                spo2_before = float(request.form["spo2_before"])
                spo2_after = float(request.form["spo2_after"])
                hr_before = float(request.form["hr_before"])
                hr_after = float(request.form["hr_after"])
            except ValueError:
                error = "Please enter valid numbers for all fields"
                raise ValueError(error)

            # Range validation (realistic limits)
            if not (0 < duration <= 14):
                error = "Duration must be between 0 and 14 hours (realistic maximum)"
                raise ValueError(error)
            if not (0 <= stress <= 10):
                error = "Stress level must be between 0 and 10"
                raise ValueError(error)
            if not (0 <= activity <= 180):
                error = "Activity must be between 0 and 180 minutes (max 3 hours)"
                raise ValueError(error)
            if not (0 <= caffeine <= 10):
                error = "Caffeine must be between 0 and 10 cups"
                raise ValueError(error)
            if not (0 <= screen <= 10):
                error = "Screen time must be between 0 and 10 hours"
                raise ValueError(error)
            if not (85 <= spo2_before <= 100):
                error = "SpO2 before must be between 85 and 100%"
                raise ValueError(error)
            if not (85 <= spo2_after <= 100):
                error = "SpO2 after must be between 85 and 100%"
                raise ValueError(error)
            if not (50 <= hr_before <= 150):
                error = "Heart rate before must be between 50 and 150 bpm (normal range)"
                raise ValueError(error)
            if not (50 <= hr_after <= 150):
                error = "Heart rate after must be between 50 and 150 bpm (normal range)"
                raise ValueError(error)
            sleep_date = request.form.get("sleep_date", datetime.now().strftime("%Y-%m-%d"))

            # Get score and prediction from ML model
            score, prediction = calculate_score(
                duration,stress,activity,
                caffeine,screen,
                spo2_before,spo2_after,
                hr_before,hr_after
            )


            conn = sqlite3.connect("sleep.db")

            conn.execute("""

            INSERT INTO sleep_records

            (username,
             duration,stress,activity,caffeine,screen,
             spo2_before,spo2_after,
             hr_before,hr_after,
             quality,score,date)

            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)

            """,(

                session["user"],

                duration,
                stress,
                activity,
                caffeine,
                screen,

                spo2_before,
                spo2_after,

                hr_before,
                hr_after,

                prediction,
                score,
                str(datetime.now())

            ))

            conn.commit()
            conn.close()


            # Generate dynamic recommendations
            recommendations = []

            if duration < 7:
                recommendations.append(f"Try to sleep at least 7 hours (current: {duration}h)")
            else:
                recommendations.append("Great job maintaining good sleep duration!")

            if stress > 5:
                recommendations.append("Consider stress management techniques like meditation")
            else:
                recommendations.append("Your stress levels are well managed")

            if activity < 30:
                recommendations.append("Try to get at least 30 minutes of physical activity daily")
            else:
                recommendations.append("Good physical activity level - keep it up!")

            if caffeine > 2:
                recommendations.append(f"Limit caffeine intake (current: {caffeine} cups)")
            else:
                recommendations.append("Caffeine consumption is at a healthy level")

            if screen > 2:
                recommendations.append(f"Reduce screen time before bed (current: {screen}h)")
            else:
                recommendations.append("Screen time before bed is reasonable")

            if spo2_after < 95:
                recommendations.append("Consider consulting a doctor for low SpO2 levels")

            report = {
                "quality": prediction,
                "score": score,
                "recommendations": recommendations
            }


        except Exception as e:
            error = str(e)


    conn = sqlite3.connect("sleep.db")

    data = conn.execute(
        "SELECT date,score FROM sleep_records WHERE username=?",
        (session["user"],)
    ).fetchall()

    conn.close()


    dates = [i[0][:10] for i in data]
    scores = [i[1] for i in data]

    # Calculate stats
    records_count = len(scores)
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    best_score = max(scores) if scores else 0

    # Calculate streak
    current_streak = 0
    if dates:
        dates_sorted = sorted(set(dates), reverse=True)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        if dates_sorted[0] == today or dates_sorted[0] == yesterday:
            current_streak = 1
            for i in range(1, len(dates_sorted)):
                prev = datetime.strptime(dates_sorted[i-1], "%Y-%m-%d")
                curr = datetime.strptime(dates_sorted[i], "%Y-%m-%d")
                if (prev - curr).days == 1:
                    current_streak += 1
                else:
                    break

    # Get recent records
    conn = sqlite3.connect("sleep.db")
    recent = conn.execute(
        "SELECT date, score FROM sleep_records WHERE username=? ORDER BY date DESC LIMIT 10",
        (session["user"],)
    ).fetchall()
    conn.close()

    recent_records = [{"date": r[0][:10], "score": r[1]} for r in recent]


    return render_template(
        "index.html",
        report=report,
        error=error,
        dates=dates,
        scores=scores,
        user=session["user"],
        records_count=records_count,
        avg_score=avg_score,
        current_streak=current_streak,
        best_score=best_score,
        recent_records=recent_records,
        dark_mode=session.get("dark_mode", False)
    )


# DASHBOARD
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    username = session["user"]

    conn = sqlite3.connect("sleep.db")

    all_records = conn.execute(
        "SELECT * FROM sleep_records WHERE username=? ORDER BY date DESC",
        (username,)
    ).fetchall()

    stats = conn.execute("""
        SELECT
            COUNT(*) as total_records,
            AVG(score) as avg_score,
            MAX(score) as max_score,
            MIN(score) as min_score
        FROM sleep_records WHERE username=?
    """, (username,)).fetchone()

    quality_dist = conn.execute("""
        SELECT quality, COUNT(*) as count
        FROM sleep_records WHERE username=?
        GROUP BY quality
    """, (username,)).fetchall()

    recent = conn.execute(
        "SELECT * FROM sleep_records WHERE username=? ORDER BY date DESC LIMIT 7",
        (username,)
    ).fetchall()

    trend_data = conn.execute(
        "SELECT date, score FROM sleep_records WHERE username=? ORDER BY date ASC",
        (username,)
    ).fetchall()

    conn.close()

    dates = [d[0][:10] for d in trend_data]
    scores = [d[1] for d in trend_data]

    return render_template(
        "dashboard.html",
        user=username,
        records=all_records,
        stats=stats,
        quality_dist=quality_dist,
        recent=recent,
        dates=dates,
        scores=scores,
        dark_mode=session.get("dark_mode", False)
    )


# PDF REPORT GENERATION
@app.route("/download_report")
def download_report():

    if "user" not in session:
        return redirect("/login")

    username = session["user"]

    conn = sqlite3.connect("sleep.db")

    data = conn.execute(
        "SELECT date,score FROM sleep_records WHERE username=?",
        (username,)
    ).fetchall()

    conn.close()

    if not data:
        return "No records"


    dates = [d[0][:10] for d in data]
    scores = [d[1] for d in data]


    # CREATE GRAPH IMAGE
    plt.figure()

    plt.plot(dates, scores, marker="o")

    plt.title("Sleep Score Trend")

    plt.xlabel("Date")
    plt.ylabel("Score")

    plt.xticks(rotation=45)

    plt.tight_layout()


    if not os.path.exists("static"):
        os.makedirs("static")

    graph_path = "static/sleep_graph.png"

    plt.savefig(graph_path)

    plt.close()


    # TREND ANALYSIS
    analysis = ""

    for i in range(1, len(scores)):

        diff = scores[i] - scores[i-1]

        if diff > 0:

            analysis += f"Sleep improved by {diff} points on {dates[i]}.<br/>"

        elif diff < 0:

            analysis += f"Sleep decreased by {abs(diff)} points on {dates[i]}.<br/>"

        else:

            analysis += f"No change on {dates[i]}.<br/>"


    avg = sum(scores) / len(scores)

    analysis += f"<br/><b>Average Score:</b> {avg:.1f}"


    # CREATE PDF
    pdf_path = "static/sleep_report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            f"<b>Sleep Analysis Report: {username}</b>",
            styles["Heading2"]
        )
    )

    elements.append(Spacer(1,20))

    elements.append(
        Paragraph(analysis, styles["BodyText"])
    )

    elements.append(Spacer(1,20))

    elements.append(
        Image(graph_path, width=400, height=300)
    )

    doc.build(elements)


    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":

    app.run(debug=True)