from flask import Flask, render_template, request, redirect, session, send_file
import sqlite3
import pickle
from datetime import datetime, timedelta
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)
app.secret_key = "sleep_ai_secret"


# LOAD ML MODEL
model = pickle.load(open("sleep_model.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))


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


# SCORE FUNCTION
def calculate_score(duration, stress, activity,
                    caffeine, screen,
                    spo2_before, spo2_after,
                    hr_before, hr_after):

    score = 50

    score += duration * 5
    score -= stress * 4
    score += activity * 0.2
    score -= caffeine * 3
    score -= screen * 2
    score += (spo2_after - 90) * 1.5
    score -= abs(hr_after - hr_before) * 0.2

    return max(0, min(100, int(score)))


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


# HOME
@app.route("/", methods=["GET","POST"])
def home():

    if "user" not in session:
        return redirect("/login")

    report = None
    error = None

    if request.method == "POST":

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


            prediction = model.predict([[

                duration,
                stress,
                activity,
                caffeine,
                screen,
                spo2_before,
                spo2_after,
                hr_before,
                hr_after

            ]])

            prediction = le.inverse_transform(prediction)[0]


            score = calculate_score(
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

    # Calculate streak (consecutive days with records)
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
        recent_records=recent_records
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