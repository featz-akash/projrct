

import os, json, pickle
from datetime import date, timedelta
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from dotenv import load_dotenv
import calendar
from datetime import date, timedelta


def calculate_weekly_streak(habits, logs_df):
    today = date.today()
    streak_data = []
    safe_dates = pd.to_datetime(logs_df["date"], errors="coerce").dt.date

    for i in range(7):
        d = today - timedelta(days=6 - i)
        daily_scores = []

        for h in habits:
            mask = (logs_df["habit"] == h["name"]) & (safe_dates == d)
            done = float(logs_df.loc[mask, "value"].sum()) if mask.sum() else 0
            goal = max(float(h["goal"]), 1)
            pct = min(done / goal, 1.0)
            daily_scores.append(pct)

        avg_progress = np.mean(daily_scores) if daily_scores else 0
        streak_data.append(avg_progress >= 0.6)

    return streak_data



def calculate_monthly_streak(habits, logs_df, month=None, year=None):
    if month is None: month = date.today().month
    if year is None: year = date.today().year

    safe_dates = pd.to_datetime(logs_df["date"], errors="coerce").dt.date
    days_in_month = calendar.monthrange(year, month)[1]

    daily_values = []
    for day in range(1, days_in_month + 1):
        d = date(year, month, day)
        daily_scores = []

        for h in habits:
            mask = (logs_df["habit"] == h["name"]) & (safe_dates == d)
            done = float(logs_df.loc[mask, "value"].sum()) if mask.sum() else 0
            goal = max(float(h["goal"]), 1)
            pct = min(done / goal, 1.0)
            daily_scores.append(pct)

        avg_progress = np.mean(daily_scores) if daily_scores else 0
        daily_values.append(avg_progress)

    return daily_values



load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")



st.set_page_config(page_title="AI Personal Coach", page_icon="🧠", layout="wide")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
    background: #f4f7fb;
}

[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 2px solid #e5e9f2;
    padding-top:20px;
}

.sidebar-btn {
    display: block;
    padding: 12px 18px;
    margin: 6px 0;
    font-weight:600;
    text-decoration: none;
    border-radius: 10px;
    transition: 0.2s;
    border: 1px solid transparent;
    cursor:pointer;
}
.sidebar-btn:hover {
    background:#eef3ff;
    border-color:#4b7bec;
}
.sidebar-btn.active {
    background: linear-gradient(90deg,#4b7bec,#3867d6);
    color:white !important;
    border: none !important;
}

.stButton>button {
    background: linear-gradient(90deg,#4b7bec,#3867d6);
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    font-weight:600;
    border:none;
    transition:0.2s;
}
.stButton>button:hover { transform: scale(1.04); }

.secondary-btn>button {
    background: white !important;
    border: 2px solid #4b7bec !important;
    color: #4b7bec !important;
    padding: 8px 16px;
    border-radius: 10px;
    font-weight:600;
}
.secondary-btn>button:hover {
    background:#f0f4ff !important;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 18px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
AI_METRICS_CSV = os.path.join(DATA_DIR, "ai_metrics.csv")
HABITS_JSON = os.path.join(DATA_DIR, "habits.json")
HABIT_LOGS_CSV = os.path.join(DATA_DIR, "habit_logs.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")


def seed_ai_metrics(days=28):
    rng = np.random.default_rng(42)
    dates = [date.today() - timedelta(days=i) for i in range(days)][::-1]
    df = pd.DataFrame({
        "date": dates,
        "sleep_hours": rng.normal(7,1,days).clip(4,10),
        "study_hours": rng.normal(3,1.2,days).clip(0,10),
        "screen_time_hours": rng.normal(4,1.5,days).clip(0,12),
        "exercise_minutes": rng.normal(30,20,days).clip(0,180),
        "water_glasses": rng.integers(4,11,days),
        "mood_1_10": rng.integers(4,10,days),
    })
    df["productivity_1_10"] = (
        0.8*(df.sleep_hours-6.5)
        +1.2*(df.study_hours-2)
        -0.5*(df.screen_time_hours-4)
        +0.02*(df.exercise_minutes-30)
        +0.2*(df.water_glasses-6)
        +0.5*(df.mood_1_10-6)
        +6.0
    ).clip(1,10).round(1)
    return df

if not os.path.exists(AI_METRICS_CSV):
    seed_ai_metrics().to_csv(AI_METRICS_CSV, index=False)

ai_df = pd.read_csv(AI_METRICS_CSV, parse_dates=["date"])

if "clicked_page" not in st.session_state:
    st.session_state.clicked_page = "Home"

with st.sidebar:
    st.markdown("### 🧠 AI Personal Coach")
    st.write("---")
    pages = ["Home", "Habits", "Analytics", "AI Coach", "Admin Panel", "Settings"]
    for p in pages:
        active = "active" if st.session_state.clicked_page == p else ""
        if st.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.clicked_page = p
    st.write("---")
    st.caption("Made by Akash S")

page = st.session_state.clicked_page


def load_habits():
    if not os.path.exists(HABITS_JSON):
        save_habits([
            {"name": "Drink Water", "unit": "Glasses", "goal": 8},
            {"name": "Exercise", "unit": "Minutes", "goal": 45},
            {"name": "Read", "unit": "Minutes", "goal": 30},
        ])
    with open(HABITS_JSON, "r") as f:
        return json.load(f)

def save_habits(habits):
    with open(HABITS_JSON, "w") as f:
        json.dump(habits, f, indent=2)

def load_logs():
    if not os.path.exists(HABIT_LOGS_CSV):
        pd.DataFrame(columns=["date", "habit", "value"]).to_csv(HABIT_LOGS_CSV, index=False)
    df = pd.read_csv(HABIT_LOGS_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def save_logs(df):
    df.to_csv(HABIT_LOGS_CSV, index=False)

def today_total_for(habit_name, logs_df):
    """Return total progress for a specific habit *today only*."""
    today = date.today()

    
    safe_dates = pd.to_datetime(logs_df["date"], errors="coerce").dt.date

    
    mask = (logs_df["habit"] == habit_name) & (safe_dates == today)

    if mask.sum() == 0:
        return 0.0  

    return float(logs_df.loc[mask, "value"].sum())


habits = load_habits()
logs_df = load_logs()



if "last_opened_date" not in st.session_state:
    st.session_state.last_opened_date = date.today()


if st.session_state.last_opened_date != date.today():
   
    st.session_state.last_opened_date = date.today()

    
    if "chat_history" in st.session_state:
        st.session_state.chat_history = []

    
    st.rerun()



if page == "Home":
    today_str = date.today().strftime("%A, %d %B %Y")

    
    total_done = sum(today_total_for(h["name"], logs_df) for h in habits)
    total_goal = sum(h["goal"] for h in habits)
    progress = int((total_done / total_goal) * 100) if total_goal > 0 else 0

    
    st.markdown(f"""
    <style>
    .sticky-header {{
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding: 20px 25px 15px 25px;
        border-bottom: 1px solid #e5e9f2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}
    .sticky-header h1 {{
        margin-bottom: 0;
        font-weight: 700;
    }}
    .progress-bar-bg {{
        position: relative;
        background: #e5e9f2;
        border-radius: 12px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        margin-top: 12px;
    }}
    .progress-bar-fill {{
        height: 14px;
        border-radius: 12px;
        background: linear-gradient(90deg, #22c55e, #3b82f6);
        width: 0%;
        animation: fillAnim 1.4s ease-in-out forwards;
        position: relative;
    }}
    @keyframes fillAnim {{
        from {{ width: 0%; }}
        to {{ width: {progress}%; }}
    }}
    /* Glowing tip light */
    .glow-tip {{
        content: '';
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        right: 0;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: radial-gradient(circle, #3b82f6 0%, transparent 70%);
        animation: glowPulse 1.5s infinite ease-in-out;
        opacity: 0.7;
    }}
    @keyframes glowPulse {{
        0% {{ transform: translateY(-50%) scale(1); opacity: 0.8; }}
        50% {{ transform: translateY(-50%) scale(1.3); opacity: 1; }}
        100% {{ transform: translateY(-50%) scale(1); opacity: 0.8; }}
    }}
    </style>

    <div class="sticky-header">
        <h1>🏠 Dashboard</h1>
        <p style="margin:0; color:#666;">Welcome back, <b>Akash S</b> 👋 — {today_str}</p>
        <div class="progress-bar-bg">
            <div class="progress-bar-fill">
                <div class="glow-tip"></div>
            </div>
        </div>
        <div style='font-weight:600; margin-top:8px;'>Today's Progress: {progress}% 🎯</div>
    </div>

    <script>
    // JS fallback to ensure animation triggers properly
    const bar = window.parent.document.querySelector('.progress-bar-fill');
    if (bar) {{
        bar.style.width = '{progress}%';
    }}
    </script>
    """, unsafe_allow_html=True)

    
    st.markdown("<div style='margin-bottom:25px;'></div>", unsafe_allow_html=True)



   
    c1, c2, c3 = st.columns(3)
    stats = [
        ("Tracked Habits", len(habits)),
        ("Total Logs", len(logs_df)),
        ("Avg Productivity", f"{ai_df['productivity_1_10'].mean():.1f}/10")
    ]
    for col, (label, val) in zip([c1, c2, c3], stats):
        with col:
            st.markdown(f"""
            <div class='card' style='text-align:center;'>
                <h3 style='margin-bottom:0;'>{val}</h3>
                <span style='color:#777;'>{label}</span>
            </div>
            """, unsafe_allow_html=True)

    
    st.markdown("<br><h3>✅ Today's Habits & Progress</h3>", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, h in enumerate(habits):
        with cols[i % 2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            done = today_total_for(h["name"], logs_df)
            goal = float(h["goal"])
            percent = min(done / goal, 1)
            percent_int = int(percent * 100)

           
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h4 style='margin-bottom:6px;'>{h['name']}</h4>
                <div style="font-weight:600; color:#3867d6;">{done:.0f}/{goal:.0f} {h['unit']}</div>
            </div>
            <div style="
                width:100%; 
                height:10px; 
                border-radius:6px; 
                background:#e5e9f2; 
                margin-top:4px;
            ">
                <div style="
                    width:{percent_int}%; 
                    height:10px; 
                    border-radius:6px; 
                    background:linear-gradient(90deg,#4b7bec,#3867d6);
                    transition:width 0.3s ease;
                "></div>
            </div>
            """, unsafe_allow_html=True)

            
            st.markdown("<br>", unsafe_allow_html=True)
            colA, colB = st.columns([3, 1])
            with colA:
                add_val = st.number_input(
                    f"Add {h['unit']}",
                    min_value=0.0,
                    step=1.0,
                    key=f"add_{h['name']}",
                    label_visibility="collapsed",
                    placeholder=f"Add {h['unit']}..."
                )
            with colB:
                if st.button("Save", key=f"save_{h['name']}", use_container_width=True):
                    new = pd.DataFrame([{
                        "date": pd.Timestamp(date.today()),
                        "habit": h["name"],
                        "value": add_val
                    }])
                    logs_df = pd.concat([logs_df, new], ignore_index=True)
                    save_logs(logs_df)
                    st.success(f"Saved progress for {h['name']} ✅")
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔥 Weekly Streak")

    weekly_streak = calculate_weekly_streak(habits, logs_df)
    week_html = ""
    for active in weekly_streak:
        color = "#22c55e" if active else "#d1d5db"
        week_html += f"<span style='display:inline-block;width:18px;height:18px;margin:4px;border-radius:50%;background:{color};'></span>"

    st.markdown(week_html, unsafe_allow_html=True)
    st.caption("Each green dot represents a productive day this week 🌿")
    st.markdown("</div>", unsafe_allow_html=True)



    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📆 Monthly Productivity Heatmap")

    monthly_data = calculate_monthly_streak(habits, logs_df)
    days_in_month = len(monthly_data)

    heatmap_html = "<div style='display:flex;flex-wrap:wrap;width:260px;'>"
    for i, progress in enumerate(monthly_data, start=1):
        intensity = int(progress * 255)
        color = f"rgb({255-intensity},{255},{255-intensity})"
        heatmap_html += f"<div title='Day {i}: {progress*100:.0f}%' style='width:30px;height:30px;margin:2px;border-radius:6px;background:{color};display:flex;align-items:center;justify-content:center;font-size:12px;'>{i}</div>"
    heatmap_html += "</div>"

    st.markdown(heatmap_html, unsafe_allow_html=True)
    st.caption("Darker colors = more productive days 🌞")
    st.markdown("</div>", unsafe_allow_html=True)

            




elif page == "Habits":
    st.markdown("<h1>🧩 Habit Manager</h1>", unsafe_allow_html=True)
    left, right = st.columns([1.4, 1])

    
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Your Habits")

        if not habits:
            st.info("No habits found. Add one on the right 👉")
        else:
            for idx, h in enumerate(habits):
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                with c1:
                    st.text_input("Name", h["name"], key=f"hname_{idx}", disabled=True)
                with c2:
                    h["unit"] = st.selectbox("Unit", ["Minutes", "Times", "Glasses", "Pages", "Steps"], index=0, key=f"unit_{idx}")
                with c3:
                    h["goal"] = st.number_input("Goal", min_value=1.0, step=1.0, value=float(h["goal"]), key=f"goal_{idx}")
                with c4:
                    if st.button("Delete", key=f"del_{idx}", use_container_width=True):
                        habits.pop(idx)
                        save_habits(habits)
                        st.rerun()

            if st.button("Save Changes ✅", use_container_width=True):
                save_habits(habits)
                st.success("Habits updated successfully!")
        st.markdown("</div>", unsafe_allow_html=True)

    
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Add New Habit")

        name = st.text_input("Habit Name")
        unit = st.selectbox("Unit", ["Minutes", "Times", "Glasses", "Pages", "Steps"])
        goal = st.number_input("Goal", min_value=1.0, step=1.0, value=10.0)

        if st.button("Add Habit", use_container_width=True):
            if not name.strip():
                st.error("Please enter a habit name.")
            elif any(h["name"].lower() == name.strip().lower() for h in habits):
                st.warning("Habit already exists.")
            else:
                habits.append({"name": name.strip(), "unit": unit, "goal": float(goal)})
                save_habits(habits)
                st.success("Habit added successfully ✅")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Analytics":
    st.markdown("<h1>📊 Analytics</h1>", unsafe_allow_html=True)
    st.caption("Visualize your progress and productivity trends over time.")

   
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Habit Activity Over Time")

    logs_df = load_logs()
    if logs_df.empty:
        st.info("No habit logs yet. Add progress on the Dashboard.")
    else:
        df = logs_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.groupby(["date", "habit"], as_index=False)["value"].sum()

        fig = px.line(
            df,
            x="date",
            y="value",
            color="habit",
            markers=True,
            title="Daily Habit Progress"
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("AI Productivity Trend")

    if ai_df.empty:
        st.info("AI data not available yet.")
    else:
        df2 = ai_df.sort_values("date")
        fig2 = px.area(
            df2,
            x="date",
            y="productivity_1_10",
            title="AI Predicted Productivity Over Time",
            color_discrete_sequence=["#4b7bec"]
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Date",
            yaxis_title="Productivity (1–10)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)



elif page == "AI Coach": 
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>🤖 Your AI Productivity Coach</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Chat with your AI mentor for motivation, focus, and habit tips 💬</p>", unsafe_allow_html=True)

    from openai import OpenAI

    
    if not openai_api_key:
        st.error("⚠️ OpenAI API key not found. Please check your .env file.")
    else:
        client = OpenAI(api_key=openai_api_key)

        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        
        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_area(
            "💬 Talk to your coach:",
            placeholder="Hey Coach, I’ve been procrastinating lately...",
            height=100
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            send = st.button("Send 🚀", use_container_width=True)
        with col2:
            clear = st.button("🗑️ Clear Chat", use_container_width=True)

        if clear:
            st.session_state.chat_history = []
            st.experimental_rerun()

        if send:
            if not user_input.strip():
                st.warning("Please type a message before sending.")
            else:
                with st.spinner("💡 Coach is thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": (
                                    "You are a kind, motivational, and professional personal productivity coach. "
                                    "Give concise, powerful, and encouraging advice to help the user stay consistent."
                                )},
                                {"role": "user", "content": user_input}
                            ],
                            max_tokens=250,
                            temperature=0.8
                        )

                        reply = response.choices[0].message.content.strip()
                        st.session_state.chat_history.append(("user", user_input))
                        st.session_state.chat_history.append(("coach", reply))

                    except Exception as e:
                        st.error(f"⚠️ Something went wrong: {e}")

        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("🧠 Chat History")

        if not st.session_state.chat_history:
            st.info("Start chatting with your coach above ☝️")
        else:
            for role, msg in reversed(st.session_state.chat_history):
                if role == "user":
                    st.markdown(
                        f"""
                        <div style='text-align:right;margin:10px 0;'>
                            <div style='display:inline-block;background:linear-gradient(90deg,#3b82f6,#06b6d4);
                            color:white;padding:10px 15px;border-radius:12px 12px 0px 12px;
                            max-width:75%;word-wrap:break-word;'>{msg}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='text-align:left;margin:10px 0;'>
                            <div style='display:inline-block;background:#f1f5f9;color:#1e293b;
                            padding:10px 15px;border-radius:12px 12px 12px 0px;
                            max-width:75%;box-shadow:0 2px 4px rgba(0,0,0,0.1);
                            word-wrap:break-word;'>{msg}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    st.markdown("</div>", unsafe_allow_html=True)





elif page == "Admin Panel":
    st.markdown("<h1>🔐 Admin Panel</h1>", unsafe_allow_html=True)
    st.caption("Train the AI model that powers your productivity predictions.")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Train / Retrain AI Model")

    FEATURE_COLS = [
        "sleep_hours", "study_hours", "screen_time_hours",
        "exercise_minutes", "water_glasses", "mood_1_10"
    ]
    TARGET = "productivity_1_10"

    def train_model(df):
        df = df.dropna(subset=[TARGET])
        if len(df) < 10:
            return None, {"error": "Insufficient data to train (need ≥ 10 rows)."}

        X = df[FEATURE_COLS]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        metrics = {
            "r2": round(r2_score(y_test, preds), 3),
            "mae": round(mean_absolute_error(y_test, preds), 2)
        }
        return model, metrics

    if st.button("Train AI Model 🧠", use_container_width=True):
        model, metrics = train_model(ai_df)
        if model is None:
            st.error(metrics["error"])
        else:
            st.success("✅ Model trained successfully!")
            st.metric("R² Score", metrics["r2"])
            st.metric("MAE", metrics["mae"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Model Info")
    if os.path.exists(MODEL_PATH):
        st.success("Model is available and ready to use.")
    else:
        st.warning("No trained model found yet.")
    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Settings":
    st.markdown("<h1>⚙️ Settings</h1>", unsafe_allow_html=True)
    st.caption("Manage your app data and preferences here.")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Data Management")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "⬇ Download Habit Logs",
            logs_df.to_csv(index=False).encode(),
            file_name="habit_logs.csv"
        )
    with col2:
        st.download_button(
            "⬇ Download AI Data",
            ai_df.to_csv(index=False).encode(),
            file_name="ai_metrics.csv"
        )
    with col3:
        if st.button("🗑️ Reset Habit Logs", use_container_width=True):
            pd.DataFrame(columns=["date", "habit", "value"]).to_csv(HABIT_LOGS_CSV, index=False)
            st.success("All habit logs cleared.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Theme Options (coming soon 🌙)")
    st.caption("Dark mode and custom themes will be added in the next update.")
    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
---
<div style='text-align:center; padding:10px; color:#555;'>
    Made by <b>Akash S</b> | AI Personal Productivity Coach (v2025)
</div>
""", unsafe_allow_html=True)
