# ================= IMPORT PACKAGES =================
import streamlit as st
import sqlite3
import hashlib
import re
import base64

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Dry Eye Detection System", layout="centered")

st.markdown(
    '<h1 style="color:#000000;text-align:center;">AI-Driven Dry Eye Detection System</h1>',
    unsafe_allow_html=True
)

# ================= BACKGROUND IMAGE =================
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Uncomment if image exists
# add_bg_from_local("2.jpg")

# ================= DATABASE FUNCTIONS =================
def create_connection():
    return sqlite3.connect("users.db", check_same_thread=False)

conn = create_connection()

conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    password TEXT,
    email TEXT UNIQUE,
    phone TEXT
)
""")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def user_exists(email):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    return cur.fetchone()

def create_user(name, password, email, phone):
    hashed = hash_password(password)
    conn.execute(
        "INSERT INTO users(name, password, email, phone) VALUES (?, ?, ?, ?)",
        (name, hashed, email, phone),
    )
    conn.commit()

def validate_email(email):
    return re.match(r'^\S+@\S+\.\S+$', email)

def validate_phone(phone):
    return re.match(r'^[6-9]\d{9}$', phone)

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""

# ================= MENU =================
menu = st.sidebar.selectbox("Menu", ["Register", "Login", "Dashboard"])

# ================= REGISTER PAGE =================
if menu == "Register":

    st.subheader("Register Here")

    with st.form("register_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        submit = st.form_submit_button("Register")

    if submit:
        if not name or not email or not phone or not password:
            st.error("All fields are required!")

        elif password != confirm:
            st.error("Passwords do not match!")

        elif not validate_email(email):
            st.error("Invalid email format!")

        elif not validate_phone(phone):
            st.error("Invalid phone number!")

        elif user_exists(email):
            st.error("User already exists!")

        else:
            create_user(name, password, email, phone)
            st.success("Registration successful! Please login.")

# ================= LOGIN PAGE =================
elif menu == "Login":

    st.subheader("Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = user_exists(email)

        if user and user[2] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.user = user[1]
            st.success("Login successful!")

        else:
            st.error("Invalid email or password!")

# ================= DASHBOARD =================
elif menu == "Dashboard":

    if st.session_state.logged_in:

        st.subheader(f"Welcome {st.session_state.user}")

        st.write("✅ You are logged in!")
        st.write("This is your Dry Eye Detection Dashboard.")

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user = ""
            st.success("Logged out successfully!")

    else:
        st.warning("Please login first!")
