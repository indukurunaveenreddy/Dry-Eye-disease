"""Login and Authentication Portal for OculoAI Dry Eye Diagnostic Suite."""

import sqlite3
import runpy
import streamlit as st
import ui_theme

# Page Configuration
st.set_page_config(
    page_title="OculoAI • Clinical Dry Eye Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply unified design system
ui_theme.inject_custom_css()


# ─── Database helpers ───────────────────────────────────────────────

def create_connection(db_file="dbs.db"):
    """Create a database connection to SQLite."""
    try:
        return sqlite3.connect(db_file)
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return None


def init_db(conn):
    """Ensure the users table exists."""
    if conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                password TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone TEXT NOT NULL
            );"""
        )
        conn.commit()


def create_user(conn, user):
    """Insert a new user record."""
    cur = conn.cursor()
    cur.execute("INSERT INTO users(name, password, email, phone) VALUES(?,?,?,?)", user)
    conn.commit()
    return cur.lastrowid


def validate_user(conn, identity, password):
    """Validate credentials matching username or email."""
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE (name=? OR email=?) AND password=?",
        (identity, identity, password),
    )
    user = cur.fetchone()
    return (True, user[1]) if user else (False, None)


# ─── Login portal ───────────────────────────────────────────────────

def render_login_portal():
    """Render the clinical login & registration portal."""

    # Animated hero with eye scan icon + floating orbs
    ui_theme.render_auth_hero()

    conn = create_connection("dbs.db")
    if conn:
        init_db(conn)

    # ── Centered auth card ──
    _, col_center, _ = st.columns([1.1, 2.2, 1.1])

    with col_center:
        with st.container(border=True):
            ui_theme.render_html(
                """
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">
                    <div class="auth-badge"><i class="bi bi-shield-lock-fill"></i> SECURE CLINICAL ACCESS</div>
                    <div style="font-size: 11px; color: #64748b; display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>
                        System Online • TLS Enclave
                    </div>
                </div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 22px; font-weight: 700; color: #ffffff; letter-spacing: -0.3px; margin-bottom: 4px;">
                    Clinician Authentication Portal
                </div>
                <div style="font-size: 13px; color: #94a3b8; margin-bottom: 20px; line-height: 1.5;">
                    Sign in to access AI meibography analysis, gaze tracking & clinical records.
                </div>
                """
            )

            # ── Tabs ──
            tab_login, tab_register, tab_demo = st.tabs([
                "🔐  Sign In",
                "📝  New Clinician",
                "⚡  Instant Demo",
            ])

            # ─────────── TAB 1: SIGN IN ───────────
            with tab_login:
                st.markdown("")  # spacer

                login_identity = st.text_input(
                    "Username or Email",
                    placeholder="e.g. aaa or doctor@hospital.org",
                    key="input_login_id",
                )
                login_password = st.text_input(
                    "Access Key",
                    type="password",
                    placeholder="Enter your password",
                    key="input_login_pw",
                )

                ui_theme.render_html("<div style='height:8px'></div>")

                if st.button(
                    "🔒  Sign In to Diagnostic Suite",
                    key="btn_signin",
                    use_container_width=True,
                ):
                    if not login_identity or not login_password:
                        st.warning("Please provide both username and password.")
                    elif conn is None:
                        st.error("Cannot reach local authentication database.")
                    else:
                        ok, name = validate_user(conn, login_identity.strip(), login_password)
                        if ok:
                            st.session_state["authenticated"] = True
                            st.session_state["user_name"] = name
                            st.success(f"Welcome back, Dr. {name}! Launching clinical suite …")
                            st.rerun()
                        else:
                            st.error("Invalid credentials — please check username and password.")

                # Quick-access hint
                ui_theme.render_html(
                    """
                    <div style="text-align: center; margin-top: 16px; font-size: 12px; color: #64748b;">
                        <i class="bi bi-info-circle"></i>
                        Don't have an account? Switch to <strong style="color: #94a3b8;">New Clinician</strong>,
                        or use <strong style="color: #38bdf8;">Instant Demo</strong> for one-click entry.
                    </div>
                    """
                )

            # ─────────── TAB 2: REGISTER ───────────
            with tab_register:
                st.markdown("")

                r_col1, r_col2 = st.columns(2)
                with r_col1:
                    reg_name = st.text_input("Clinician Name", placeholder="Dr. Smith", key="r_name")
                with r_col2:
                    reg_email = st.text_input("Institutional Email", placeholder="smith@hospital.org", key="r_email")

                r_col3, r_col4 = st.columns(2)
                with r_col3:
                    reg_phone = st.text_input("Phone Number", placeholder="+91 98765 43210", key="r_phone")
                with r_col4:
                    reg_pass = st.text_input("Set Password", type="password", placeholder="Min 3 chars", key="r_pass")

                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="r_confirm")

                ui_theme.render_html("<div style='height:8px'></div>")

                if st.button("📝  Create Clinician Profile", key="btn_register", use_container_width=True):
                    if not all([reg_name, reg_email, reg_phone, reg_pass]):
                        st.warning("All fields are required.")
                    elif "@" not in reg_email or "." not in reg_email:
                        st.warning("Enter a valid email address.")
                    elif reg_pass != reg_confirm:
                        st.error("Passwords do not match.")
                    elif len(reg_pass) < 3:
                        st.warning("Password must be ≥ 3 characters.")
                    elif conn is None:
                        st.error("Database unavailable.")
                    else:
                        try:
                            create_user(conn, (reg_name.strip(), reg_pass, reg_email.strip(), reg_phone.strip()))
                            st.success("✅ Registration successful! Switch to the Sign In tab.")
                        except sqlite3.IntegrityError:
                            st.error("An account with this email already exists.")
                        except Exception as ex:
                            st.error(f"Registration failed: {ex}")

            # ─────────── TAB 3: INSTANT DEMO ───────────
            with tab_demo:
                ui_theme.render_html(
                    """
                    <div style="font-size: 13px; color: #94a3b8; margin: 10px 0 16px 0; text-align: center; line-height: 1.5;">
                        Select a preconfigured clinician profile to enter the full diagnostic suite immediately — no passwords required.
                    </div>
                    """
                )

                d_col1, d_col2 = st.columns(2)

                with d_col1:
                    ui_theme.render_html(
                        """
                        <div class="demo-card" style="margin-bottom: 10px;">
                            <div style="font-size: 26px; margin-bottom: 6px;">👨‍⚕️</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 700; color: #f8fafc;">Dr. Naveen</div>
                            <div style="font-size: 11px; color: #38bdf8; margin-top: 2px;">Lead Ophthalmologist</div>
                        </div>
                        """
                    )
                    if st.button("Launch as Dr. Naveen", key="d_naveen", use_container_width=True):
                        st.session_state.update(authenticated=True, user_name="Naveen")
                        st.rerun()

                with d_col2:
                    ui_theme.render_html(
                        """
                        <div class="demo-card" style="margin-bottom: 10px;">
                            <div style="font-size: 26px; margin-bottom: 6px;">👁️</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 700; color: #f8fafc;">Dr. Basha</div>
                            <div style="font-size: 11px; color: #38bdf8; margin-top: 2px;">Cornea Specialist</div>
                        </div>
                        """
                    )
                    if st.button("Launch as Dr. Basha", key="d_basha", use_container_width=True):
                        st.session_state.update(authenticated=True, user_name="Basha")
                        st.rerun()

                d_col3, d_col4 = st.columns(2)

                with d_col3:
                    ui_theme.render_html(
                        """
                        <div class="demo-card" style="margin-top: 12px; margin-bottom: 10px;">
                            <div style="font-size: 26px; margin-bottom: 6px;">🔬</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 700; color: #f8fafc;">Clinical Researcher</div>
                            <div style="font-size: 11px; color: #818cf8; margin-top: 2px;">AI Validation Engineer</div>
                        </div>
                        """
                    )
                    if st.button("Launch Researcher", key="d_research", use_container_width=True):
                        st.session_state.update(authenticated=True, user_name="Researcher")
                        st.rerun()

                with d_col4:
                    ui_theme.render_html(
                        """
                        <div class="demo-card" style="margin-top: 12px; margin-bottom: 10px;">
                            <div style="font-size: 26px; margin-bottom: 6px;">⚡</div>
                            <div style="font-family: 'Outfit', sans-serif; font-size: 15px; font-weight: 700; color: #f8fafc;">Guest Clinician</div>
                            <div style="font-size: 11px; color: #34d399; margin-top: 2px;">Full Evaluation Access</div>
                        </div>
                        """
                    )
                    if st.button("Launch Guest Demo", key="d_guest", use_container_width=True):
                        st.session_state.update(authenticated=True, user_name="Guest Clinician")
                        st.rerun()

    if conn:
        conn.close()

    # ── Bottom trust bar ──
    ui_theme.render_html(
        """
        <div style="max-width: 720px; margin: 32px auto 16px auto; text-align: center;
                    border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 22px;">
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 18px;
                        font-size: 12px; color: #64748b;">
                <span><i class="bi bi-database-check" style="color: #38bdf8;"></i> Local SQLite Auth</span>
                <span style="color: #334155;">•</span>
                <span><i class="bi bi-shield-shaded" style="color: #10b981;"></i> 100% On-Device Processing</span>
                <span style="color: #334155;">•</span>
                <span><i class="bi bi-cpu" style="color: #818cf8;"></i> PyTorch & CV Models</span>
                <span style="color: #334155;">•</span>
                <span><i class="bi bi-lock-fill" style="color: #fbbf24;"></i> Zero Cloud Data Upload</span>
            </div>
            <div style="font-size: 11px; color: #475569; margin-top: 12px; letter-spacing: 0.3px;">
                OculoAI v2.0 • AI-Driven Advanced Dry Eye Diagnostics • Multi-Source Evidence Platform
            </div>
        </div>
        """
    )


# ─── Main entry ─────────────────────────────────────────────────────

def main():
    """Route between login portal and diagnostic suite."""
    if st.session_state.get("authenticated", False):
        runpy.run_path("Prediction.py")
    else:
        render_login_portal()


if __name__ == "__main__":
    main()