import textwrap
import streamlit as st


def render_html(html_str, *args, **kwargs):
    """Render HTML safely without markdown indentation issues."""
    clean_lines = [line.lstrip() for line in html_str.splitlines()]
    clean_str = "\n".join(clean_lines).strip()
    st.markdown(clean_str, unsafe_allow_html=True)


def inject_custom_css():
    """Inject premium glassmorphism dark theme CSS with Google Fonts & icons."""
    render_html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">

        <style>
        /* Global Base */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f1f5f9;
        }

        .stApp {
            background: radial-gradient(circle at 10% 10%, #0d1b38 0%, #070c18 45%, #03060d 100%);
            background-attachment: fixed;
            color: #f8fafc;
        }

        /* Hide unnecessary Streamlit chrome */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: rgba(7, 12, 24, 0.4); backdrop-filter: blur(10px);}

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.6);
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #38bdf8;
        }

        /* Hero Header */
        .oculo-hero {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.75) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 24px 30px;
            margin-bottom: 22px;
            backdrop-filter: blur(18px);
            box-shadow: 0 14px 40px -12px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .oculo-hero::after {
            content: '';
            position: absolute;
            top: -60px;
            right: -40px;
            width: 260px;
            height: 260px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.15) 0%, rgba(0, 0, 0, 0) 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .hero-top-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            padding: 5px 13px;
            border-radius: 50px;
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #38bdf8;
            box-shadow: 0 0 10px #38bdf8;
            animation: pulse-glow 2s infinite;
        }
        @keyframes pulse-glow {
            0% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.8); }
            70% { transform: scale(1.1); box-shadow: 0 0 0 8px rgba(56, 189, 248, 0); }
            100% { transform: scale(0.9); box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }
        .hero-title {
            font-family: 'Outfit', sans-serif;
            font-size: 30px;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
            line-height: 1.2;
        }
        .gradient-brand {
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero-desc {
            color: #94a3b8;
            font-size: 14px;
            margin: 0 0 16px 0;
            line-height: 1.5;
            max-width: 960px;
        }
        .hero-status-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            font-size: 12px;
            color: #94a3b8;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            padding: 5px 11px;
            border-radius: 8px;
            color: #cbd5e1;
            font-weight: 500;
        }
        .status-pill i {
            color: #38bdf8;
        }

        /* Section Cards */
        .section-header-box {
            display: flex;
            align-items: center;
            gap: 14px;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.5) 100%);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 14px 18px;
            margin: 16px 0 20px 0;
            backdrop-filter: blur(12px);
        }
        .section-icon-wrap {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(56, 189, 248, 0.12);
            color: #38bdf8;
            font-size: 20px;
            border: 1px solid rgba(56, 189, 248, 0.25);
            flex-shrink: 0;
        }
        .section-title {
            font-family: 'Outfit', sans-serif;
            font-size: 19px;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
            line-height: 1.25;
        }
        .section-sub {
            font-size: 12.5px;
            color: #94a3b8;
            margin: 2px 0 0 0;
        }

        /* Modern Glass Card */
        .glass-card {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 18px;
            backdrop-filter: blur(14px);
            box-shadow: 0 8px 24px -6px rgba(0, 0, 0, 0.35);
        }

        /* Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 14px;
            margin: 16px 0;
        }
        .metric-card-box {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.65) 100%);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card-box:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.3);
        }
        .metric-card-val {
            font-family: 'Outfit', sans-serif;
            font-size: 24px;
            font-weight: 800;
            color: #38bdf8;
            margin-bottom: 2px;
            letter-spacing: -0.5px;
        }
        .metric-card-lbl {
            font-size: 11.5px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
        }

        /* Image Preview Box */
        .preview-box {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 14px;
            padding: 12px;
            text-align: center;
            margin-bottom: 10px;
        }
        .preview-label {
            font-size: 12px;
            font-weight: 600;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        /* Diagnosis Cards */
        .diag-card {
            border-radius: 18px;
            padding: 24px 26px;
            margin: 20px 0;
            backdrop-filter: blur(16px);
            position: relative;
            overflow: hidden;
            box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.5);
        }
        .diag-affected {
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.16) 0%, rgba(15, 23, 42, 0.85) 100%);
            border: 1.5px solid rgba(244, 63, 94, 0.45);
        }
        .diag-normal {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(15, 23, 42, 0.85) 100%);
            border: 1.5px solid rgba(16, 185, 129, 0.45);
        }
        .diag-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }
        .diag-pill-aff {
            background: rgba(244, 63, 94, 0.2);
            border: 1px solid rgba(244, 63, 94, 0.4);
            color: #fb7185;
            padding: 5px 12px;
            border-radius: 50px;
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }
        .diag-pill-norm {
            background: rgba(16, 185, 129, 0.2);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
            padding: 5px 12px;
            border-radius: 50px;
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }
        .diag-main-title {
            font-family: 'Outfit', sans-serif;
            font-size: 26px;
            font-weight: 800;
            margin: 0 0 6px 0;
            letter-spacing: -0.3px;
        }
        .diag-affected .diag-main-title { color: #fda4af; }
        .diag-normal .diag-main-title { color: #6ee7b7; }
        .diag-subtext {
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.5;
            margin-bottom: 16px;
        }

        /* Progress Bar */
        .conf-bar-wrap {
            margin: 14px 0 18px 0;
        }
        .conf-bar-labels {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 6px;
        }
        .conf-bar-track {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            height: 10px;
            overflow: hidden;
        }
        .conf-bar-fill-aff {
            background: linear-gradient(90deg, #f43f5e 0%, #fb7185 100%);
            height: 100%;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(244, 63, 94, 0.6);
        }
        .conf-bar-fill-norm {
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            height: 100%;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.6);
        }

        /* Recommendations List */
        .recs-box {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 14px 18px;
            margin-top: 14px;
        }
        .recs-heading {
            font-size: 13px;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .recs-list {
            margin: 0;
            padding-left: 20px;
            font-size: 13px;
            color: #94a3b8;
            line-height: 1.6;
        }

        /* Custom Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 10px 22px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            letter-spacing: 0.3px !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
            color: #ffffff !important;
        }

        /* File Uploader styling */
        [data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.5);
            border: 1.5px dashed rgba(56, 189, 248, 0.35);
            border-radius: 16px;
            padding: 14px;
            transition: border-color 0.2s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #38bdf8;
        }

        /* Camera input styling */
        [data-testid="stCameraInput"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.65) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 14px 18px !important;
            margin-bottom: 12px !important;
            backdrop-filter: blur(12px) !important;
        }

        /* Modern Tabs styling */
        /* Modern Tabs as Sleek Segmented Controls */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px !important;
            background: rgba(15, 23, 42, 0.75) !important;
            padding: 6px !important;
            border-radius: 14px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(12px) !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px !important;
            color: #94a3b8 !important;
            padding: 9px 18px !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            border: none !important;
            transition: all 0.2s ease !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.08) !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.45) !important;
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* Container with border (Auth Card & Panels) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 24px !important;
            border: none !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            background: linear-gradient(145deg, rgba(15, 25, 50, 0.88) 0%, rgba(8, 14, 32, 0.96) 100%) !important;
            border: 1.5px solid rgba(56, 189, 248, 0.22) !important;
            border-radius: 24px !important;
            box-shadow:
                0 30px 80px -15px rgba(0, 0, 0, 0.85),
                0 0 60px -20px rgba(56, 189, 248, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
            backdrop-filter: blur(24px) !important;
            -webkit-backdrop-filter: blur(24px) !important;
            padding: 26px 30px !important;
        }

        /* Form Inputs & Controls — aggressive dark theme overrides */
        div[data-testid="stTextInput"] input,
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stTextInput"] input:active,
        input[type="text"],
        input[type="password"] {
            background: rgba(15, 23, 42, 0.95) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: none !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="base-input"] {
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1.5px solid rgba(56, 189, 248, 0.25) !important;
            border-radius: 12px !important;
            transition: all 0.25s ease !important;
            overflow: hidden !important;
        }
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="base-input"]:focus-within {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2), 0 0 20px rgba(56, 189, 248, 0.15) !important;
        }
        input::placeholder {
            color: #64748b !important;
            font-family: 'Plus Jakarta Sans', sans-serif !important;
            -webkit-text-fill-color: #64748b !important;
            opacity: 1 !important;
        }
        /* Password toggle icon */
        button[kind="icon"] svg { fill: #64748b !important; }

        label[data-testid="stWidgetLabel"] p {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            letter-spacing: 0.3px;
        }

        /* Text area */
        textarea {
            background: rgba(15, 23, 42, 0.9) !important;
            color: #f8fafc !important;
            border: 1.5px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 12px !important;
        }
        /* Select box */
        div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.9) !important;
            border: 1.5px solid rgba(56, 189, 248, 0.2) !important;
            border-radius: 12px !important;
            color: #f8fafc !important;
        }

        /* File Uploader styling */
        section[data-testid="stFileUploadDropzone"] {
            background: rgba(15, 23, 42, 0.75) !important;
            border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            transition: all 0.25s ease !important;
        }
        section[data-testid="stFileUploadDropzone"]:hover {
            border-color: #38bdf8 !important;
            background: rgba(15, 23, 42, 0.9) !important;
            box-shadow: 0 0 24px rgba(56, 189, 248, 0.15) !important;
        }
        section[data-testid="stFileUploadDropzone"] div,
        section[data-testid="stFileUploadDropzone"] span,
        section[data-testid="stFileUploadDropzone"] small {
            color: #94a3b8 !important;
        }
        section[data-testid="stFileUploadDropzone"] button {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        }

        /* Auth Portal Card */
        .auth-portal-card {
            background: linear-gradient(145deg, rgba(15, 25, 50, 0.85) 0%, rgba(8, 14, 32, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 24px;
            padding: 0;
            box-shadow:
                0 30px 80px -15px rgba(0, 0, 0, 0.8),
                0 0 60px -20px rgba(56, 189, 248, 0.12),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(24px);
            overflow: hidden;
            position: relative;
        }
        .auth-portal-card::before {
            content: '';
            position: absolute;
            top: -80px; right: -60px;
            width: 260px; height: 260px;
            background: radial-gradient(circle, rgba(56, 189, 248, 0.08) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .auth-portal-card::after {
            content: '';
            position: absolute;
            bottom: -100px; left: -40px;
            width: 300px; height: 300px;
            background: radial-gradient(circle, rgba(129, 140, 248, 0.06) 0%, transparent 70%);
            border-radius: 50%;
            pointer-events: none;
        }
        .auth-card-header {
            padding: 28px 32px 20px 32px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            position: relative;
        }
        .auth-card-body {
            padding: 24px 32px 32px 32px;
            position: relative;
        }
        .auth-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: #38bdf8;
            padding: 4px 12px;
            border-radius: 50px;
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 0.7px;
            text-transform: uppercase;
        }

        /* Floating orbs behind content */
        @keyframes float-orb {
            0%, 100% { transform: translate(0, 0) scale(1); }
            25% { transform: translate(30px, -20px) scale(1.05); }
            50% { transform: translate(-15px, -40px) scale(0.95); }
            75% { transform: translate(20px, 15px) scale(1.02); }
        }
        .orb {
            position: fixed;
            border-radius: 50%;
            filter: blur(80px);
            pointer-events: none;
            z-index: 0;
            opacity: 0.5;
        }
        .orb-1 {
            width: 400px; height: 400px;
            background: rgba(56, 189, 248, 0.08);
            top: 5%; left: 10%;
            animation: float-orb 20s ease-in-out infinite;
        }
        .orb-2 {
            width: 350px; height: 350px;
            background: rgba(129, 140, 248, 0.06);
            bottom: 10%; right: 5%;
            animation: float-orb 25s ease-in-out infinite reverse;
        }
        .orb-3 {
            width: 250px; height: 250px;
            background: rgba(192, 132, 252, 0.05);
            top: 50%; left: 50%;
            animation: float-orb 18s ease-in-out infinite 3s;
        }

        /* Eye scan animation */
        @keyframes iris-scan {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes iris-pulse {
            0%, 100% { box-shadow: 0 0 20px rgba(56, 189, 248, 0.3), inset 0 0 20px rgba(56, 189, 248, 0.1); }
            50% { box-shadow: 0 0 40px rgba(56, 189, 248, 0.5), inset 0 0 30px rgba(56, 189, 248, 0.2); }
        }
        .eye-icon-wrap {
            width: 80px; height: 80px;
            border-radius: 50%;
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.15) 0%, rgba(129, 140, 248, 0.1) 100%);
            border: 2px solid rgba(56, 189, 248, 0.3);
            display: flex; align-items: center; justify-content: center;
            margin: 0 auto 18px auto;
            position: relative;
            animation: iris-pulse 3s ease-in-out infinite;
        }
        .eye-icon-wrap::after {
            content: '';
            position: absolute;
            inset: -4px;
            border-radius: 50%;
            border: 2px dashed rgba(56, 189, 248, 0.2);
            animation: iris-scan 8s linear infinite;
        }
        .eye-icon-wrap i {
            font-size: 32px;
            background: linear-gradient(135deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Feature pills row */
        .feature-pills {
            display: flex; flex-wrap: wrap; gap: 8px;
            justify-content: center;
            margin: 16px 0 4px 0;
        }
        .feature-pill {
            display: inline-flex; align-items: center; gap: 5px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 11.5px;
            color: #94a3b8;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .feature-pill:hover {
            border-color: rgba(56, 189, 248, 0.3);
            color: #38bdf8;
        }
        .feature-pill i { color: #38bdf8; font-size: 12px; }

        /* Demo access cards */
        .demo-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.7) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        .demo-card:hover {
            border-color: rgba(56, 189, 248, 0.35);
            transform: translateY(-3px);
            box-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.5);
        }
        .demo-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .demo-card:hover::before { opacity: 1; }

        /* Animated bottom border for sign-in button */
        @keyframes shimmer {
            0% { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    """Render the master brand header."""
    render_html(
        """
        <div class="oculo-hero">
            <div class="hero-top-row">
                <div class="hero-badge">
                    <span class="pulse-dot"></span> MULTI-SOURCE EVIDENCE SUITE
                </div>
                <div class="hero-status-strip">
                    <span class="status-pill"><i class="bi bi-shield-check"></i> On-Device Engine</span>
                    <span class="status-pill"><i class="bi bi-cpu"></i> 94.5% Accuracy</span>
                    <span class="status-pill"><i class="bi bi-eye"></i> Meibography & Gaze</span>
                </div>
            </div>
            <h1 class="hero-title">
                <span class="gradient-brand">OculoAI</span> Clinical Intelligence
            </h1>
            <p class="hero-desc">
                Advanced AI-driven diagnostics for Dry Eye Disease (DED) combining infrared clinical meibography, real-time ocular camera tracking, 12-item OSDI symptom burden analysis, and blink gaze dynamics.
            </p>
        </div>
        """
    )


def render_section_header(title, subtitle, icon="bi-grid"):
    """Render a consistent styled section title banner."""
    render_html(
        f"""
        <div class="section-header-box">
            <div class="section-icon-wrap">
                <i class="bi {icon}"></i>
            </div>
            <div>
                <div class="section-title">{title}</div>
                <div class="section-sub">{subtitle}</div>
            </div>
        </div>
        """
    )


def render_metrics_row(metrics_list):
    """Render a styled grid of stat cards: [(label, value), ...]"""
    cards_html = ""
    for label, val in metrics_list:
        cards_html += f"""
        <div class="metric-card-box">
            <div class="metric-card-val">{val}</div>
            <div class="metric-card-lbl">{label}</div>
        </div>
        """
    render_html(f'<div class="metric-grid">{cards_html}</div>')


def render_eye_diagnosis(is_affected, prob_aff, prob_not):
    """Render high-impact clinical eye diagnosis card."""
    if is_affected:
        render_html(
            f"""
            <div class="diag-card diag-affected">
                <div class="diag-top">
                    <div class="diag-pill-aff"><i class="bi bi-exclamation-triangle-fill"></i> Positive Finding</div>
                    <span style="font-size: 13px; color: #fda4af; font-weight: 600;">Ocular Surface Screening</span>
                </div>
                <div class="diag-main-title">Identified: Affected (Dry Eye Pathology)</div>
                <div class="diag-subtext">
                    Meibomian gland structural alterations and characteristic infrared reflection patterns consistent with dry eye disease were identified.
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-labels">
                        <span>Pathology Probability: <strong>{prob_aff:.1f}%</strong></span>
                        <span>Normal Probability: {prob_not:.1f}%</span>
                    </div>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill-aff" style="width: {prob_aff:.1f}%"></div>
                    </div>
                </div>
                <div class="recs-box">
                    <div class="recs-heading"><i class="bi bi-clipboard2-pulse"></i> Clinical Guidance & Next Steps</div>
                    <ul class="recs-list">
                        <li><strong>Specialist Consultation:</strong> Schedule an evaluation with an optometrist or ophthalmologist for Meibomian Gland Dysfunction (MGD).</li>
                        <li><strong>Ocular Lubrication:</strong> Preservative-free artificial tears can provide symptomatic relief during screen usage.</li>
                        <li><strong>Eyelid Hygiene:</strong> Warm compress therapy (10 minutes daily) promotes meibum lipid flow.</li>
                        <li><strong>AI Consultation:</strong> Open the <strong>Chatbot</strong> tab for tailored symptom inquiries and lifestyle guidance.</li>
                    </ul>
                </div>
            </div>
            """
        )
    else:
        render_html(
            f"""
            <div class="diag-card diag-normal">
                <div class="diag-top">
                    <div class="diag-pill-norm"><i class="bi bi-check-circle-fill"></i> Negative Finding</div>
                    <span style="font-size: 13px; color: #6ee7b7; font-weight: 600;">Ocular Surface Screening</span>
                </div>
                <div class="diag-main-title">Identified: Not Affected (Normal Ocular Status)</div>
                <div class="diag-subtext">
                    Infrared image demonstrates normal gland distribution with no significant meibomian dropout or dry eye signatures detected.
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-labels">
                        <span>Healthy Probability: <strong>{prob_not:.1f}%</strong></span>
                        <span>Pathology Probability: {prob_aff:.1f}%</span>
                    </div>
                    <div class="conf-bar-track">
                        <div class="conf-bar-fill-norm" style="width: {prob_not:.1f}%"></div>
                    </div>
                </div>
                <div class="recs-box">
                    <div class="recs-heading"><i class="bi bi-heart-pulse"></i> Preventive Ocular Wellness</div>
                    <ul class="recs-list">
                        <li><strong>The 20-20-20 Rule:</strong> Every 20 minutes, look at an object 20 feet away for at least 20 seconds.</li>
                        <li><strong>Conscious Blinking:</strong> Ensure complete blink cycles, especially during intense computer or mobile screen reading.</li>
                        <li><strong>Environmental Humidity:</strong> Maintain adequate ambient humidity in air-conditioned or heated workspaces.</li>
                    </ul>
                </div>
            </div>
            """
        )


def render_gaze_card(label_id):
    """Render gaze / blink diagnosis card."""
    gaze_map = {
        1: ("Eye Closed", "bi-eye-slash-fill", "#94a3b8", "Eyelids fully apposed; blink in progress or resting state."),
        2: ("Forward Look", "bi-bullseye", "#38bdf8", "Central fixation aligned directly with the optical axis."),
        3: ("Left Look", "bi-arrow-left-circle-fill", "#818cf8", "Horizontal lateral gaze directed toward patient left."),
        4: ("Eye Opened", "bi-eye-fill", "#34d399", "Complete palpebral aperture with clear pupil exposure."),
        5: ("Partially Open", "bi-circle-half", "#fbbf24", "Incomplete palpebral fissure closure, typical of partial blink pattern."),
        6: ("Right Look", "bi-arrow-right-circle-fill", "#818cf8", "Horizontal lateral gaze directed toward patient right."),
    }
    title, icon, color, desc = gaze_map.get(label_id, ("Gaze Detected", "bi-eye", "#38bdf8", ""))
    render_html(
        f"""
        <div class="glass-card" style="border-left: 4px solid {color}; margin-top: 18px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <div style="font-size: 32px; color: {color};"><i class="bi {icon}"></i></div>
                <div>
                    <div style="font-size: 12px; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.5px;">Blink & Gaze Orientation</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 22px; font-weight: 700; color: #ffffff;">{title}</div>
                    <div style="font-size: 13.5px; color: #94a3b8; margin-top: 2px;">{desc}</div>
                </div>
            </div>
        </div>
        """
    )


def render_user_bar(user_name):
    """Render top clinician authentication & status bar."""
    render_html(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 10px 18px; margin-bottom: 20px; backdrop-filter: blur(14px); box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 36px; height: 36px; border-radius: 50%; background: linear-gradient(135deg, #0284c7, #2563eb); display: flex; align-items: center; justify-content: center; font-size: 16px; color: #fff; font-weight: 700; box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);">
                    <i class="bi bi-person-fill"></i>
                </div>
                <div>
                    <div style="font-size: 13.5px; font-weight: 700; color: #f8fafc;">
                        Clinician: <span style="color: #38bdf8;">Dr. {user_name}</span>
                    </div>
                    <div style="font-size: 11px; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981;"></span>
                        Session Active • OculoAI Verified Access
                    </div>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span class="status-pill" style="font-size: 11px; padding: 4px 10px;"><i class="bi bi-shield-lock-fill"></i> Encrypted Local Auth</span>
                <span class="status-pill" style="font-size: 11px; padding: 4px 10px; color: #34d399;"><i class="bi bi-activity"></i> Clinical Suite Ready</span>
            </div>
        </div>
        """
    )


def render_auth_hero():
    """Render the login portal hero section with animated eye icon and floating orbs."""
    render_html(
        """
        <!-- Animated floating orbs -->
        <div class="orb orb-1"></div>
        <div class="orb orb-2"></div>
        <div class="orb orb-3"></div>

        <div style="text-align: center; max-width: 680px; margin: 15px auto 25px auto; position: relative; z-index: 1;">
            <!-- Animated eye scan icon -->
            <div class="eye-icon-wrap">
                <i class="bi bi-eye-fill"></i>
            </div>

            <div class="hero-badge" style="margin-bottom: 14px;">
                <span class="pulse-dot"></span> AI-POWERED OCULAR DIAGNOSTICS
            </div>

            <h1 style="font-family: 'Outfit', sans-serif; font-size: 36px; font-weight: 800; color: #ffffff; margin: 0 0 10px 0; letter-spacing: -0.8px; line-height: 1.15;">
                <span class="gradient-brand">OculoAI</span> Clinical<br>Dry Eye Suite
            </h1>

            <p style="color: #94a3b8; font-size: 14px; line-height: 1.6; margin: 0 auto 16px auto; max-width: 520px;">
                Next-generation diagnostic intelligence platform combining infrared meibography, real-time gaze tracking, OSDI-12 neural modeling, and blink dynamics for comprehensive dry eye assessment.
            </p>

            <div class="feature-pills">
                <span class="feature-pill"><i class="bi bi-eye"></i> IR Meibography</span>
                <span class="feature-pill"><i class="bi bi-camera-video"></i> Live CV Tracking</span>
                <span class="feature-pill"><i class="bi bi-clipboard2-data"></i> OSDI-12 Survey</span>
                <span class="feature-pill"><i class="bi bi-activity"></i> Blink Analysis</span>
                <span class="feature-pill"><i class="bi bi-chat-dots"></i> AI Chatbot</span>
            </div>
        </div>
        """
    )


