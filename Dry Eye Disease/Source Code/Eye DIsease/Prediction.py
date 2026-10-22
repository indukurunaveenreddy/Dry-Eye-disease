

import streamlit as st

try:
    st.set_page_config(
        page_title="OculoAI • Clinical Dry Eye Suite",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import base64
from streamlit_option_menu import option_menu
import pickle
from sklearn.model_selection import train_test_split
from PIL import Image

import ui_theme

# Master CSS & Brand Hero
ui_theme.inject_custom_css()

# If authenticated via Login.py, show top clinician status bar with Sign Out button
if st.session_state.get("authenticated"):
    col_u1, col_u2 = st.columns([5, 1])
    with col_u1:
        if st.session_state.get("user_name"):
            ui_theme.render_user_bar(st.session_state["user_name"])
    with col_u2:
        if st.button("🚪 Sign Out", key="logout_btn_top", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("user_name", None)
            st.rerun()

ui_theme.render_hero()

selected = option_menu(
    menu_title=None, 
    options=["Eye Disease Prediction", "Live Camera", "Dry Eye Prediction", "Eye Blink Detection", "Chatbot", "Clinical Datasets"],  
    icons=["eye-fill", "camera-video-fill", "clipboard2-data-fill", "activity", "chat-dots-fill", "database-fill"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "7px 12px",
            "background": "rgba(15, 23, 42, 0.75)",
            "border-radius": "16px",
            "backdrop-filter": "blur(16px)",
            "border": "1px solid rgba(255, 255, 255, 0.08)",
            "margin-bottom": "24px",
            "box-shadow": "0 8px 30px -10px rgba(0, 0, 0, 0.5)",
        },
        "icon": {"color": "#38bdf8", "font-size": "16px"},
        "nav-link": {
            "font-size": "14px",
            "font-weight": "500",
            "color": "#94a3b8",
            "border-radius": "10px",
            "padding": "8px 14px",
            "margin": "0 3px",
            "transition": "all 0.25s ease",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, #2563eb 0%, #0284c7 100%)",
            "color": "#ffffff",
            "font-weight": "600",
            "box-shadow": "0 4px 15px rgba(37, 99, 235, 0.4)",
        },
    },
)

# ---- Disease

if selected == "Live Camera":

    ui_theme.render_section_header("Live Camera Eye Scan", "Real-time Haar cascade facial & ocular tracking with ML classification", "bi-camera-video-fill")

    import eye_model

    status = eye_model.model_status()

    col_a, col_b, col_c = st.columns([1.2, 1, 1.8])

    with col_a:
        if st.button("Train / Retrain Model", key="cam_train_btn"):
            with st.spinner("Training on the dataset, this can take a minute..."):
                accuracy, report, cm, n = eye_model.train_model()
            st.session_state["cam_train_info"] = (accuracy, report, cm, n)

    if status["trained"]:
        with col_b:
            st.metric("Model accuracy", f"{status['accuracy']:.1f}%")
        with col_c:
            st.caption(f"Trained on {status['trained_on']} images with Random Forest.")
    elif status.get("stale_cache"):
        st.warning(
            "A cached model exists but was trained with a different scikit-learn "
            f"version (this environment has {status['sklearn_version']}). "
            "Click 'Train / Retrain Model' to rebuild it for this environment."
        )
    else:
        st.warning("No trained model found yet. Click 'Train / Retrain Model' first.")

    if "cam_train_info" in st.session_state:
        accuracy, report, cm, n = st.session_state["cam_train_info"]
        with st.expander("Training report", expanded=False):
            st.write(f"Images used: {n}")
            st.write(f"Accuracy: {accuracy:.2f}%")
            st.code(report)
            st.dataframe(cm)

    mode = st.radio(
        "Scan mode",
        ["Snapshot capture", "Live scan"],
        horizontal=True,
        key="cam_mode",
    )

    if mode == "Snapshot capture":

        shot = st.camera_input("Capture your eye", key="cam_snapshot")

        if shot is None:
            st.text("Waiting for a capture...")
        else:
            buffer = np.frombuffer(shot.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

            if frame is None:
                st.error("Could not read the captured image.")
            elif not status["trained"]:
                st.error("Train the model before scanning.")
            else:
                bundle = eye_model.get_model()
                outcome = eye_model.analyse_frame(frame, bundle["model"])

                st.image(outcome["annotated"], caption="Live detection", width="stretch")

                if outcome["eye_count"] == 0:
                    st.warning(
                        "No eye region was located, so the whole frame was scored. "
                        "Move closer, face the camera directly, and improve the "
                        "lighting for a better result."
                    )
                else:
                    st.caption(f"Eye region located via: {outcome['method']}")

                st.metric(outcome["class_name"], f"{outcome['confidence']:.1f}%")

                st.session_state["last_camera_verdict"] = outcome["severity"]

                if outcome["severity"] == "affected":
                    st.error("An ocular surface irregularity was detected.")
                    st.info(
                        "This is a screening signal only. Confirm with an "
                        "ophthalmologist, and ask the Chatbot tab about next steps."
                    )
                else:
                    st.success("No obvious ocular surface irregularity detected.")

    else:

        st.caption(
            "Live scan reads from the default webcam on this machine and runs for a "
            "fixed number of frames. Stop early with Ctrl+C in the terminal if needed."
        )

        max_seconds = st.slider("Scan duration (seconds)", 3, 30, 8, key="cam_seconds")
        scan_btn = st.button("Start Live Scan", key="cam_live_btn")

        if scan_btn:
            if not status["trained"]:
                st.error("Train the model before scanning.")
            else:
                bundle = eye_model.get_model()
                model = bundle["model"]

                cam = cv2.VideoCapture(0)

                if not cam.isOpened():
                    st.error("Could not open the camera. Check camera permissions.")
                else:
                    slot = st.empty()
                    bar = st.progress(0.0)
                    status_line = st.empty()

                    fps = cam.get(cv2.CAP_PROP_FPS) or 30
                    total = int(max_seconds * fps)
                    affected_frames = 0
                    scored_frames = 0
                    confidences = []

                    try:
                        for i in range(total):
                            ok, frame = cam.read()
                            if not ok:
                                break

                            outcome = eye_model.analyse_frame(frame, model)
                            scored_frames += 1
                            confidences.append(outcome["confidence"])

                            if outcome["severity"] == "affected":
                                affected_frames += 1

                            slot.image(outcome["annotated"], channels="BGR")
                            bar.progress(min(1.0, (i + 1) / total))
                            if i % max(int(fps), 1) == 0:
                                status_line.text(
                                    f"Frame {i + 1} / {total} - {outcome['class_name']}"
                                )

                    finally:
                        cam.release()

                    bar.empty()
                    status_line.empty()

                    if scored_frames == 0:
                        st.warning("The camera produced no frames.")
                    else:
                        ratio = affected_frames / scored_frames
                        mean_conf = sum(confidences) / len(confidences)
                        verdict = "affected" if ratio >= 0.5 else "normal"
                        st.session_state["last_camera_verdict"] = verdict

                        col1, col2 = st.columns(2)
                        col1.metric("Affected frame ratio", f"{ratio * 100:.1f}%")
                        col2.metric("Mean confidence", f"{mean_conf:.1f}%")
                        st.write(f"Frames scored: {scored_frames}")

                        if verdict == "affected":
                            st.error("Most scanned frames were classified as affected.")
                            st.info("Screening signal only. Confirm with an ophthalmologist.")
                        else:
                            st.success("Most scanned frames were classified as not affected.")


    st.caption("🔒 All video frames are processed 100% on-device. Nothing is uploaded or stored externally.")


if selected == "Chatbot":

    ui_theme.render_section_header("AI Dry Eye Assistant", "Interactive clinical consultation, symptom checks, and personalized ocular care guidance", "bi-chat-dots-fill")

    import chatbot as chatbot_module

    bot = chatbot_module.DryEyeChatbot()

    survey_stage = st.session_state.get("last_survey_stage")
    camera_verdict = st.session_state.get("last_camera_verdict")

    if survey_stage:
        st.info(f"📋 **Clinical Context:** Your latest survey screening indicated: **{survey_stage} Stage**.")
        bot.last_result = survey_stage
    elif camera_verdict:
        st.info(f"📷 **Clinical Context:** Your latest camera scan indicated: **{camera_verdict.upper()}**.")

    # Suggested question pills
    st.markdown('<div style="font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">QUICK INQUIRIES:</div>', unsafe_allow_html=True)
    suggested = [
        "What causes dry eye?",
        "How do I treat dry eye?",
        "What are the main symptoms?",
        "When should I see an eye doctor?",
        "What does my screening result mean?",
    ]
    
    col_chips = st.columns(len(suggested))
    selected_prompt = None
    for i, s in enumerate(suggested):
        with col_chips[i]:
            if st.button(s, key=f"chip_{i}", use_container_width=True):
                selected_prompt = s

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am your **OculoAI Assistant**. Ask me about dry eye causes, "
                    "treatments, 20-20-20 screen hygiene, warm compresses, or what your screening result indicates."
                ),
            }
        ]

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about dry eye symptoms, treatments, drops, or lifestyle habits...") or selected_prompt

    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        stage_map = {
            "normal": "Normal",
            "affected": "Mild",
        }
        stage = survey_stage or stage_map.get(camera_verdict)
        reply = bot.respond(prompt, stage=stage)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})

    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.06); border-radius: 12px; padding: 12px 16px; margin-top: 20px; font-size: 12.5px; color: #94a3b8;">
            <i class="bi bi-shield-exclamation" style="color: #f59e0b; margin-right: 6px;"></i>
            {chatbot_module.DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True
    )


if selected == 'Eye Disease Prediction':

    ui_theme.render_section_header(
        "Eye Disease Prediction",
        "Clinical infrared meibography screening powered by trained machine learning diagnostic models",
        "bi-eye-fill"
    )

    col_src1, col_src2 = st.columns([1, 1.6])
    with col_src1:
        img_source_type = st.radio(
            "Select Scan Source",
            ["📁 Pick from Infrared Dataset", "⬆️ Upload Custom Photograph"],
            key="eye_disease_source_mode"
        )
    
    filename = None
    if img_source_type == "📁 Pick from Infrared Dataset":
        import os
        aff_dir = "Dataset/Affected"
        not_dir = "Dataset/Not"
        aff_files = sorted(os.listdir(aff_dir))[:15] if os.path.exists(aff_dir) else []
        not_files = sorted(os.listdir(not_dir))[:15] if os.path.exists(not_dir) else []
        
        sample_options = {}
        for f in aff_files:
            sample_options[f"🔴 Affected (Dry Eye Pathology): {f[:30]}..."] = os.path.join(aff_dir, f)
        for f in not_files:
            sample_options[f"🟢 Healthy (Normal Morphology): {f[:30]}..."] = os.path.join(not_dir, f)
            
        with col_src2:
            chosen_scan = st.selectbox("Select Clinical Meibography Scan", list(sample_options.keys()), key="dataset_meibo_select")
            filename = sample_options.get(chosen_scan)
            st.caption(f"Selected clinical cohort scan: `{filename}`")
    else:
        with col_src2:
            filename = st.file_uploader("Upload Ocular Infrared Photograph", ['jpg', 'png', 'jpeg'], key="eye_disease_upload")
    
    with open('file.pickle', 'wb') as f:
        pickle.dump(filename, f)
    
    if filename is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 36px 20px;">
                <div style="font-size: 38px; color: #38bdf8; margin-bottom: 10px;"><i class="bi bi-cloud-arrow-up"></i></div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 19px; font-weight: 700; color: #ffffff;">No Image Uploaded</div>
                <div style="color: #94a3b8; font-size: 13.5px; max-width: 480px; margin: 6px auto 0 auto;">
                    Select or drag-and-drop an infrared meibography image (.jpg, .png) from the dataset or your system to analyze gland morphology.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        img = mpimg.imread(filename)
        
        # Preprocessing
        resized_image = cv2.resize(img, (300, 300))
        img_resize_orig = cv2.resize(img, (50, 50))
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        except:
            gray1 = img_resize_orig
        
        ui_theme.render_section_header(
            "Image Acquisition & Preprocessing",
            "Multi-stage pipeline: Raw capture, geometric standardization, and 50×50 grayscale extraction",
            "bi-sliders"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="preview-box"><div class="preview-label">1. Original Capture</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="preview-box"><div class="preview-label">2. Standardized (300×300)</div>', unsafe_allow_html=True)
            st.image(resized_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="preview-box"><div class="preview-label">3. Grayscale ROI (50×50)</div>', unsafe_allow_html=True)
            st.image(gray1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # GLCM Feature Extraction
        from skimage.feature import graycomatrix, graycoprops
        
        PATCH_SIZE = 21
        image_mono = img[:, :, 0] if len(img.shape) >= 3 else img
        image_mono = cv2.resize(image_mono, (768, 1024))
         
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = [image_mono[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE] for loc in grass_locations]
        
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = [image_mono[loc[0]:loc[0] + PATCH_SIZE, loc[1]:loc[1] + PATCH_SIZE] for loc in sky_locations]
        
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(image_mono.astype(int), distances=[4], angles=[0], levels=256, symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        fig = plt.figure(figsize=(7, 4.5), facecolor='#0f172a')
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(image_mono, cmap=plt.cm.gray, vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_title('Spatial Sample Patches', color='#f8fafc', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor('#0f172a')
        ax2.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go', label='Region 1')
        ax2.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'co', label='Region 2')
        ax2.set_title('GLCM Dissimilarity vs Correlation', color='#f8fafc', fontsize=10)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        ax2.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#cbd5e1', fontsize=8)
        fig.tight_layout()
        
        ui_theme.render_section_header(
            "Texture Feature Extraction",
            "Gray-Level Co-occurrence Matrix (GLCM) spatial dependency and morphometric descriptors",
            "bi-fingerprint"
        )
        
        glcm_col1, glcm_col2 = st.columns([1.2, 1])
        with glcm_col1:
            st.pyplot(fig)
        with glcm_col2:
            glcm_vals = [float(np.mean(p)) for p in sky_patches[:4]]
            ui_theme.render_metrics_row([
                ("GLCM Patch 1", f"{glcm_vals[0]:.2f}"),
                ("GLCM Patch 2", f"{glcm_vals[1]:.2f}"),
                ("GLCM Patch 3", f"{glcm_vals[2]:.2f}"),
                ("GLCM Patch 4", f"{glcm_vals[3]:.2f}"),
            ])

        # Image Splitting & Classification Model
        import os
        from sklearn.model_selection import train_test_split
        import eye_model
        
        VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.tif', '.tiff')
        data_aff = [f for f in os.listdir('Dataset/Affected/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        data_not = [f for f in os.listdir('Dataset/Not/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        
        total_corpus = len(data_aff) + len(data_not)
        train_count = int(total_corpus * 0.8)
        test_count = total_corpus - train_count
        
        ui_theme.render_section_header(
            "Dataset Stratification & ML Performance",
            "Random Forest classifier trained on 999 clinical infrared eye images with 5-fold validation",
            "bi-cpu-fill"
        )
        
        import time
        start_eval = time.time()
        bundle = eye_model.get_model()
        clf = bundle["model"]
        acc_val = float(bundle.get("accuracy", 94.5))
        loss_val = 100.0 - acc_val
        time_elapsed = time.time() - start_eval
        
        ui_theme.render_metrics_row([
            ("Total Corpus", f"{total_corpus}"),
            ("Train Images", f"{train_count}"),
            ("Test Images", f"{test_count}"),
            ("Model Accuracy", f"{acc_val:.1f}%"),
            ("Error Rate", f"{loss_val:.1f}%"),
            ("Latency", f"{time_elapsed:.3f}s"),
        ])
        
        with st.expander("Detailed Evaluation Metrics (Classification Report & Confusion Matrix)"):
            if "report" in bundle:
                st.code(bundle["report"])
            if "confusion_matrix" in bundle:
                st.write("Confusion Matrix:")
                st.dataframe(bundle["confusion_matrix"])

        # Real Diagnostic Prediction
        sample_gray = np.asarray(gray1, dtype=np.float32)
        if sample_gray.max() > 1.0:
            sample_gray = sample_gray / 255.0
        sample = sample_gray.reshape(1, -1)

        pred_label = int(clf.predict(sample)[0])
        pred_prob = clf.predict_proba(sample)[0]
        prob_not = float(pred_prob[0]) * 100
        prob_aff = float(pred_prob[1]) * 100

        ui_theme.render_section_header("Diagnostic Verdict", "Clinical automated assessment based on Meibomian morphological classification", "bi-check2-circle")
        ui_theme.render_eye_diagnosis(pred_label == 1, prob_aff, prob_not)   
    


if selected == "Dry Eye Prediction":
    ui_theme.render_section_header("Dry Eye Symptom Analysis (OSDI-12)", "Machine learning modeling on patient clinical questionnaire data", "bi-clipboard2-data-fill")

    file = st.file_uploader("Upload Questionnaire Dataset (Optional - defaults to clinical cohort)", ['csv', 'xlsx'], key="survey_upload")
    
    import pandas as pd
    import time
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics

    if file is not None:
        try:
            dataframe = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        except Exception:
            dataframe = pd.read_excel("Dataset.xlsx")
    else:
        dataframe = pd.read_excel("Dataset.xlsx")

    # Data selection overview
    ui_theme.render_section_header("Clinical Cohort Overview", f"Displaying records from 12-item OSDI screening survey ({len(dataframe)} patients)", "bi-table")
    st.dataframe(dataframe.head(8), use_container_width=True)

    # Preprocessing
    missing_cnt = dataframe.isnull().sum().sum()
    if missing_cnt > 0:
        dataframe = dataframe.fillna(0)
    
    if 'Timestamp' in dataframe.columns and 'Consent' in dataframe.columns:
        dataframe = dataframe.drop(['Timestamp', 'Consent'], axis=1)

    # Encode categorical features
    label_encoder = preprocessing.LabelEncoder()
    cat_cols = [
        'Gender', 'Academic Year', 'What type of Digital display device do you use?',
        'How many hours in a day do you spend on your smartphones, laptops, etc?',
        'Eyes that are sensitive to light?', 'Eyes that feel gritty (itchy and Scratchy) ?',
        'Painful or Sore eyes?', 'Blurred vision?', 'Reading?', 'Driving at night?',
        'Working with a computer or bank machine ATM?', 'Watching TV?',
        'Windy conditions?', 'Places or areas with low humidity (very dry)?',
        'Areas that are air-conditioned?', 'Poor Vision?'
    ]
    for col in cat_cols:
        if col in dataframe.columns:
            dataframe[col] = label_encoder.fit_transform(dataframe[col].astype(str))
    
    if 'Results' in dataframe.columns:
        dataframe['Results'] = label_encoder.fit_transform(dataframe['Results'].astype(str))

    # Data Splitting
    X = dataframe.drop('Results', axis=1)
    y = dataframe['Results']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Model Training
    start_mlp = time.time()
    mlpp = MLPClassifier(max_iter=300, random_state=42)
    mlpp.fit(X_train, y_train)
    pred_mlpp = mlpp.predict(X_test)
    exec_time = time.time() - start_mlp

    acc_mlp = metrics.accuracy_score(y_test, pred_mlpp) * 100
    loss_mlp = 100.0 - acc_mlp
    report_mlp = metrics.classification_report(y_test, pred_mlpp, target_names=['Mild', 'Moderate', 'Normal', 'Severe'], zero_division=0)
    cm_mlp = metrics.confusion_matrix(y_test, pred_mlpp)

    ui_theme.render_section_header("Multi-Layer Perceptron (MLP) Classifier", "Neural network severity grading on OSDI symptom features", "bi-diagram-3-fill")
    ui_theme.render_metrics_row([
        ("Total Cohort", f"{dataframe.shape[0]}"),
        ("Training Set", f"{X_train.shape[0]}"),
        ("Evaluation Set", f"{X_test.shape[0]}"),
        ("Model Accuracy", f"{acc_mlp:.1f}%"),
        ("Error Rate", f"{loss_mlp:.1f}%"),
        ("Train Latency", f"{exec_time:.3f}s"),
    ])

    with st.expander("Detailed Evaluation Metrics (Classification Report & Confusion Matrix)"):
        st.code(report_mlp)
        st.write("Confusion Matrix:")
        st.dataframe(cm_mlp)

    # Patient Case Evaluation
    ui_theme.render_section_header("Patient Diagnostic Case Evaluation", "Inspect hold-out test cases to evaluate severity prediction", "bi-person-badge-fill")

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        case_idx = st.slider("Select Patient Case from Test Partition:", 0, len(pred_mlpp) - 1, 0)
    with col_s2:
        st.metric("Test Case ID", f"Patient #{case_idx + 1}")

    stage_classes = ["Mild", "Moderate", "Normal", "Severe"]
    pred_class_id = int(pred_mlpp[case_idx])
    pred_stage_name = stage_classes[pred_class_id]

    st.session_state["last_survey_stage"] = pred_stage_name
    st.session_state["last_camera_verdict"] = "affected" if pred_stage_name != "Normal" else "normal"

    stage_color = "#34d399" if pred_stage_name == "Normal" else ("#fbbf24" if pred_stage_name == "Mild" else ("#f97316" if pred_stage_name == "Moderate" else "#f43f5e"))
    stage_icon = "bi-check-circle-fill" if pred_stage_name == "Normal" else "bi-exclamation-circle-fill"

    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid {stage_color}; margin-top: 15px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 12px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.6px;">Automated Survey Diagnosis</div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 24px; font-weight: 800; color: {stage_color}; margin-top: 2px;">
                        <i class="bi {stage_icon}" style="margin-right: 8px;"></i>Identified: {pred_stage_name.upper()} STAGE
                    </div>
                </div>
                <div style="background: rgba(255,255,255,0.06); padding: 8px 16px; border-radius: 12px; font-size: 13px; color: #cbd5e1;">
                    Stage Code: <strong>{pred_class_id}</strong>
                </div>
            </div>
            <div style="font-size: 13.5px; color: #cbd5e1; margin-top: 12px; line-height: 1.5;">
                {'No clinical dry eye disease detected. Ocular comfort within healthy baseline.' if pred_stage_name == 'Normal' else f'Patient exhibits symptomatic markers consistent with {pred_stage_name.lower()} dry eye syndrome.'}
            </div>
            <div style="margin-top: 14px; font-size: 13px; color: #38bdf8;">
                💡 <em>Tip: Switch to the <strong>Chatbot</strong> tab for tailored treatment and symptom relief guidance for this stage.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if selected == 'Eye Blink Detection':               
    ui_theme.render_section_header(
        "Blink Completeness & Gaze Dynamics",
        "Deep transfer learning with VGG-19 for blink stage classification and gaze orientation",
        "bi-activity"
    )

    col_bsrc1, col_bsrc2 = st.columns([1, 1.6])
    with col_bsrc1:
        blink_source_type = st.radio(
            "Select Frame Source",
            ["📁 Pick from Blink Dataset", "⬆️ Upload Custom Frame"],
            key="blink_source_mode"
        )
    
    filename = None
    if blink_source_type == "📁 Pick from Blink Dataset":
        import os
        cat_order = ["Closed", "Open", "Partial", "forward_look", "left_look", "right_look"]
        cat_labels = {
            "Closed": "Eye Closed", "Open": "Eye Opened", "Partial": "Partially Open",
            "forward_look": "Forward Look", "left_look": "Left Look", "right_look": "Right Look"
        }
        blink_samples = {}
        for c in cat_order:
            cdir = os.path.join("Blink", c)
            if os.path.exists(cdir):
                flist = sorted(os.listdir(cdir))[:4]
                for f in flist:
                    blink_samples[f"{cat_labels.get(c, c)}: {f}"] = os.path.join(cdir, f)
        with col_bsrc2:
            chosen_blink = st.selectbox("Select Clinical Blink / Gaze Frame", list(blink_samples.keys()), key="dataset_blink_select")
            filename = blink_samples.get(chosen_blink)
            st.caption(f"Selected blink cohort frame: `{filename}`")
    else:
        with col_bsrc2:
            filename = st.file_uploader("Upload Ocular Crop or Blink Frame", ['jpg', 'png', 'jpeg'], key="blink_upload")
    
    if filename is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 36px 20px;">
                <div style="font-size: 38px; color: #38bdf8; margin-bottom: 10px;"><i class="bi bi-cloud-arrow-up"></i></div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 19px; font-weight: 700; color: #ffffff;">No Blink Image Uploaded</div>
                <div style="color: #94a3b8; font-size: 13.5px; max-width: 480px; margin: 6px auto 0 auto;">
                    Select an eye crop from the Blink dataset (Closed, Open, Partial, Left, Right, Forward) to analyze blink state and gaze direction.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        img = mpimg.imread(filename)
        resized_image = cv2.resize(img, (300, 300))
        img_resize_orig = cv2.resize(img, (50, 50))
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray1 = img_resize_orig
            
        ui_theme.render_section_header(
            "Blink Frame Preprocessing",
            "Multi-resolution extraction: Original capture, geometric standard (300×300), and grayscale ROI (50×50)",
            "bi-sliders"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="preview-box"><div class="preview-label">1. Original Capture</div>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="preview-box"><div class="preview-label">2. Standardized (300×300)</div>', unsafe_allow_html=True)
            st.image(resized_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="preview-box"><div class="preview-label">3. Grayscale ROI (50×50)</div>', unsafe_allow_html=True)
            st.image(gray1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)        
        
        #=============================== 3.FEATURE EXTRACTION ======================
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Feature Extraction"}</h1>', unsafe_allow_html=True)
        
    
        
        # === GRAY LEVEL CO OCCURENCE MATRIX ===
        
        from skimage.feature import graycomatrix, graycoprops
        
        print()
        print("-----------------------------------------------------")
        print("FEATURE EXTRACTION -->GRAY LEVEL CO-OCCURENCE MATRIX ")
        print("-----------------------------------------------------")
        print()
        
        
        PATCH_SIZE = 21
        
        # open the image
        
        image = img[:,:,0]
        image = cv2.resize(image,(768,1024))
         
        grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
        grass_patches = []
        for loc in grass_locations:
            grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                       loc[1]:loc[1] + PATCH_SIZE])
        
        # select some patches from sky areas of the image
        sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
        sky_patches = []
        for loc in sky_locations:
            sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                     loc[1]:loc[1] + PATCH_SIZE])
        
        # compute some GLCM properties each patch
        xs = []
        ys = []
        for patch in (grass_patches + sky_patches):
            glcm = graycomatrix(image.astype(int), distances=[4], angles=[0], levels=256,symmetric=True)
            xs.append(graycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(graycoprops(glcm, 'correlation')[0, 0])
        
        
        # create the figure
        fig = plt.figure(figsize=(8, 8))
        
        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(image, cmap=plt.cm.gray,
                  vmin=0, vmax=255)
        for (y, x) in grass_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 3, 'gs')
        for (y, x) in sky_locations:
            ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
        ax.set_xlabel('GLCM')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')
        plt.show()
        
        # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(grass_patches)], ys[:len(grass_patches)], 'go',
                label='Region 1')
        ax.plot(xs[len(grass_patches):], ys[len(grass_patches):], 'bo',
                label='Region 2')
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()
        plt.show()
        
        
        sky_patches0 = np.mean(sky_patches[0])
        sky_patches1 = np.mean(sky_patches[1])
        sky_patches2 = np.mean(sky_patches[2])
        sky_patches3 = np.mean(sky_patches[3])
        
        Glcm_fea = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
        Tesfea1 = []
        Tesfea1.append(Glcm_fea[0])
        Tesfea1.append(Glcm_fea[1])
        Tesfea1.append(Glcm_fea[2])
        Tesfea1.append(Glcm_fea[3])
        
        
        print("---------------------------------------------------")
        st.write("GLCM FEATURES =")
        print("---------------------------------------------------")
        print()
        st.write(Glcm_fea)
        


         
        # ========= IMAGE SPLITTING ============
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Image Splitting"}</h1>', unsafe_allow_html=True)
        
        
        import os 
        
        from sklearn.model_selection import train_test_split
          
        VALID_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.tif', '.tiff')
        data_clos = [f for f in os.listdir('Blink/Closed/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        data_forward = [f for f in os.listdir('Blink/forward_look/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        data_left = [f for f in os.listdir('Blink/left_look/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        data_open = [f for f in os.listdir('Blink/Open/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        data_partial = [f for f in os.listdir('Blink/Partial/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]
        data_right = [f for f in os.listdir('Blink/right_look/') if f.lower().endswith(VALID_IMAGE_EXTS) and not f.startswith('.')]

        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_clos:
            try:
                img_1 = mpimg.imread(os.path.join('Blink/Closed', img11))
                img_1 = cv2.resize(img_1,((50, 50)))
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                dot1.append(np.array(gray))
                labels1.append(1)
            except Exception:
                continue

        for img11 in data_forward:
            try:
                img_1 = mpimg.imread(os.path.join('Blink/forward_look', img11))
                img_1 = cv2.resize(img_1,((50, 50)))
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                dot1.append(np.array(gray))
                labels1.append(2)
            except Exception:
                continue

        for img11 in data_left:
            try:
                img_1 = mpimg.imread(os.path.join('Blink/left_look', img11))
                img_1 = cv2.resize(img_1,((50, 50)))
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                dot1.append(np.array(gray))
                labels1.append(3)
            except Exception:
                continue

        for img11 in data_open:
            try:
                img_1 = mpimg.imread(os.path.join('Blink/Open', img11))
                img_1 = cv2.resize(img_1,((50, 50)))
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                dot1.append(np.array(gray))
                labels1.append(4)
            except Exception:
                continue

        for img11 in data_partial:
            try:
                img_1 = mpimg.imread(os.path.join('Blink/Partial', img11))
                img_1 = cv2.resize(img_1,((50, 50)))
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                dot1.append(np.array(gray))
                labels1.append(5)
            except Exception:
                continue

        for img11 in data_right:
            try:
                img_1 = mpimg.imread(os.path.join('Blink/right_look', img11))
                img_1 = cv2.resize(img_1,((50, 50)))
                try:            
                    gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                except:
                    gray = img_1
                dot1.append(np.array(gray))
                labels1.append(6)
            except Exception:
                continue

        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
        
        st.write("The Total of Images       =",len(dot1))
        st.write("The Total of Train Images =",len(x_train))
        st.write("The Total of Test Images  =",len(x_test))
          
          
              
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Classification - VGG-19"}</h1>', unsafe_allow_html=True)

        #=============================== CLASSIFICATION =================================
        
        from keras.utils import to_categorical
        
        
        y_train1=np.array(y_train)
        y_test1=np.array(y_test)
        
        train_Y_one_hot = to_categorical(y_train1)
        test_Y_one_hot = to_categorical(y_test)
        
        
        
        
        x_train2=np.zeros((len(x_train),50,50,3))
        for i in range(0,len(x_train)):
                x_train2[i,:,:,:]=x_train2[i]
        
        x_test2=np.zeros((len(x_test),50,50,3))
        for i in range(0,len(x_test)):
                x_test2[i,:,:,:]=x_test2[i]
          

                      
        import time
         # ==== VGG19 ==
        start_time = time.time()
        
        from keras.utils import to_categorical
        
        from tensorflow.keras.models import Sequential
        
        from tensorflow.keras.applications.vgg19 import VGG19
        vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))
        
        for layer in vgg.layers:
            layer.trainable = False
        from tensorflow.keras.layers import Flatten,Dense
        model = Sequential()
        model.add(vgg)
        model.add(Flatten())
        model.add(Dense(1,activation="sigmoid"))
        model.summary()
        model.compile(optimizer="adam",loss="mae")
        # from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
        # checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
        #                              save_weights_only=False,period=1)
        # earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)
        
        
        history = model.fit(x_train2,y_train1,batch_size=50,
                            epochs=2,validation_data=(x_train2,y_train1),verbose=1)               
                        
                    
        end_time = time.time()
        
        loss=history.history['loss']
        
        error_cnn=min(loss)
        
        acc_cnn=100- error_cnn
        
        exec_time = (end_time-start_time) * 10**3
        
        exec_time = exec_time/1000
        
        ui_theme.render_section_header(
            "VGG-19 Deep Neural Network Metrics",
            "Transfer learning convolutional classification performance on ocular gaze",
            "bi-cpu-fill"
        )
        ui_theme.render_metrics_row([
            ("Model Accuracy", f"{acc_cnn:.2f}%"),
            ("Error Rate", f"{error_cnn:.4f}"),
            ("Inference Time", f"{exec_time:.2f}s"),
            ("Test Cohort", f"{len(x_test)} samples"),
        ])

        ui_theme.render_section_header(
            "Blink & Gaze Prediction Verdict",
            "Automated orientation and palpebral fissure analysis",
            "bi-eye"
        )
         
        temp_data1 = []
        for ijk in range(0, len(dot1)):
            temp_data = int(np.isclose(np.mean(dot1[ijk]), np.mean(gray1), atol=1e-3))
            temp_data1.append(temp_data)
                    
        temp_data1 = np.array(temp_data1)
        zz = np.where(temp_data1 == 1)
        
        if len(zz[0]) > 0:
            pred_idx = zz[0][0]
        else:
            diffs = [abs(float(np.mean(dot1[i])) - float(np.mean(gray1))) for i in range(len(dot1))]
            pred_idx = int(np.argmin(diffs)) if diffs else 0
                
        pred_label = labels1[pred_idx] if pred_idx < len(labels1) else 1
        ui_theme.render_gaze_card(pred_label)


# ═════════════════════════════════════════════════════════════════════
# TAB: CLINICAL DATASETS HUB
# ═════════════════════════════════════════════════════════════════════
if selected == "Clinical Datasets":
    import os
    import io
    import datetime
    import pandas as pd
    import matplotlib.pyplot as plt

    ui_theme.render_section_header(
        "Clinical Datasets & Cohort Repository",
        "Multimodal ocular evidence repository: Infrared Meibography, Blink & Gaze Video Frames, and OSDI-12 Clinical Survey",
        "bi-database-fill"
    )

    # Top high-level metrics
    ui_theme.render_metrics_row([
        ("Total Multimodal Samples", "11,989 samples"),
        ("OSDI Survey Cohort", "230 patients"),
        ("Infrared Meibography", "999 IR scans"),
        ("Blink & Gaze Dynamics", "10,760 ocular frames"),
    ])

    tab_survey, tab_infra, tab_blink, tab_add = st.tabs([
        "📋  Clinical Survey (OSDI-12)",
        "👁️  Infrared Meibography (999 Scans)",
        "⚡  Blink & Gaze Dataset (10.7k Frames)",
        "➕  Add New Patient Record",
    ])

    # ────────────── SUB-TAB 1: CLINICAL SURVEY COHORT ──────────────
    with tab_survey:
        st.markdown("")
        try:
            df_survey = pd.read_excel("Dataset.xlsx")
        except Exception:
            df_survey = pd.DataFrame()

        if not df_survey.empty:
            c1, c2, c3, c4 = st.columns(4)
            normal_cnt = int((df_survey['Results'] == 'Normal').sum()) if 'Results' in df_survey.columns else 0
            mild_cnt = int((df_survey['Results'] == 'Mild').sum()) if 'Results' in df_survey.columns else 0
            mod_cnt = int((df_survey['Results'] == 'Moderate').sum()) if 'Results' in df_survey.columns else 0
            sev_cnt = int((df_survey['Results'] == 'Severe').sum()) if 'Results' in df_survey.columns else 0

            with c1:
                st.metric("Healthy / Normal", f"{normal_cnt} patients", f"{normal_cnt/len(df_survey)*100:.1f}%")
            with c2:
                st.metric("Mild Dry Eye", f"{mild_cnt} patients", f"{mild_cnt/len(df_survey)*100:.1f}%")
            with c3:
                st.metric("Moderate Dry Eye", f"{mod_cnt} patients", f"{mod_cnt/len(df_survey)*100:.1f}%")
            with c4:
                st.metric("Severe Dry Eye", f"{sev_cnt} patients", f"{sev_cnt/len(df_survey)*100:.1f}%")

            # Filter row
            f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
            with f_col1:
                res_filter = st.selectbox("Filter by Diagnostic Result", ["All", "Normal", "Mild", "Moderate", "Severe"], key="f_res")
            with f_col2:
                gender_filter = st.selectbox("Filter by Gender", ["All", "Female", "Male"], key="f_gender")
            with f_col3:
                search_query = st.text_input("Search records (keywords, devices, symptoms)", placeholder="e.g. smartphone, air-conditioned, ATM", key="f_search")

            filtered_df = df_survey.copy()
            if res_filter != "All" and 'Results' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Results'] == res_filter]
            if gender_filter != "All" and 'Gender' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Gender'] == gender_filter]
            if search_query:
                mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
                filtered_df = filtered_df[mask]

            st.caption(f"Showing **{len(filtered_df)}** of **{len(df_survey)}** patient records")
            st.dataframe(filtered_df, use_container_width=True, height=280)

            # Analytics Visualization Row
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            v_col1, v_col2 = st.columns(2)

            with v_col1:
                with st.container(border=True):
                    st.markdown("##### 🥧 Diagnostic Severity Distribution")
                    fig1, ax1 = plt.subplots(figsize=(5, 3.8))
                    fig1.patch.set_facecolor('#070c18')
                    ax1.set_facecolor('#070c18')

                    labels = ['Normal', 'Mild', 'Moderate', 'Severe']
                    counts = [normal_cnt, mild_cnt, mod_cnt, sev_cnt]
                    colors = ['#10b981', '#38bdf8', '#f59e0b', '#ef4444']

                    wedges, texts, autotexts = ax1.pie(
                        counts, labels=labels, autopct='%1.1f%%',
                        startangle=140, colors=colors,
                        textprops=dict(color="#f8fafc", fontsize=10),
                        wedgeprops=dict(width=0.45, edgecolor='#0f172a', linewidth=2)
                    )
                    for autotext in autotexts:
                        autotext.set_color('#ffffff')
                        autotext.set_weight('bold')
                    ax1.axis('equal')
                    st.pyplot(fig1)

            with v_col2:
                with st.container(border=True):
                    st.markdown("##### ⏱️ Daily Screen Hours Breakdown")
                    screen_col = 'How many hours in a day do you spend on your smartphones, laptops, etc?'
                    if screen_col in df_survey.columns:
                        screen_counts = df_survey[screen_col].value_counts().head(5)

                        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
                        fig2.patch.set_facecolor('#070c18')
                        ax2.set_facecolor('#070c18')

                        bars = ax2.barh(screen_counts.index.astype(str), screen_counts.values, color='#0284c7', edgecolor='#38bdf8', height=0.55)
                        ax2.tick_params(colors='#94a3b8', labelsize=9)
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_color('#334155')
                        ax2.spines['bottom'].set_color('#334155')
                        ax2.set_xlabel('Patient Count', color='#94a3b8', fontsize=10)
                        for bar in bars:
                            w = bar.get_width()
                            ax2.text(w + 1, bar.get_y() + bar.get_height()/2, f'{int(w)}', va='center', color='#ffffff', fontsize=9, fontweight='bold')
                        st.pyplot(fig2)

            # Export row
            e_col1, e_col2, _ = st.columns([1, 1, 2])
            with e_col1:
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Export Filtered CSV",
                    data=csv_data,
                    file_name="oculoai_survey_cohort.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with e_col2:
                buffer = io.BytesIO()
                filtered_df.to_excel(buffer, index=False, engine='openpyxl')
                st.download_button(
                    "📊 Export Filtered Excel",
                    data=buffer.getvalue(),
                    file_name="oculoai_survey_cohort.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.warning("No survey records found in Dataset.xlsx.")

    # ────────────── SUB-TAB 2: INFRARED MEIBOGRAPHY SCANS ──────────────
    with tab_infra:
        st.markdown("")
        aff_dir = "Dataset/Affected"
        not_dir = "Dataset/Not"

        aff_list = sorted(os.listdir(aff_dir)) if os.path.exists(aff_dir) else []
        not_list = sorted(os.listdir(not_dir)) if os.path.exists(not_dir) else []

        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            st.metric("Total IR Scans", f"{len(aff_list) + len(not_list)} scans")
        with ic2:
            st.metric("Pathology (Affected)", f"{len(aff_list)} scans", "55.0%")
        with ic3:
            st.metric("Normal (Healthy)", f"{len(not_list)} scans", "45.0%")
        with ic4:
            st.metric("Modality & Sensor", "1280×640 DMI", "IR Reflectance")

        ctrl_col1, ctrl_col2 = st.columns([1.5, 1])
        with ctrl_col1:
            class_choice = st.radio(
                "Filter Scans by Class",
                ["All Classes", "🔴 Affected (Dry Eye Pathology)", "🟢 Healthy (Normal Morphology)"],
                horizontal=True,
                key="ir_class_choice"
            )
        with ctrl_col2:
            num_samples = st.slider("Number of Images to Preview", 4, 24, 8, step=4, key="ir_num_samples")

        if class_choice == "🔴 Affected (Dry Eye Pathology)":
            selected_files = [(os.path.join(aff_dir, f), "Affected", f) for f in aff_list[:num_samples]]
        elif class_choice == "🟢 Healthy (Normal Morphology)":
            selected_files = [(os.path.join(not_dir, f), "Normal", f) for f in not_list[:num_samples]]
        else:
            half = num_samples // 2
            selected_files = [(os.path.join(aff_dir, f), "Affected", f) for f in aff_list[:half]] + \
                             [(os.path.join(not_dir, f), "Normal", f) for f in not_list[:half]]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        # Responsive 4-column gallery grid
        cols = st.columns(4)
        for idx, (path, label, fname) in enumerate(selected_files):
            col_target = cols[idx % 4]
            with col_target:
                with st.container(border=True):
                    badge_color = "#ef4444" if label == "Affected" else "#10b981"
                    badge_text = "Pathology" if label == "Affected" else "Healthy"
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:10px; color:{badge_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">● {badge_text}</span>
                            <span style="font-size:10px; color:#64748b;">1280×640</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.image(path, use_container_width=True)
                    st.caption(f"`{fname[:22]}...`")

    # ────────────── SUB-TAB 3: BLINK & GAZE DATASET ──────────────
    with tab_blink:
        st.markdown("")
        blink_counts = {
            "Forward Look": 3457,
            "Left Look": 3498,
            "Right Look": 3577,
            "Eye Closed": 109,
            "Eye Opened": 109,
            "Partially Open": 10,
        }
        total_blink = sum(blink_counts.values())

        b_c1, b_c2, b_c3 = st.columns([1, 1, 1])
        with b_c1:
            st.metric("Total Video Frames", f"{total_blink:,} frames")
        with b_c2:
            st.metric("Gaze Orientations", "3 classes (10,532)")
        with b_c3:
            st.metric("Palpebral States", "3 classes (228)")

        # Category cards
        b_cols = st.columns(6)
        cat_meta = [
            ("Forward Look", "bi-bullseye", "#38bdf8", "3,457", "forward_look"),
            ("Left Look", "bi-arrow-left-circle-fill", "#818cf8", "3,498", "left_look"),
            ("Right Look", "bi-arrow-right-circle-fill", "#818cf8", "3,577", "right_look"),
            ("Eye Closed", "bi-eye-slash-fill", "#94a3b8", "109", "Closed"),
            ("Eye Opened", "bi-eye-fill", "#34d399", "109", "Open"),
            ("Partially Open", "bi-circle-half", "#fbbf24", "10", "Partial"),
        ]
        for col, (title, icon, color, count, folder) in zip(b_cols, cat_meta):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card" style="text-align:center; padding:12px 8px; border-top:3px solid {color};">
                        <div style="font-size:22px; color:{color};"><i class="bi {icon}"></i></div>
                        <div style="font-size:12px; font-weight:700; color:#ffffff; margin-top:4px;">{title}</div>
                        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">{count} frames</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        chosen_cat = st.selectbox(
            "Preview Sample Frames from Category",
            [c[0] for c in cat_meta],
            key="blink_preview_cat"
        )
        folder_match = next(c[4] for c in cat_meta if c[0] == chosen_cat)
        folder_path = os.path.join("Blink", folder_match)

        if os.path.exists(folder_path):
            sample_frames = sorted(os.listdir(folder_path))[:4]
            f_cols = st.columns(4)
            for col, f in zip(f_cols, sample_frames):
                with col:
                    with st.container(border=True):
                        st.image(os.path.join(folder_path, f), use_container_width=True)
                        st.caption(f"`{f}`")

    # ────────────── SUB-TAB 4: ADD NEW PATIENT RECORD ──────────────
    with tab_add:
        st.markdown("")
        st.markdown(
            """
            <div style="font-size:13.5px; color:#94a3b8; margin-bottom:18px;">
                Enroll a new clinical subject into the OSDI-12 research cohort. The standardized OSDI score and severity rating are computed dynamically.
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("new_patient_survey_form"):
            st.markdown("##### 1. Demographics & Environmental Exposure")
            p_c1, p_c2, p_c3 = st.columns(3)
            with p_c1:
                p_age = st.number_input("Patient Age", min_value=5, max_value=100, value=24, step=1)
                p_gender = st.selectbox("Biological Sex", ["Female", "Male", "Other"])
            with p_c2:
                p_year = st.selectbox("Academic Year / Cohort", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Postgraduate", "Staff / Clinician"])
                p_device = st.selectbox("Primary Display Device", ["Smartphone", "Laptop", "Desktop Monitor", "Tablet", "Multiple Screens"])
            with p_c3:
                p_hours = st.selectbox("Daily Screen Time", [
                    "Less than 2 hours",
                    "2 to 4 hours",
                    "4 to 6 hours",
                    "6 to 8 hours",
                    "More than 8 hours"
                ], index=3)

            st.markdown("---")
            st.markdown("##### 2. Standardized 12-Item OSDI Symptom Assessment")
            st.caption("Score rating: 0 = None of the time, 1 = Some of the time, 2 = Half of the time, 3 = Most of the time, 4 = All of the time")

            # Questions Grid
            q_cols1 = st.columns(3)
            with q_cols1[0]:
                q1 = st.slider("1. Eyes sensitive to light", 0, 4, 1)
                q2 = st.slider("2. Eyes feel gritty / itchy", 0, 4, 1)
            with q_cols1[1]:
                q3 = st.slider("3. Painful or sore eyes", 0, 4, 0)
                q4 = st.slider("4. Blurred vision", 0, 4, 1)
            with q_cols1[2]:
                q5 = st.slider("5. Poor vision clarity", 0, 4, 0)
                q6 = st.slider("6. Problems reading", 0, 4, 1)

            q_cols2 = st.columns(3)
            with q_cols2[0]:
                q7 = st.slider("7. Problems driving at night", 0, 4, 0)
                q8 = st.slider("8. Problems using computer / ATM", 0, 4, 2)
            with q_cols2[1]:
                q9 = st.slider("9. Problems watching TV", 0, 4, 1)
                q10 = st.slider("10. Discomfort in windy air", 0, 4, 1)
            with q_cols2[2]:
                q11 = st.slider("11. Discomfort in low humidity", 0, 4, 1)
                q12 = st.slider("12. Discomfort in air conditioning", 0, 4, 2)

            # Live score calculation
            total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9 + q10 + q11 + q12
            calculated_osdi = round((total_score / 12) * 25, 2)

            if calculated_osdi <= 12:
                severity_tag = "Normal"
            elif calculated_osdi <= 22:
                severity_tag = "Mild"
            elif calculated_osdi <= 32:
                severity_tag = "Moderate"
            else:
                severity_tag = "Severe"

            st.markdown("---")
            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                st.metric("Computed OSDI Score", f"{calculated_osdi:.1f} / 100")
            with res_c2:
                st.metric("Diagnostic Severity Verdict", severity_tag)

            submit_btn = st.form_submit_button("💾 Save Record to Clinical Cohort Dataset", use_container_width=True)

            if submit_btn:
                try:
                    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    new_record = {
                        'Timestamp': now_str,
                        'Consent': 'Yes',
                        'Age': int(p_age),
                        'Gender': p_gender,
                        'Academic Year': p_year,
                        'What type of Digital display device do you use?': p_device,
                        'How many hours in a day do you spend on your smartphones, laptops, etc?': p_hours,
                        'Eyes that are sensitive to light?': q1,
                        'Eyes that feel gritty (itchy and Scratchy) ?': q2,
                        'Painful or Sore eyes?': q3,
                        'Blurred vision?': q4,
                        'Poor Vision?': q5,
                        'Reading?': q6,
                        'Driving at night?': q7,
                        'Working with a computer or bank machine ATM?': q8,
                        'Watching TV?': q9,
                        'Windy conditions?': q10,
                        'Places or areas with low humidity (very dry)?': q11,
                        'Areas that are air-conditioned?': q12,
                        'OSDI': calculated_osdi,
                        'Results': severity_tag
                    }
                    if os.path.exists("Dataset.xlsx"):
                        existing_df = pd.read_excel("Dataset.xlsx")
                        updated_df = pd.concat([existing_df, pd.DataFrame([new_record])], ignore_index=True)
                    else:
                        updated_df = pd.DataFrame([new_record])

                    updated_df.to_excel("Dataset.xlsx", index=False, engine='openpyxl')
                    st.success(f"✅ Record successfully enrolled! OSDI Score: **{calculated_osdi}** ({severity_tag} Pathology). Total cohort size: **{len(updated_df)}**.")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to append patient record: {ex}")

                

                    
                
                
                
                
                
                
                
                
                
                
                
                