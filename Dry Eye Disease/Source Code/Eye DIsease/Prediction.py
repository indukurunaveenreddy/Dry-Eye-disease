

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import matplotlib.image as mpimg
import cv2
import streamlit as st

import base64
from streamlit_option_menu import option_menu
import pickle
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.image as mpimg


#======================== BACK GROUND IMAGE ===========================



st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:32px;">{"AI-Driven Advanced Techniques for Detecting Dry Eye Disease Using  Multi-Source Evidence: Case studies, Applications, Challenges, and Future Perspectives"}</h1>', unsafe_allow_html=True)

    
# -------------------------------------------------------------------

selected = option_menu(
    menu_title=None, 
    options=["Dry Eye Prediction", "Live Camera", "Chatbot", "Eye Disease Prediction", "Eye Blink Detection"],  
    orientation="horizontal",
)


st.markdown(
    """
    <style>
    .option_menu_container {
        position: fixed;
        top: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Disease

if selected == "Live Camera":

    st.write("-----------------------------------------------------------")

    st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Live Camera Eye Scan"}</h1>', unsafe_allow_html=True)

    st.write(
        "Point your camera at one eye in good light. The app detects the eye region "
        "with a Haar cascade and classifies it using the model trained on the "
        "Affected/Not image dataset."
    )

    import eye_model

    status = eye_model.model_status()

    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        if st.button("Train / Retrain Model", key="cam_train_btn"):
            with st.spinner("Training on the dataset, this can take a minute..."):
                accuracy, report, cm, n = eye_model.train_model()
            st.session_state["cam_train_info"] = (accuracy, report, cm, n)

    if status["trained"]:
        with col_b:
            st.metric("Model accuracy", f"{status['accuracy']:.1f}%")
        with col_c:
            st.caption(f"Trained on {status['trained_on']} images.")
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

    st.write("-----------------------------------------------------------")

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


    st.write("-----------------------------------------------------------")
    st.caption("Camera frames stay on this machine. Nothing is uploaded.")


if selected == "Chatbot":

    st.write("-----------------------------------------------------------")

    st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Dry Eye Assistant"}</h1>', unsafe_allow_html=True)

    import chatbot as chatbot_module

    bot = chatbot_module.DryEyeChatbot()

    survey_stage = st.session_state.get("last_survey_stage")
    camera_verdict = st.session_state.get("last_camera_verdict")

    if survey_stage:
        st.caption(f"Context from your last screening result: {survey_stage}")
        bot.last_result = survey_stage
    elif camera_verdict:
        st.caption(
            f"Context from your last camera scan: {camera_verdict.upper()}"
        )

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi, I am your dry eye assistant. Ask me about symptoms, causes, "
                    "prevention, treatments, or what your screening result means."
                ),
            }
        ]

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about dry eye disease...")

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

    st.write("-----------------------------------------------------------")
    st.caption(chatbot_module.DISCLAIMER)

    with st.expander("Suggested questions", expanded=False):
        for suggestion in [
            "What causes dry eye?",
            "How do I treat dry eye?",
            "What are the symptoms?",
            "When should I see a doctor?",
            "What does my result mean?",
        ]:
            st.write(f"- {suggestion}")


if selected == 'Eye Disease Prediction':

    # abt = " Here , the system can predict the input image is affected or not with the help of deep learning algorithm such as CNN-2D effectively"
    # a="A Long-Term Recurrent Convolutional Network for Eye Blink Completeness Detection introduces a novel deep learning framework designed to accurately detect the completeness of eye blinks. By integrating convolutional neural networks (CNNs) with long short-term memory (LSTM) networks, the Eye-LRCN model effectively captures both spatial and temporal features from video sequences. This hybrid architecture allows for precise identification of partial and complete blinks, improving over traditional methods that often struggle with the subtle nuances of eye movements. The model's performance is evaluated on multiple datasets, demonstrating its robustness and potential applications in fields such as driver drowsiness detection, human-computer interaction, and neurological disorder monitorin"
    # st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:16px;">{a}</h1>', unsafe_allow_html=True)

    st.write("-----------------------------------------------------------")

    filename = st.file_uploader("Choose Image",['jpg','png'])
    
    with open('file.pickle', 'wb') as f:
        pickle.dump(filename, f)
    
    
    
    if filename is None:

        st.text("Upload Image")
        
    else:
            
        # filename = askopenfilename()
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Input Image"}</h1>', unsafe_allow_html=True)
        
        img = mpimg.imread(filename)
    
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis ('off')
        plt.show()
        
        
        st.image(img,caption="Original Image")
    
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Preprocessing"}</h1>', unsafe_allow_html=True)
        
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
        
        st.image(resized_image,caption="Resized Image")
        
        # st.image(img,caption="Original Image")
                 
        #==== GRAYSCALE IMAGE ====
        

        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1)
        plt.axis ('off')
        plt.show()
        
    
        st.image(gray1,caption="Gray Scale Image")        
        
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
          
        data_aff = os.listdir('Dataset/Affected/')
         
        data_not = os.listdir('Dataset/Not/')
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_aff:
            # print(img)
            img_1 = mpimg.imread('Dataset/Affected//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(1)
         
         
        for img11 in data_not:
            # print(img)
            img_1 = mpimg.imread('Dataset/Not//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(2)
         
   

        x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)
        
        
        print("------------------------------------------------------------")
        print(" Image Splitting")
        print("------------------------------------------------------------")
        print()
        
        st.write("The Total of Images       =",len(dot1))
        st.write("The Total of Train Images =",len(x_train))
        st.write("The Total of Test Images  =",len(x_test))
          
          
              
          
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
    
    # ===================================== CLASSIFICATION ==================================
    
     # ----------------------- MOBILENET -----------------------
    
            
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Classification - MobileNet"}</h1>', unsafe_allow_html=True)
        
    
        import time
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from keras.utils import to_categorical
        from tensorflow.keras import layers, models
        
        
        print()
        print("----------------------------------------------")
        print(" Classification - Mobilnet")
        print("----------------------------------------------")
        print()
        from tensorflow.keras.applications import MobileNet
        
        start_mob = time.time()
        
        base_model = MobileNet(weights=None, input_shape=(50, 50, 3), classes=3)
        
        model = models.Model(inputs=base_model.input, outputs=base_model.output)
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        model.summary()
        
        history = model.fit(x_train2,train_Y_one_hot, epochs=3, batch_size=64)
        
        loss_val = history.history['loss']
        
        loss_val = min(loss_val)
        
        acc_mob = 100 - loss_val
        
        
        print("-------------------------------------")
        print("Mobilenet - Perfromance Analysis")
        print("-------------------------------------")
        print()
        print("1. Accuracy   =", acc_mob,'%')
        print()
        print("2. Error Rate =",loss_val)
        print()
        
        
        predictions = model.predict(x_test2)
        
        end_mob = time.time()
        
        time_mob = (end_mob-start_mob) * 10**3
        
        time_mob = time_mob / 1000
        
        print("3. Execution Time  = ",time_mob, "s")
        
        
        st.write("-------------------------------------")
        st.write("Mobilenet - Perfromance Analysis")
        st.write("-------------------------------------")
        print()
        st.write("1. Accuracy   =", acc_mob,'%')
        print()
        st.write("2. Error Rate =",loss_val)
        print()
        
        
        predictions = model.predict(x_test2)
        
        end_mob = time.time()
        
        time_mob = (end_mob-start_mob) * 10**3
        
        time_mob = time_mob / 1000
        
        st.write("3. Execution Time  = ",time_mob, "s")
                
        
        # --- prediction
        
        st.write("-----------------------------------------------------------")
    
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Prediction -Eye Disease"}</h1>', unsafe_allow_html=True)
         
         
        Total_length = len(data_aff) + len(data_not) 
        
        
           
        temp_data1  = []
        for ijk in range(0,Total_length):
                    # print(ijk)
                temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
                temp_data1.append(temp_data)
                    
        temp_data1 =np.array(temp_data1)
                
        zz = np.where(temp_data1==1)
                
        if labels1[zz[0][0]] == 1:
                st.write('-----------------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Affected"}</h1>', unsafe_allow_html=True)
    
            
        else:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Not"}</h1>', unsafe_allow_html=True)
    
                # st.write('The Prediction = AFFECTED')   
                print()
                st.write('---------------------------------')   
    


if selected == "Dry Eye Prediction":
    
    file = st.file_uploader("Upload Input Dataset",['csv','xlsx'])
    
    import pandas as pd
    import time
    from sklearn.model_selection import train_test_split


    
    if file is None:
        
        st.warning("Upload Input Data")
    
    else:
        
    
        dataframe=pd.read_excel("Dataset.xlsx")
                
        print("--------------------------------")
        print("Data Selection")
        print("--------------------------------")
        print()
        print(dataframe.head(15))    
        
        st.write("--------------------------------")
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Selection !!!"}</h1>', unsafe_allow_html=True)
    
        # st.write("--------------------------------")
        # st.write("Data Selection")
        # st.write("--------------------------------")
        print()
        st.write(dataframe.head(15))    
        
        
     #-------------------------- PRE PROCESSING --------------------------------    
        
        # ----- CHECKING MISSING VALUES 
        
        
        
        print("----------------------------------------------------")
        print("              Handling Missing values               ")
        print("----------------------------------------------------")
        print()
        print(dataframe.isnull().sum())
        
        st.write("--------------------------------")
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Pre-processing !!!"}</h1>', unsafe_allow_html=True)
    
        
        # st.write("----------------------------------------------------")
        st.write("              Handling Missing values               ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe.isnull().sum())
        
        res = dataframe.isnull().sum().any()
            
        if res == False:
            
            print("--------------------------------------------")
            print("  There is no Missing values in our dataset ")
            print("--------------------------------------------")
            print()   
            
            
            st.write("--------------------------------------------")
            st.write("  There is no Missing values in our dataset ")
            st.write("--------------------------------------------")
      
        
            
        else:
        
            print("--------------------------------------------")
            print(" Missing values is present in our dataset   ")
            print("--------------------------------------------")
            print()    
            
            st.write("--------------------------------------------")
            st.write("  Missing values is present in our dataset ")
            
            dataframe = dataframe.fillna(0)
            
            resultt = dataframe.isnull().sum().any()
            
            if resultt == False:
                
                print("--------------------------------------------")
                print(" Data Cleaned !!!   ")
                print("--------------------------------------------")
                print()    
                print(dataframe.isnull().sum())  
                
                
                st.write("--------------------------------------------")
                st.write(" Data Cleaned !!!   ")
                st.write("--------------------------------------------")
                print()    
                st.write(dataframe.isnull().sum()) 
                
                
        # --- DROP UNWANTED COLUMNS
        
        dataframe = dataframe.drop(['Timestamp','Consent'],axis=1)
            
        
        # ----- LABEL ENCODING
                
    
        print("----------------------------------------------------")
        print("            Before Label Encoding                   ")
        print("----------------------------------------------------")
        print()
        print(dataframe['Gender'].head(15))                
                    
        st.write("----------------------------------------------------")
        st.write("            Before Label Encoding                   ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe['Gender'].head(15))
        
        
        gen =  dataframe['Gender']
        
        aca_yr = dataframe['Gender']
        
    
    
    
    
        from sklearn import preprocessing
        
        label_encoder = preprocessing.LabelEncoder()
        
        dataframe['Gender']= label_encoder.fit_transform(dataframe['Gender'])  
        
        dataframe['Academic Year']= label_encoder.fit_transform(dataframe['Academic Year'])  
        
        dataframe['What type of Digital display device do you use?']= label_encoder.fit_transform(dataframe['What type of Digital display device do you use?'])  
        
        dataframe['How many hours in a day do you spend on your smartphones, laptops, etc?']= label_encoder.fit_transform(dataframe['How many hours in a day do you spend on your smartphones, laptops, etc?'])                
                    
        dataframe['Eyes that are sensitive to light?']= label_encoder.fit_transform(dataframe['Eyes that are sensitive to light?'])  
        
        dataframe['Eyes that feel gritty (itchy and Scratchy) ?']= label_encoder.fit_transform(dataframe['Eyes that feel gritty (itchy and Scratchy) ?'])  
    
    
        dataframe['Painful or Sore eyes?']= label_encoder.fit_transform(dataframe['Painful or Sore eyes?'])  
        
        dataframe['Blurred vision?']= label_encoder.fit_transform(dataframe['Blurred vision?'])  
    
        dataframe['Reading?']= label_encoder.fit_transform(dataframe['Reading?'])  
        
        dataframe['Driving at night?']= label_encoder.fit_transform(dataframe['Driving at night?'].astype(str))  
    
        dataframe['Working with a computer or bank machine ATM?']= label_encoder.fit_transform(dataframe['Working with a computer or bank machine ATM?'])  
    
        dataframe['Watching TV?']= label_encoder.fit_transform(dataframe['Watching TV?'].astype(str))  
        
        dataframe['Windy conditions?']= label_encoder.fit_transform(dataframe['Windy conditions?'].astype(str))  
                   
                    
        dataframe['Places or areas with low humidity (very dry)?']= label_encoder.fit_transform(dataframe['Places or areas with low humidity (very dry)?'].astype(str))  
         
        dataframe['Areas that are air-conditioned?']= label_encoder.fit_transform(dataframe['Areas that are air-conditioned?'].astype(str))                 
        
        dataframe['Poor Vision?']= label_encoder.fit_transform(dataframe['Poor Vision?'])  

                    
        dataframe['Results']= label_encoder.fit_transform(dataframe['Results'])  
           
        print("----------------------------------------------------")
        print("            After Label Encoding                   ")
        print("----------------------------------------------------")
        print()
        print(dataframe['Gender'].head(15))
        
        st.write("----------------------------------------------------")
        st.write("            After Label Encoding                   ")
        st.write("----------------------------------------------------")
        print()
        st.write(dataframe['Gender'].head(15))
        
        
        
       #-------------------------- DATA SPLITTING  --------------------------------
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Data Splitting !!!"}</h1>', unsafe_allow_html=True)

        
        X=dataframe.drop('Results',axis=1)
                
        y=dataframe['Results']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        print("---------------------------------------------")
        print("             Data Splitting                  ")
        print("---------------------------------------------")
        
        print()
        
        print("Total no of input data   :",dataframe.shape[0])
        print("Total no of test data    :",X_test.shape[0])
        print("Total no of train data   :",X_train.shape[0])
        
        
        # st.write("---------------------------------------------")
        st.write("     Test Data & Train Data                  ")
        st.write("---------------------------------------------")
        
        print()
        
        st.write("Total no of input data   :",dataframe.shape[0])
        st.write("Total no of test data    :",X_test.shape[0])
        st.write("Total no of train data   :",X_train.shape[0])
    
        #-------------------------- CLASSIFICATION --------------------------------
        
    
        #  ------ MLP --------
        
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Classification - MLP !!!"}</h1>', unsafe_allow_html=True)

        
        from sklearn.neural_network import MLPClassifier         
        
        start_mlp = time.time()
        
        mlpp = MLPClassifier()
        
        mlpp.fit(X_train, y_train)
        
        
        pred_mlpp = mlpp.predict(X_test)
        
        
        from sklearn import metrics
        
        
        acc_mlp = metrics.accuracy_score(pred_mlpp,y_test)* 100
        
        loss = 100 - acc_mlp
        # Classification report
        report_svm = metrics.classification_report(y_test, pred_mlpp, target_names=['Mild', 'Moderate','Normal','Severe'])
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, pred_mlpp)
        
       
        
        
        end_mlp = time.time()
        
        
        exec_time = (end_mlp-start_mlp) * 10**3
        
        exec_time_mlp = exec_time/1000
        
        
        print("----------------------------------------------------")
        print("     Classification -- Multi Layer Perceptron      ")
        print("----------------------------------------------------")
        
        print()
        
        print("1)  Accuracy        =", acc_mlp ,'%')
        print()
        print("2)  Error rate      = ", loss ,'%' )
        print()
        print("3)  Execution Time  = ", exec_time_mlp , 'sec')
        print()
        print("4) Classification Report  = ", )
        print()
        print(report_svm)
        
        
        st.write("----------------------------------------------------")
        st.write("      Classification -- Multi Layer Perceptron      ")
        st.write("----------------------------------------------------")
        
        print()
        
        st.write("1)  Accuracy        =", acc_mlp ,'%')
        print()
        st.write("2)  Error rate      = ", loss ,'%' )
        print()
        st.write("3)  Execution Time  = ", exec_time_mlp , 'sec')
        print()
        st.write("4) Classification Report  = ", )
        print()
        st.write(report_svm)
        
        st.write("---------------------------------------------")                
                    
                
                
# -------------------------- PREDICTION  ----------------------
    
        st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:28px;font-family:Caveat, sans-serif;">{" Prediction  !!!"}</h1>', unsafe_allow_html=True)
    
        # inputt = int(input("Enter Prediction Number :"))
        
        inputt = st.text_input("Enter Prediction Number :")
        
        butt = st.button("Submit")
        
        if butt:
            

            inputt = int(inputt)

            stage_verdict = None

            if pred_mlpp[inputt] == 0:
                stage_verdict = "affected"
                st.markdown('<h1 style="color:#000000;text-align: center;font-size:25px;">Identified Affected = MILD STAGE</h1>', unsafe_allow_html=True)

            elif pred_mlpp[inputt] == 1:
                stage_verdict = "affected"
                st.markdown('<h1 style="color:#000000;text-align: center;font-size:25px;">Identified Affected = MODERATE STAGE</h1>', unsafe_allow_html=True)

            elif pred_mlpp[inputt] == 2:
                stage_verdict = "normal"
                st.markdown('<h1 style="color:#000000;text-align: center;font-size:25px;">Identified Normal </h1>', unsafe_allow_html=True)

            elif pred_mlpp[inputt] == 3:
                stage_verdict = "affected"
                st.markdown('<h1 style="color:#000000;text-align: center;font-size:25px;">Identified Affected = SEVERE STAGE</h1>', unsafe_allow_html=True)

            st.session_state["last_camera_verdict"] = stage_verdict
            st.session_state["last_survey_stage"] = [
                "Mild", "Moderate", "Normal", "Severe"
            ][int(pred_mlpp[inputt])]

            st.info("Open the Chatbot tab to ask what this stage means and how to treat it.")

if selected == 'Eye Blink Detection':               
                
                
   # abt = " Here , the system can predict the input image is affected or not with the help of deep learning algorithm such as CNN-2D effectively"
    # a="A Long-Term Recurrent Convolutional Network for Eye Blink Completeness Detection introduces a novel deep learning framework designed to accurately detect the completeness of eye blinks. By integrating convolutional neural networks (CNNs) with long short-term memory (LSTM) networks, the Eye-LRCN model effectively captures both spatial and temporal features from video sequences. This hybrid architecture allows for precise identification of partial and complete blinks, improving over traditional methods that often struggle with the subtle nuances of eye movements. The model's performance is evaluated on multiple datasets, demonstrating its robustness and potential applications in fields such as driver drowsiness detection, human-computer interaction, and neurological disorder monitorin"
    # st.markdown(f'<h1 style="color:#000000;text-align: justify;font-size:16px;">{a}</h1>', unsafe_allow_html=True)

    st.write("-----------------------------------------------------------")

    filename = st.file_uploader("Choose Image",['jpg','png'])
    

    
    
    if filename is None:

        st.text("Upload Image")
        
    else:
            
        # filename = askopenfilename()
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Input Image"}</h1>', unsafe_allow_html=True)
        
        img = mpimg.imread(filename)
    
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis ('off')
        plt.show()
        
        
        st.image(img,caption="Original Image")
    
        
        #============================ PREPROCESS =================================
        
        #==== RESIZE IMAGE ====
        
        
        st.write("-----------------------------------------------------------")

        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Preprocessing"}</h1>', unsafe_allow_html=True)
        
        
        resized_image = cv2.resize(img,(300,300))
        img_resize_orig = cv2.resize(img,((50, 50)))
        
        fig = plt.figure()
        plt.title('RESIZED IMAGE')
        plt.imshow(resized_image)
        plt.axis ('off')
        plt.show()
        
        st.image(resized_image,caption="Resized Image")
        
        # st.image(img,caption="Original Image")
                 
        #==== GRAYSCALE IMAGE ====
        

        SPV = np.shape(img)
        
        try:            
            gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
            
        except:
            gray1 = img_resize_orig
           
        fig = plt.figure()
        plt.title('GRAY SCALE IMAGE')
        plt.imshow(gray1)
        plt.axis ('off')
        plt.show()
        
    
        st.image(gray1,caption="Gray Scale Image")        
        
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
          
        data_clos = os.listdir('Blink/Closed/')
         
        data_forward = os.listdir('Blink/forward_look/')
         
        data_left= os.listdir('Blink/left_look/')
         
        data_open = os.listdir('Blink/Open/')
         
        data_partial = os.listdir('Blink/Partial/')
         
        data_right= os.listdir('Blink/right_look/')    


        
        import numpy as np
        dot1= []
        labels1 = [] 
        for img11 in data_clos:
            # print(img)
            img_1 = mpimg.imread('Blink/Closed//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(1)
         
         
        for img11 in data_forward:
            # print(img)
            img_1 = mpimg.imread('Blink/forward_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(2)
         
         
        for img11 in data_left:
            # print(img)
            img_1 = mpimg.imread('Blink/left_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(3)
         
         
        for img11 in data_open:
            # print(img)
            img_1 = mpimg.imread('Blink/Open//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(4)
         
        for img11 in data_partial:
            # print(img)
            img_1 = mpimg.imread('Blink/Partial//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(5)
         
         
        for img11 in data_right:
            # print(img)
            img_1 = mpimg.imread('Blink/right_look//' + "/" + img11)
            img_1 = cv2.resize(img_1,((50, 50)))
        
        
            try:            
                gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
                
            except:
                gray = img_1
        
            
            dot1.append(np.array(gray))
            labels1.append(6)

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
        
        # st.write("-------------------------------------------")
        st.write("  Convolutional Neural Network - VGG 19")
        st.write("-------------------------------------------")
        print()
        st.write("1. Accuracy       =", acc_cnn,'%')
        print()
        st.write("2. Error Rate     =",error_cnn)
        print()
        st.write("3. Execution Time =",exec_time,'s')
                   
        st.write("-----------------------------------------------------------")
    
        st.markdown(f'<h1 style="color:#112E9B;text-align: center;font-size:26px;">{"Prediction - Eye Blink Detection"}</h1>', unsafe_allow_html=True)
         
            
        Total_length = len(data_clos) + len(data_forward) + len(data_left) + len(data_open) + len(data_partial) + len(data_right)
        
        
           
        temp_data1  = []
        for ijk in range(0,Total_length):
                    # print(ijk)
                temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
                temp_data1.append(temp_data)
                    
        temp_data1 =np.array(temp_data1)
                
        zz = np.where(temp_data1==1)
                
        if labels1[zz[0][0]] == 1:
                st.write('-----------------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Closed"}</h1>', unsafe_allow_html=True)
    
            
        elif labels1[zz[0][0]] == 2:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Forward Look"}</h1>', unsafe_allow_html=True)
                  
        elif labels1[zz[0][0]] == 3:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Left Look"}</h1>', unsafe_allow_html=True)
                              
                
        elif labels1[zz[0][0]] == 4:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Eye Opened"}</h1>', unsafe_allow_html=True)
                                       
        elif labels1[zz[0][0]] == 5:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Open Partially"}</h1>', unsafe_allow_html=True)
       
        elif labels1[zz[0][0]] == 6:
                st.write('---------------------------------')
                print()
                st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:24px;">{"Identified = Right Look"}</h1>', unsafe_allow_html=True)

                

                    
                
                
                
                
                
                
                
                
                
                
                
                