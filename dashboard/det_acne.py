import streamlit as st

def display():
    from keras.models import load_model
    from keras.preprocessing import image
    from keras.layers import Input, TFSMLayer
    from keras.models import Model
    from PIL import Image, ImageOps
    import numpy as np

    # Function to preprocess image for prediction
    def preprocess_image(img_path, target_size=(224, 224)):
        try:
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(img)
            normalized_img = (img_array.astype(np.float32) / 127.5) - 1
            data = np.expand_dims(normalized_img, axis=0)
            return data, img
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None, None

    # Fungsi untuk prediksi tipe jerawat
    def predict_acne_type(img_path, acne_type_model, labels):
        try:
            data, _ = preprocess_image(img_path)
            prediction = acne_type_model.predict(data, verbose=0)

            if isinstance(prediction, dict):
                prediction = prediction['sequential_7']

            index = np.argmax(prediction[0])  # Asumsi prediksi batch pertama
            acne_type = labels[index]
            return acne_type
        except Exception as e:
            st.error(f"Error during acne type prediction: {e}")
            return None

    # Fungsi untuk rekomendasi pengobatan
    def acne_treatment(acne_type):
        treatments = {
            "White Comedo": "Salicylic acid, retinoids, clay masks, and gentle exfoliation.",
            "Nodule": "Oral medications, isotretinoin, and traditional remedies like green tea.",
            "Pustule": "Topical salicylic acid, honey, and clay masks.",
            "Papule": "Benzoyl peroxide, tea tree oil, and aloe vera.",
            "Black Comedo": "Chemical exfoliants, clay masks, and gentle scrubs.",
        }
        return treatments.get(acne_type, "No specific treatment available.")    

    # Function to predict acne severity
    def predict_acne_severity(img_path, acne_severity_model, labels):
        try:
            data, img = preprocess_image(img_path)
            prediction = acne_severity_model.predict(data, verbose=0)
            
            if isinstance(prediction, dict):
                prediction = prediction['sequential_11']
            
            index = np.argmax(prediction[0])  # Assuming prediction is a batch
            severity_name = labels[index]
            return severity_name
        except Exception as e:
            st.error(f"Error during acne severity prediction: {e}")
            return None

    st.title("Acne Prediction and Treatment Recommendation")
    st.sidebar.header("Upload an Image")

    # Upload an image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")
        
        # Step 1: Load face detection model
        try:
            st.subheader("Step 1: Checking for a face...")
            face_model = load_model('model/my_face_classifier_model (2).h5')  # Replace with the path to your face detection model
            face_data, _ = preprocess_image(uploaded_file, target_size=(128, 128))
            prediction = face_model.predict(face_data, verbose=0)
            if prediction[0][0] > 0.5:
                st.warning("This is not a face. Please upload a valid face image.")
                return
            else:
                st.success("Face detected successfully!")
        except Exception as e:
            st.error(f"Error loading or using the face detection model: {e}")
            return
        
        # Step 2: Predict acne presence
        try:
            st.subheader("Step 2: Checking for acne presence...")
            acne_presence_model_path = "model/acne_nonacne/model.savedmodel"  # Replace with your SavedModel path
            labels_acne_presence = ["Non Acne", "Acne"]
            acne_presence_layer = TFSMLayer(acne_presence_model_path, call_endpoint='serving_default')
            input_shape = (224, 224, 3)
            inputs = Input(shape=input_shape)
            outputs_presence = acne_presence_layer(inputs)
            acne_presence_model = Model(inputs=inputs, outputs=outputs_presence)

            acne_presence_data, _ = preprocess_image(uploaded_file)
            presence_prediction = acne_presence_model.predict(acne_presence_data, verbose=0)
            
            if isinstance(presence_prediction, dict):
                # Mengakses key pertama dari dictionary
                presence_prediction = presence_prediction[next(iter(presence_prediction.keys()))]
            
            acne_presence_label = labels_acne_presence[np.argmax(presence_prediction)]
            
            st.info(f"Acne Presence: {acne_presence_label}")

            if acne_presence_label == "Non Acne":
                st.success("No acne detected! Your skin looks great.")
                return
        except Exception as e:
            st.error(f"Error loading or using the acne presence model: {e}")
            return

        # Step 3: Prediksi tipe jerawat
        try:
            st.subheader("Step 3: Predicting acne type...")
            acne_type_model_path = "model/acne_type/model.savedmodel"
            labels_acne_type = ["White Comedo", "Nodule", "Pustule", "Papule", "Black Comedo"]
            acne_type_layer = TFSMLayer(acne_type_model_path, call_endpoint='serving_default')
            outputs_type = acne_type_layer(inputs)
            acne_type_model = Model(inputs=inputs, outputs=outputs_type)

            acne_type = predict_acne_type(uploaded_file, acne_type_model, labels_acne_type)
            if acne_type:
                st.success(f"Predicted Acne Type: {acne_type}")

                # Step 4: Rekomendasi pengobatan
                st.subheader("Step 4: Treatment Recommendation")
                treatment = acne_treatment(acne_type)
                st.info(f"Recommended Treatment: {treatment}")
            else:
                st.error("Failed to determine acne type.")
        except Exception as e:
            st.error(f"Error during acne type prediction: {e}")
            return

        # Step 4: Predict acne severity
        try:
            st.subheader("Step 5: Predicting acne severity...")
            acne_severity_model_path = "model/acne_severity/model.savedmodel"  # Replace with your SavedModel path
            labels_acne_severity = ["Mild", "Moderate", "Severe", "Very Severe"]
            acne_severity_layer = TFSMLayer(acne_severity_model_path, call_endpoint='serving_default')
            outputs_severity = acne_severity_layer(inputs)
            acne_severity_model = Model(inputs=inputs, outputs=outputs_severity)

            severity_name = predict_acne_severity(uploaded_file, acne_severity_model, labels_acne_severity)
            if severity_name:
                st.success(f"Predicted Acne Severity: {severity_name}")
        except Exception as e:
            st.error(f"Error loading or using the acne severity model: {e}")