import streamlit as st
import tensorflow as tf   
import numpy as np

# Tensorflow Model Prediction

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) # To convert image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

#slidebar 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page

if(app_mode == "Home"):
    st.header ("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home.jpg"
    st.image(image_path, use_container_width = True)
    st.markdown("""
                
    Welcome to the **Plant Disease Recognition System**—your smart assistant for detecting plant diseases with precision and ease!  

    ## 🌟 How It Works  
    1. **Upload an Image** – Navigate to the **Disease Recognition** page and upload a photo of a plant showing possible signs of disease.  
    2. **AI-Powered Analysis** – Our advanced machine learning models will analyze the image to identify potential diseases.  
    3. **Instant Results** – Get a detailed diagnosis along with expert recommendations to protect your plants.  

    ## 🚀 Why Choose Our System?  
    ✔ **Highly Accurate** – Powered by cutting-edge AI to ensure precise disease detection.  
    ✔ **User-Friendly Interface** – Simple, intuitive, and easy to use for everyone.  
    ✔ **Fast & Reliable** – Receive real-time results to take immediate action.  

    ## 🌍 Get Started Today!  
    Head to the **Disease Recognition** page and upload an image to experience the power of AI-driven plant disease detection.  
    
   💡 Want to learn more? Visit the **About** page to explore our mission, the team behind this project, and how we're working to revolutionize plant health monitoring!  
""")
    
    ## About Page
elif(app_mode == "About"):
    st.header("About")
    st.markdown("""
     ## 🌿 Plant Disease Dataset 📊  

     Our dataset contains images of various plant species, categorized as **healthy** or **diseased**. It supports AI-driven disease detection for efficient diagnosis.  
    
    ### Content 
    1. Train (70295 Images)
    2. Valid (17572 Images)
    3. Test (33 Images)
    
    ### 📂 Dataset Overview  
    - **Plant Types:** Tomatoes, potatoes, apple, corn, and more.  
    - **Diseases Covered:** Blight, rust, mildew, bacterial infections, etc.  
    - **Image Quality:** High-resolution, diverse environments.  

    ### 🏷 Data Structure  
    - **Images:** Labeled folders based on plant and disease type.  
    - **Metadata:** Includes plant species, disease name, severity level.  

    ### 🔬 Why It Matters  
    ✔ Improves AI accuracy for disease recognition.  
    ✔ Supports farmers & researchers in agriculture.  
    ✔ Continuously expandable with new disease data.  

    📥 Used for **training, validation, and testing** in our deep learning model to enhance plant disease detection.  
    🌱 **Empowering Agriculture with AI!** 🌍  
    """)
    
    # Prediction Page 
    
elif(app_mode == "Disease Recognition"):
    st.header ("Disease Recognition")
    test_image = st.file_uploader ("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, use_container_width = True)
    
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
    # Define class
        
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))


