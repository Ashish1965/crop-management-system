from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the models and scalers
try:
    model = joblib.load(open('model.pkl', 'rb'))
    sc = joblib.load(open('standscaler.pkl', 'rb'))
    mx = joblib.load(open('minmaxscaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model or scaler file not found!")
    # Optionally, return an error page or default model
except Exception as e:
    print(f"An error occurred: {e}")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get user input from form
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    # Prepare the feature list for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    try:
        # Apply scaling and prediction
        mx_features = mx.transform(single_pred)   # MinMaxScaler transformation
        sc_mx_features = sc.transform(mx_features)  # StandardScaler transformation
        prediction = model.predict(sc_mx_features)  # Predict the crop
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', result="Error during prediction. Please try again.")

    # Crop mapping dictionary
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    # Image mapping for crops
    crop_images = {
        "Rice": "Rice.png", "Maize": "Maize.webp", "Jute": "Jute.png", 
        "Cotton": "Cotton.png", "Coconut": "Coconut.webp", "Papaya": "Papaya.avif", 
        "Orange": "Orange.png", "Apple": "Apple.png", "Muskmelon": "Musk melon.png", 
        "Watermelon": "Water melon.png", "Grapes": "Grapes.png", "Mango": "Mango.png", 
        "Banana": "Banana.png", "Pomegranate": "Pomegranate.png", "Lentil": "Lentil.png", 
        "Blackgram": "Blackgrams.png", "Mungbean": "Mungbean.webp", 
        "Mothbeans": "MothBean.webp", "Pigeonpeas": "Pigeon.jpg", "Kidneybeans": "Kidney Bean.png", 
        "Chickpea": "Checkpea.avif", "Coffee": "Coee.png"
    }

    # Map prediction to crop and image
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated there."
        crop_img = crop_images.get(crop, "crop.png")  # Default image if crop image is missing
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
        crop_img = "sorry.png"

    # Return result to the template
    return render_template('index.html', result=result, crop_img=crop_img)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
