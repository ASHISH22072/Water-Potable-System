import numpy as np
from flask import Flask, request,render_template
import pickle
import json
import requests # type: ignore

app = Flask(__name__)
model = pickle.load(open('Model.pkl', 'rb'))

@app.route('/')
def home():
    temp=gemini("what is water ")
    print(temp)
    return render_template('index.html',temp1=temp)

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = np.array(int_features).reshape(1, -1)
    
    prediction = model.predict(final_features)
    output = prediction[0]

    # Define the headings
    headings = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

    # Create the query string
    query_parts = [f"{headings[i]}={final_features[0][i]}" for i in range(len(headings))]
    query_string = "on basis on given parameters can you tell in list diseases this water can cause " + ", ".join(query_parts) + " give only names of possible diseases with diseases as key and if there is explaininatory text then in that in key Explaination Word limit for response is 70 words"

    # Call the gemini function with the query string

    ai_response = gemini(query_string)
    print(ai_response)  # For debugging purposes

    if output == 0:
        return render_template('index.html', prediction_text='WATER IS NOT POTABLE!!!', temp1=ai_response)
    else:
        return render_template('index.html', prediction_text='WATER IS POTABLE!!!', temp1=ai_response)


def gemini(input_text):
    
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCY7XunuYYYmtZGgYBUz0vLGuQ1tF2eSPs"  
    headers = {  
        'Content-Type': 'application/json'
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": input_text
                    }
                ]
            }
        ]
    }
    
   
    response = requests.post(api_url, headers=headers, json=data, timeout=10)
    ai_response = response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response from AI')

    return ai_response
    

if __name__ == "__main__":
    app.run(debug=True)