import numpy as np
from flask import Flask, request,  render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('Model.pkl','rb'))


    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   
    int_features = [x for x in request.form.values()]
    final_features =np.array(int_features).reshape(1,-1)
    
    prediction = model.predict(final_features)

    output = prediction[0]
    if(output==0):
        return render_template('index.html',prediction_text='WATER IS NOT POTABLE!!!')
    else:
        return render_template('index.html',prediction_text='WATER IS POTABLE!!!')

if __name__ == "__main__":
    app.run(debug=False)