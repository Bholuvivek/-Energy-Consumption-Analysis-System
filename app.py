from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle as pkl

app = Flask(__name__)

# Load the trained model
model = pkl.load(open('PCASS_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Retrieve form data
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # Process the form data (e.g., save to database, send email)
        # Add your code here
        
        # Redirect to a thank you page
        return "Thank You!"
    
    return render_template('contact.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Retrieve form data
        GlobalReactivePower = float(request.form['GlobalReactivePower'])
        Global_intensity = float(request.form['Global_intensity'])
        Sub_metering_1 = float(request.form['Sub_metering_1'])
        Sub_metering_2 = float(request.form['Sub_metering_2'])
        Sub_metering_3 = float(request.form['Sub_metering_3'])
        
        # Perform prediction using the loaded model
        x = [[GlobalReactivePower, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3]]
        output = round(model.predict(x)[0], 3)
        message = "Your result is {} watt".format(output)
       
        
        return render_template('home.html', output=output, message=message )
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run()
