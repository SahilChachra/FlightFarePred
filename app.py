
# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__) # initializing a flask app
@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("home.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            airline=request.form['airline']
            doj = request.form['doj']
            source = request.form['source']
            destination = request.form['destination']
            route = request.form['route']
            dep_time = request.form['dep_time']
            arrival_time = request.form['arrival_time']
            duration = request.form['duration']
            stops = request.form['stops']
            info = request.form['info']
            
            if info=='No info':
                info='No Info'
            
            with open("label_encoder_Airline.sav", 'rb') as f:
                leA = pickle.load(f)
                airline=leA.transform([airline])
            with open("label_encoder_Source.sav",'rb') as f:
                leS=pickle.load(f)
                source=leS.transform([source])
            with open("label_encoder_Destination.sav",'rb') as f:
                leD=pickle.load(f)
                destination=leD.transform([destination])
            with open("label_encoder_Route.sav",'rb') as f:
                leR=pickle.load(f)
                route=leR.transform([route])
            with open("label_encoder_Total_Stops.sav",'rb') as f:
                leTS=pickle.load(f)
                stops=leTS.transform([stops])
            with open("label_encoder_Additional_Info.sav",'rb') as f:
                leI=pickle.load(f)
                info=leI.transform([info])
                
            date_day = doj.split('/')[0]
            date_month = doj.split('/')[1]
            date_year = doj.split('/')[2]
            
            if 'h' in duration and 'm' in duration:
                temp=int(duration[0:duration.index('h')])*60+int(duration.split()[1][0:duration.split()[1].index('m')])
            elif 'm' not in duration and 'h' in duration:
                temp=int(duration[0:duration.index('h')])*60
            elif 'm' in duration:
                temp=int(duration[0:duration.index('m')])
            else:
                print("Unhandled datapoint :==> ",duration)
            duration=temp
            
            print(airline,source,destination,route,duration,stops,info,date_day,date_month,date_year)

            with open("random_regressor.sav", 'rb') as f:
                model = pickle.load(f)
            # predictions using the loaded model file
            prediction=model.predict([[airline,source,destination,route,duration,stops,info,date_day,date_month,date_year]])
            print('Prediction is', prediction)
            
            # showing the prediction results in a UI
            return render_template('op.html',prediction=prediction)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'Invalid Input'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True) # running the app
