from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


application=Flask(__name__)

app=application

# Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        area=request.form.get('area'),
        no_of_bedrooms=int(request.form.get("no_of_bedrooms")),
        no_of_bathrooms=int(request.form.get("no_of_bathrooms")),
        total_rooms=int(request.form.get("total_rooms")),
        region=request.form.get("region"),
        building_type=request.form.get("building_type"),
        property_zone=request.form.get("property_zone"),
        house_age=request.form.get("house_age")

        if area!='' and house_age!='':

            data=CustomData(
                area=int(request.form.get('area')),
                no_of_bedrooms=int(request.form.get("no_of_bedrooms")),
                no_of_bathrooms=int(request.form.get("no_of_bathrooms")),
                total_rooms=int(request.form.get("total_rooms")),
                region=request.form.get("region"),
                building_type=request.form.get("building_type"),
                property_zone=request.form.get("property_zone"),
                house_age=int(request.form.get("house_age"))
            )

            pred_df=data.get_data_as_dataframe()
        else:
            data=CustomData(
                area=0,
                no_of_bedrooms=int(request.form.get("no_of_bedrooms")),
                no_of_bathrooms=int(request.form.get("no_of_bathrooms")),
                total_rooms=int(request.form.get("total_rooms")),
                region=request.form.get("region"),
                building_type=request.form.get("building_type"),
                property_zone=request.form.get("property_zone"),
                house_age=0
            )

            pred_df=data.get_data_as_dataframe()

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results[0])


if __name__=='__main__':
    app.run('0.0.0.0',debug=True)