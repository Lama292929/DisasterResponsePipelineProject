# DisasterResponsePipelineProject


[Udacity Data Scientists Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
<br> <b> Project 2 : Disaster Response Pipeline Project  </b>



<h2> Table of Contents </h2> 
1. Installation   <br>
2. Project Overview <br>
3. File Description / Instructions <br> 
4. Screenshots <br> 
5. Licensing, Authors, and Acknowledgements <br>



<h2> Installation </h2> 
This project was written in HTML and Python Programming Language, and requires some python packages such as: pandas, numpy, flask, plotly, re, sqlchemy, and sklearn


<h2> Project Overview </h2> 
Web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


<h2> File Description / Instructions  </h2>

Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



<h2> Screenshots </h2>
  
  
  
<h2> Licensing, Authors, Acknowledgements </h2>


