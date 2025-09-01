✈️ Airline Passenger Satisfaction Prediction
This project predicts whether an airline passenger is satisfied or dissatisfied based on their demographic, travel, and service-related data. The model leverages machine learning pipelines and a Random Forest Classifier to process, train, and evaluate predictions.

📂 Project Structure
airline-satisfaction-prediction/
│── data/
│   └── airline_satisfaction.csv        # Dataset (not included in repo)
│── notebooks/
│   └── model_training.ipynb            # Development notebook
│── src/
│   └── train_model.py                  # Main training script
│── README.md                           # Project documentation
│── requirements.txt                    # Dependencies


📊 Dataset
The dataset comes from Kaggle Airline Passenger Satisfaction Dataset. It contains passenger survey data, including demographics, travel information, and service ratings.
Features include:
Passenger details: Age, Flight Distance, Delay Times


Travel context: Customer Type (Loyal / Disloyal), Type of Travel (Business / Personal), Class (Eco / Business / Eco Plus)


Service ratings: Inflight WiFi, Seat Comfort, Online Boarding, Food & Drink, Entertainment, etc.


Target:


satisfied → True


neutral or dissatisfied → False



⚙️ Installation
Clone the repository:

 git clone https://github.com/your-username/airline-satisfaction-prediction.git
cd airline-satisfaction-prediction


Create and activate a virtual environment (optional but recommended):

 python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies:

 pip install -r requirements.txt



🚀 Model Training
Run the training script:
python src/train_model.py

This will:
Load and preprocess the dataset


Train a Random Forest Classifier


Evaluate on a test set with accuracy and classification report



📈 Model Performance
Example output:
RF Accuracy: 0.92

Classification Report:
              precision    recall  f1-score   support

       False       0.91      0.93      0.92      5000
        True       0.93      0.90      0.91      5000

    accuracy                           0.92     10000
   macro avg       0.92      0.92      0.92     10000
weighted avg       0.92      0.92      0.92     10000


🔮 Passenger Satisfaction Prediction
You can make predictions for a new passenger profile:
new_passenger = pd.DataFrame({
    "Age": [35],
    "Flight Distance": [1200],
    "Departure Delay in Minutes": [15],
    "Arrival Delay in Minutes": [10],
    "Inflight wifi service": [4],
    "Seat comfort": [5],
    "Food and drink": [3],
    "Inflight entertainment": [4],
    "Gate location": [3],
    "Customer Type": ["Loyal"],
    "Type of Travel": ["Business"],
    "Class": ["Economy"],
    "Departure/Arrival time convenient": [4],
    "Online boarding": [5],
    "Baggage handling": [4],
    "Leg room service": [3],
    "Ease of Online booking": [4],
    "On-board service": [4],
    "Checkin service": [5],
    "Cleanliness": [5],
    "Inflight service": [4]
})

predicted_satisfaction = model.predict(new_passenger)
print("Predicted satisfaction:", predicted_satisfaction[0])

Example output:
Predicted satisfaction: True


📦 Dependencies
Main libraries:
Python 3.8+


pandas


numpy


scikit-learn


Install with:
pip install pandas numpy scikit-learn


🏢 Business Use Case
Airlines can use this model to:
Identify key drivers of customer satisfaction


Improve customer retention by addressing dissatisfaction factors


Personalize passenger services


Predict satisfaction for new flight schedules or services



🔮 Future Improvements
Tune hyperparameters using Grid Search / Random Search / Bayesian Optimization


Try advanced models like XGBoost, LightGBM, or CatBoost


Apply feature selection or dimensionality reduction


Deploy as a Flask/Django API or Streamlit web app



