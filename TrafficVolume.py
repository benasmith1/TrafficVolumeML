# App to predict the chances of admission using a pre-trained ML model in Streamlit

# Import libraries
import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# password_guess = st.text_input("What is the Password?")
# if password_guess != st.secrets["password"]:
#     st.stop()


# Set up the app title and image
st.title('Traffic Volume Predictor')
st.image('traffic_image.gif', use_column_width = True, 
         caption = "Predict your traffic volume based on date time and conditions")

st.write("This app uses multiple inputs to predict traffic volume.") 

# Reading the pickle file that we created before 
model_pickle = open('reg_traffic.pickle', 'rb') 
reg_model = pickle.load(model_pickle) 
model_pickle.close()

# Load the default dataset
default_df = pd.read_csv('Traffic_Volume.csv')
default_df["date_time"] = pd.to_datetime(default_df["date_time"])
default_df["day_of_week"] = default_df["date_time"].dt.dayofweek #day_name()
default_df["month"] = default_df["date_time"].dt.month #month_name()
default_df["hour"] = default_df["date_time"].dt.hour
default_df = default_df.drop(columns = ['date_time'])
default_df["holiday"] = default_df["holiday"].fillna('None')
default_df = default_df.drop(columns = ['traffic_volume'])

# Get categories and ranges
holidays = default_df["holiday"].unique()
clouds = default_df["clouds_all"].unique()
weathers = default_df["weather_main"].unique()
months = range(1,13)
hours = range(0,24)
weathers = default_df["weather_main"].unique()
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

max_temp = max(default_df["temp"])
min_temp = min(default_df["temp"])
temp_diff = default_df["temp"].sort_values().diff().dropna()
temp_step = temp_diff[temp_diff != 0].min()

max_rain = max(default_df["rain_1h"])
min_rain = min(default_df["rain_1h"])
rain_diff = default_df["rain_1h"].sort_values().diff().dropna()
rain_step = rain_diff[rain_diff != 0].min()

max_snow = max(default_df["snow_1h"])
min_snow = min(default_df["snow_1h"])
snow_diff = default_df["snow_1h"].sort_values().diff().dropna()
snow_step = snow_diff[snow_diff != 0].min()

days = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}

months = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

submit_button_bool = False
file_input_bool = False


# FORM INPUT
with st.sidebar.expander("Predict from form"):
    # Sidebar for user inputs with an expander
    with st.form("user_inputs_form"):
        st.header("Enter The Traffic Details manually using the form below")        
        holiday = st.selectbox('Choose whether it is a holiday or not', options=holidays, help="")
        temp = st.slider('Average temperature in Kelvin', min_value=min_temp, max_value=max_temp, value=250.0, step=temp_step, help="°F = (K − 273.15) × 1.8 + 32")
        rain_1h = st.slider('Amount in mm of rain that occurred in the hour', min_value=min_rain, max_value=max_rain, value=0.0, step=rain_step, help="")
        snow_1h = st.slider('Amount in mm of snow that occurred in the hour', min_value=min_snow, max_value=max_snow, value=0.0, step=snow_step, help="")
        clouds_all = st.slider('Percentage of cloud cover', min_value=0, max_value=100, value=10, step=1, help="in %")
        weather_main = st.selectbox('Choose weather', options=weathers, help="The day's weather")
        month = st.selectbox('Choose month', options=months, help="")
        day_of_week = st.selectbox('Choose day of the week', options=days, help="")
        hour = st.selectbox('Choose hour of the day', options=hours, help="Date and hour of the data collected in local CST time 0:00-24:00")
        submit_button = st.form_submit_button("Predict")
        submit_button_bool = True
        file_input_bool = False
        

# FILE INPUT
with st.sidebar.expander("Predict from csv"):
    #sidebar file input
    file_input = st.file_uploader("Upload file of day conditions")
    file_input_bool = True

if file_input and not submit_button:
    st.success("CSV file uploaded")
    file_input_bool = True
    submit_button_bool = False
if submit_button:
    submit_button_bool = True
    file_input_bool = False
    st.success("Form data uploaded")
if not(submit_button_bool) and not(file_input_bool):
    st.info("Please choose a data upload option in the sidebar")


alpha = st.slider('Alpha', min_value=0.0, max_value=1.0, value=0.1, step=0.01, help="% Prediction Interval = 1-alpha.")

if submit_button_bool: # If form input


    encode_df = default_df.copy()

    day_of_week = days[day_of_week]
    month = months[month]

    traffic_test = [holiday, temp, rain_1h, snow_1h, clouds_all, weather_main, day_of_week, month, hour]
    encode_df.loc[len(encode_df)] = traffic_test
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(1)

    # Get the prediction with its intervals
    prediction, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are within [0, infinity]
    lower_limit = max(0, lower_limit[0][0])

    # Show the prediction on the app
    st.write("## Predicting Traffic Volume...")

    # Display results using metric card
    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value * 100:,.2f}")

    st.write(f"With a {(1-alpha)*100}% prediction interval:")
    st.write(f"**Prediction Interval**: [{lower_limit* 100:,.2f}, {upper_limit* 100:,.2f}]")

elif file_input_bool: #if file input
    #submit_button = False

    traffic_test = pd.read_csv(file_input)
    traffic_test = traffic_test.replace({"weekday": days})
    traffic_test = traffic_test.replace({"month": months})
    traffic_test.rename(columns={"weekday": "day_of_week"}, inplace=True)

    encode_df = default_df.copy()

    # Combine the csv of user data to default_df
    user_rows_n = len(traffic_test)
    encode_df = pd.concat([encode_df, traffic_test])

    # Create dummies for encode_df
    encode_dummy_df = pd.get_dummies(encode_df)

    # Extract encoded user data
    user_encoded_df = encode_dummy_df.tail(user_rows_n)

    #get predictions and prediction intervals
    preds, intervals = reg_model.predict(user_encoded_df, alpha = alpha)
    preds = pd.Series(preds)
    traffic_test["traffic_volume"] = preds
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1][0][0]

    # Ensure limits are within [0, infinity]
    lower_limit = max(0, lower_limit[0][0])

    traffic_test[f"Lower {(1-alpha)*100}% Pred Interval Limit"] = lower_limit
    traffic_test[f"Upper {(1-alpha)*100}% Pred Interval Limit"] = upper_limit

    st.write(traffic_test)


# Additional tabs for xGBoost model performance
st.subheader("Model Insights")
st.write("Below are illustrations of the model's performance on validation data.")
tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")

