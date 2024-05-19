# Load libraries
import streamlit as st
import pandas as pd
from joblib import load
import datetime
from src.funcs import extract_temporal_features
import plotly.express as px


def main():
    # Setting page config
    st.set_page_config(page_title="Seoul Biking Demand",
                       page_icon=":bike:")

    # Name tabs
    tabs = st.tabs(['🏠 Home', '🚲 Predict Rented Bikes'])

    # Home Page
    with tabs[0]:
        # Title
        st.title('🇰🇷 Seoul Bike Sharing Demand App')

        # Header and engaging introduction
        st.header('📈 Predict Seoul Bike Rentals')
        st.write("""
        Explore and predict the number of bikes rented in Seoul using an interactive web application. 
        This tool allows you to dynamically adjust parameters and instantly see how various factors, 
        such as weather conditions, influence bike rental demand.
        """)

        # Contact and further engagement
        st.header('📞 Contact Information')
        st.write("""
        For more information, please connect with me on [LinkedIn](https://www.linkedin.com/in/jordanlarot)
        or email me at jordan@larot.com!
        """)

        # Footer or additional notes
        st.markdown('Last updated: May 2024')

    # Application
    with tabs[1]:
        # Specify date range
        min_date = datetime.date(2024, 1, 1)
        max_date = datetime.date(2025, 12, 31)

        st.title('🇰🇷 Seoul Rented Bike Count')

        # Load preprocessing pipeline and model
        preprocessor = load('models/pipelines/preprocessing_pipeline.joblib')
        model = load('models/xgboost/best-model.pkl')

        date = st.date_input('Select a date', value=datetime.date.today(), min_value=min_date, max_value=max_date)

        # Create two columns for text inputs
        col1, col2 = st.columns(2)

        with col1:
            weather = st.selectbox('Temperature', options=['Mild', 'Cold', 'Hot'])

        with col2:
            humidity = st.selectbox('Humidity', options=['Low Humidity', 'Moderate Humidity', 'High Humidity'])

        # Create three columns for the text inputs
        col3, col4, col5 = st.columns(3)

        # Input for Rain Category
        with col3:
            rain = st.selectbox('Rain', options=['No Rain', 'Light Rain', 'Moderate Rain', 'Heavy Rain'])

        # Input for Wind Category
        with col4:
            wind = st.selectbox('Wind', options=['Light Wind', 'Moderate Wind', 'High Wind'])

        # Input for Snow Category
        with col5:
            snow = st.selectbox('Snow', options=['No Snow', 'Light Snow', 'Moderate Snow', 'Heavy Snow'])

        if st.button('Predict Rented Bike Count'):

            # Create input df
            input_df = pd.DataFrame({
                'Date': [date],
                'Weather Condition': [weather],
                'Humidity Category': [humidity],
                'Rain Category': [rain],
                'Wind Category': [wind],
                'Snow Category': [snow],
            })

            # Extract temporal features
            df_temporal = extract_temporal_features(input_df, web_app=True)

            # List to store each new row
            all_rows = []

            # Iterate over each row in the DataFrame
            for index, row in df_temporal.iterrows():
                for hour in range(24):  # For each hour of the day
                    new_row = row.copy()  # Copy the original row
                    new_row['Hour'] = hour  # Add the hour
                    all_rows.append(new_row)  # Add the new row to the list

            # Convert the list of rows to a DataFrame using pd.concat
            featured_engineered_df = pd.concat([pd.DataFrame([row]) for row in all_rows], ignore_index=True)

            # Drop 'Date'
            transformed_df = featured_engineered_df.drop('Date', axis=1)

            # Process the input data through the preprocessing pipeline
            processed_data = preprocessor.transform(transformed_df)

            # Predict with the XGBoost model
            prediction = model.predict(processed_data)

            # Create list of hours
            hours = [
                "12 AM", "1 AM", "2 AM", "3 AM", "4 AM", "5 AM", "6 AM", "7 AM", "8 AM", "9 AM", "10 AM", "11 AM",
                "12 PM", "1 PM", "2 PM", "3 PM", "4 PM", "5 PM", "6 PM", "7 PM", "8 PM", "9 PM", "10 PM", "11 PM"]

            # Create dataframe to store predictions
            df_predictions = pd.DataFrame()

            # Add 'Hour' and 'Rented Bike Count' columns
            df_predictions['Hour_US'] = hours
            df_predictions['Rented Bike Count'] = prediction.round(0)
            df_predictions['Hour'] = [i for i in range(0, 24)]

            # Set index to 'Hour'
            df_predictions = df_predictions.set_index('Hour_US', drop=True)
            df_predictions.index.name = 'Hour_US'

            # Create a bar chart using Plotly Express
            fig = px.bar(df_predictions, x='Hour', y='Rented Bike Count',
                         labels={'Hour': ' Hour of the Day', 'Rented Bike Count': 'Rented Bike Count'},
                         title='Predicted Hourly Rented Bike Count Visualization',
                         color_discrete_sequence=['#FF4B4B'])

            # Customize tooltip text with HTML
            fig.update_traces(hovertemplate='<b>Hour:</b> %{x}<br><b>Rented Bikes:</b> %{y}')

            # Disable zoom and pan
            fig.update_layout(
                dragmode=False,
                xaxis_fixedrange=True,
                yaxis_fixedrange=True,
                hoverlabel=dict(bgcolor="darkgrey",
                                font_size=14,
                                font_family="Montserrat")
            )

            # Display the plot in the Streamlit app
            st.plotly_chart(fig, use_container_width=True)

            # Report
            with st.expander("🔎 View Report"):

                # Display report
                st.table(df_predictions['Rented Bike Count'])


if __name__ == '__main__':
    main()