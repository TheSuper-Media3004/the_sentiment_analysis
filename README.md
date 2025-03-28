# Sentiment Analysis Project

This project performs sentiment analysis on textual data, classifying text into positive, negative, or neutral sentiments. It includes data preprocessing, model training, and a web interface for user interaction.

## Features

- **Data Preprocessing:** Cleans and prepares text data for analysis.
- **Model Training:** Trains a machine learning model to classify text sentiment.
- **Web Interface:** Provides a user-friendly interface to input text and view sentiment results.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TheSuper-Media3004/the_sentiment_analysis.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd the_sentiment_analysis
   ```
3. **Install Required Dependencies:**
   Ensure you have Python installed. Then, install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the Model:**
   Run the `train_model.py` script to train the sentiment analysis model:
   ```bash
   python train_model.py
   ```
2. **Start the Web Application:**
   Launch the Flask web application using `app.py`:
   ```bash
   python app.py
   ```
3. **Interact with the Application:**
   Open your web browser and navigate to `http://127.0.0.1:5000/` to input text and view sentiment analysis results.

## Project Structure

- `app.py`: Initializes and runs the Flask web application.
- `train_model.py`: Contains the code for training the sentiment analysis model.
- `views.py`: Manages the routes and views for the web application.
- `static/`: Contains static files like CSS, JavaScript, and images.
- `templates/`: Holds HTML templates for rendering web pages.

## Dependencies

- Python 3.11.7
- Flask
- Required Python packages listed in `requirements.txt`


