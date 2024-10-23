# product review analyzer

Amazon Echo Sentiment Analysis Application
This project is a Sentiment Analysis Application developed for analyzing customer reviews of Amazon Echo using machine learning models. The application classifies reviews into different sentiment categories by detecting the emotional tone behind them.

Overview
This application employs three machine learning models to perform sentiment analysis:

Gradient Boosting Classifier (Boosted Trees)
Random Forest Classifier
Decision Tree Classifier
Through extensive training, testing, and hyperparameter tuning, the best-performing model was selected based on accuracy and evaluation metrics. The final model achieved a mean accuracy of 96.87%, with training accuracy of 99% and testing accuracy of 93%.

A Streamlit-based front-end interface allows users to interact with the model and analyze the sentiment of customer reviews.

Features
Multi-model analysis: Uses Gradient Boosting, Random Forest, and Decision Tree classifiers.
Hyperparameter tuning: Optimizes the performance of the models.
Evaluation metrics: Tracks model performance using accuracy and other key metrics.
User-friendly front-end: Provides a simple interface using Streamlit for inputting reviews and viewing sentiment results.
Real-time sentiment detection: Detects the emotional tone behind customer reviews in real-time.
Tech Stack
Back-end:
Python
Machine Learning models (Gradient Boosting, Random Forest, Decision Tree)
Libraries: scikit-learn, pandas, numpy, etc.
Front-end:
Streamlit for creating an interactive user interface.
Installation
To run the project locally, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/your-repository-link.git
cd sentiment-analysis-app
Install required dependencies: Make sure you have Python installed, then install the necessary packages using pip:

bash
Copy code
pip install -r requirements.txt
Run the application: Start the Streamlit server by running the following command:

bash
Copy code
streamlit run app.py
Access the application: Open your browser and go to http://localhost:8501 to interact with the sentiment analysis tool.

Model Performance
Mean Accuracy: 96.87%
Training Accuracy: 99%
Testing Accuracy: 93%
Usage
Enter a review: Type or paste an Amazon Echo review into the text box.
Analyze sentiment: Click the "Analyze" button to detect the sentiment of the review.
View results: The application will display whether the review is positive, negative, or neutral, along with the detected emotional tone.
Future Improvements
Support for more machine learning models.
Fine-tuned sentiment categories (e.g., joy, sadness, anger, etc.).
Advanced NLP techniques for more accurate sentiment detection.
