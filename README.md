Data-to-Insights Assistant

This project is a GenAI-powered application that automatically converts raw datasets into meaningful insights, visualizations, and analytical summaries. It combines traditional data analysis techniques with Large Language Models (LLMs) to generate natural language insights from any uploaded CSV file.
The application is designed using Streamlit for the interface, Python for the backend, and OpenAI models for generating human-like analytical summaries. It provides automated chart generation, data profiling, and intelligent pattern recognition.

Objective
The goal of this project is to demonstrate how data analysis can be enhanced with Generative AI. The application simplifies exploratory data analysis by automating the following:
Data profiling and summarization
Generation of histograms, bar charts, scatter plots, and correlation heatmaps
Auto chart intelligence for trend and relationship detection
AI-generated insights using GPT models
This project simulates the workflow of a data scientist who leverages GenAI to save time and extract high-level insights efficiently.

Key Features
Automatic Data Profiling
Calculates column statistics including mean, standard deviation, min, max, null counts, and unique values.
Automated Visualizations
Generates multiple charts including histograms, top categorical bars, scatter plots, and correlation matrices.
Auto Chart Intelligence
Dynamically creates line, bar, and scatter plots based on data patterns such as time-like columns or categorical relationships.
AI-Powered Insights
Integrates OpenAI GPT models to summarize findings, suggest hypotheses, and detect data quality issues.
Interactive Chat Interface
Allows users to ask natural language questions about their data and receive contextual analytical answers.

Technology Stack
Language: Python
Frontend Framework: Streamlit
Libraries: Pandas, NumPy, Matplotlib, Seaborn
AI Integration: OpenAI GPT models (gpt-4o-mini, gpt-4, gpt-3.5-turbo)
Environment: Virtual Environment (venv)

Installation and Setup
Step 1: Clone the Repository
git clone https://github.com/<your-username>/DataToInsightsAssistant.git
cd DataToInsightsAssistant

Step 2: Create a Virtual Environment
python -m venv venv
venv\Scripts\activate

Step 3: Install Required Dependencies
pip install -r requirements.txt

Step 4: Add Your OpenAI API Key

Option 1 – Set environment variable:
setx OPENAI_API_KEY "your_api_key_here"
Option 2 – Paste your key directly into the sidebar input when the app is running.

Step 5: Run the Application
streamlit run data_to_insights_app.py

requirements.txt
streamlit
pandas
numpy
matplotlib
seaborn
openai>=1.0.0

Example Workflow
Upload any CSV dataset.
View automatic data summary and statistical overview.
Explore generated visualizations including histograms and bar charts.
Use Auto Chart Intelligence to discover trends and correlations.
Generate AI-powered insights or ask questions about the dataset directly.

Learning and Purpose
This project was created to strengthen my practical understanding of:
Data analysis and visualization using Python
LLM integration in real-world analytical workflows
Streamlit application development
Automating data insight generation with Generative AI
It demonstrates the ability to combine data science principles with modern AI techniques to create practical, intelligent tools for data interpretation.

Author : 
Chetan Jaiswal
Data Scientist | Python Developer | GenAI Practitioner

