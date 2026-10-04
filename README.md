# Intelligent Fitness Recommendation System

### Using Soft Computing Techniques — ANN + Fuzzy Logic

An intelligent fitness recommendation system that combines **Artificial Neural Networks (ANN)** and **Fuzzy Logic** to generate personalized fitness recommendations based on user characteristics, workout information, fitness goals, and experience level.

The system is implemented as an interactive **Streamlit web application** and supports both **Standard Mode** and **Approximation Mode**.

---

## 1. Project Overview

Fitness recommendations should consider several factors such as age, body measurements, workout frequency, experience level, fitness goals, and workout characteristics.

Traditional systems often depend on fixed thresholds and predefined recommendations. Such approaches may not adequately represent gradual or uncertain fitness conditions.

This project addresses this problem using Soft Computing techniques:

- **Artificial Neural Network (ANN)** for predicting estimated calories burned.
- **Fuzzy Logic** for determining suitable workout intensity.
- **Approximation Mode** for handling situations where exact body measurements are unavailable.
- **Recommendation Engine** for combining model outputs and generating personalized fitness guidance.
- **Streamlit** for providing an interactive web interface.

---

## 2. Problem Statement

Many basic fitness applications rely on fixed rules and predefined workout plans. These systems may provide similar recommendations to users even when their activity level, workout experience, fitness goal, or physical characteristics differ.

The objective of this project is to develop an intelligent system that considers multiple fitness parameters and applies Soft Computing techniques to generate flexible and personalized fitness recommendations.

---

## 3. Objectives

The major objectives of the project are:

- To collect and analyze fitness-related information.
- To calculate BMI and other fitness indicators.
- To use Fuzzy Logic for workout intensity determination.
- To train an ANN model for calories-burned prediction.
- To handle uncertain or approximate user information.
- To generate personalized workout recommendations.
- To provide workout duration and frequency guidance.
- To provide basic nutrition and recovery guidance.
- To develop an interactive Streamlit application.
- To demonstrate practical implementation of Soft Computing techniques.

---

# 4. System Architecture

```text
                    USER INPUT
                        |
                        v
              DATA PREPROCESSING
                        |
                        v
              BMI & FEATURE CREATION
                        |
             +----------+----------+
             |                     |
             v                     v
       FUZZY LOGIC                ANN
             |                     |
             v                     v
     WORKOUT INTENSITY       CALORIE PREDICTION
             |                     |
             +----------+----------+
                        |
                        v
              RECOMMENDATION ENGINE
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
        WORKOUT     NUTRITION    RECOVERY
          PLAN       GUIDANCE     GUIDANCE
            |           |           |
            +-----------+-----------+
                        |
                        v
                 FITNESS REPORT
                        |
                        v
                 STREAMLIT APP
```

The project follows a pipeline in which user information is processed, passed through Fuzzy Logic and ANN modules, and then converted into a personalized recommendation. Fitness_Recommendation_Project\_…

---

# 5. Soft Computing Techniques Used

## 5.1 Artificial Neural Network

The ANN is used for **regression-based prediction of estimated calories burned**.

The model learns relationships between fitness characteristics and calories burned from the training dataset.

### ANN Inputs

The model uses fitness-related features such as:

- Age
- Gender
- Weight
- Height
- Maximum BPM
- Average BPM
- Resting BPM
- Session Duration
- Fat Percentage
- Water Intake
- Workout Frequency
- Experience Level
- BMI

### ANN Output

```text
Estimated Calories Burned
```

The ANN prediction is then passed to the Recommendation Engine.

---

## 5.2 Fuzzy Logic

Fuzzy Logic is used to determine appropriate **workout intensity**.

Instead of using only strict boundaries, the system represents fitness conditions using gradual membership categories.

### Fuzzy Inputs

- BMI
- Workout Frequency
- Experience Level

### Fuzzy Output

```text
Intensity Score
```

The numerical score is converted into:

- Light
- Moderate
- High

This allows the system to handle gradual fitness conditions more flexibly than simple fixed thresholds. Intelligent_Fitness_Recommendat…

---

# 6. Approximation Mode

One of the important features of the application is **Approximation Mode**.

Users may not always know their exact:

- Weight
- BMI
- Activity level
- Workout experience

Instead of forcing users to provide inaccurate exact measurements, the system allows approximate information to be used.

### Approximation Mode Inputs

```text
Approximate Height
Body Build
Activity Level
Workout Experience
```

The system estimates an appropriate BMI/weight range and uses representative values for subsequent ANN and Fuzzy Logic processing.

The application clearly identifies these values as **estimates rather than exact measurements**. The original application implements Standard and Approximation modes separately. original_app

---

# 7. Recommendation Engine

The Recommendation Engine combines:

```text
User Information
       +
ANN Prediction
       +
Fuzzy Intensity
       +
Fitness Goal
       +
Recommendation Rules
       |
       v
Personalized Fitness Recommendation
```

The generated recommendation can contain:

- Fitness goal
- Estimated calories burned
- Fuzzy intensity score
- Recommended workout intensity
- Recommended workout type
- Workout focus
- Suggested session duration
- Suggested weekly frequency
- Recovery days
- Hydration guidance
- Protein guidance
- Goal-specific guidance

The recommendation engine combines the ANN prediction, Fuzzy Logic output, fitness goal, and explicit recommendation rules. original_app

---

# 8. Fitness Goals

The application supports different fitness goals, including:

- Weight Loss
- Muscle Gain
- General Fitness
- Endurance
- Strength

The workout type, training focus, duration, frequency, recovery guidance, and protein guidance are adjusted according to the selected goal.

---

# 9. Dataset

The project uses the:

**Gym Members Exercise Dataset**

Source:

Kaggle — Gym Members Exercise Dataset

The dataset contains fitness and workout-related attributes such as:

- Age
- Gender
- Weight
- Height
- Maximum BPM
- Average BPM
- Resting BPM
- Session Duration
- Calories Burned
- Workout Type
- Fat Percentage
- Water Intake
- Workout Frequency
- Experience Level
- BMI

---

# 10. Data Preprocessing

The preprocessing pipeline includes:

1. Loading the dataset.
2. Selecting relevant features.
3. Encoding categorical variables.
4. Creating BMI-related information.
5. Splitting the dataset into training and testing data.
6. Scaling numerical features.
7. Preparing the data for ANN training.

The project methodology includes handling missing/inconsistent data, encoding categorical variables, scaling numerical features, and splitting data for model development. Intelligent_Fitness_Recommendat…

---

# 11. Model Development

The ANN model is developed using:

```text
TensorFlow / Keras
```

The model learns the relationship between user/workout characteristics and calories burned.

The trained model and preprocessing objects are saved for later inference.

### Saved Model Files

```text
models/
│
├── fitness_ann_model.keras
├── fitness_scaler.pkl
├── feature_columns.pkl
└── gender_encoder.pkl
```

This allows the Streamlit application to load the trained model without retraining it every time the application starts.

---

# 12. Fuzzy Logic Workflow

The Fuzzy Logic module follows:

```text
BMI
 |
 +----+
      |
Workout Frequency ----> Fuzzy Inference ----> Intensity Score
      |
Experience Level
 |
 v
Light / Moderate / High
```

The system uses membership functions and fuzzy rules to calculate a numerical intensity score and then maps that score to a workout intensity category.

---

# 13. Streamlit Application

Output for Standard Mode:
<img width="1914" height="970" alt="Screenshot 2026-10-04 200204" src="https://github.com/user-attachments/assets/8eb1e6f0-cf90-40a6-a4ab-4815cfa2acdb" />
<img width="541" height="937" alt="Screenshot 2026-10-04 200222" src="https://github.com/user-attachments/assets/7bd5b193-56c8-4c3f-9132-663c6d76986b" />

Output for Approximation Mode:
<img width="1917" height="760" alt="Screenshot 2026-10-04 200236" src="https://github.com/user-attachments/assets/56503892-fea5-44dc-b829-fec0891d294d" />
<img width="530" height="934" alt="Screenshot 2026-10-04 200255" src="https://github.com/user-attachments/assets/c0910f4b-9985-47e6-970d-02e3a5be00a8" />

### Main Features

- User information form
- Standard Mode
- Approximation Mode
- BMI calculation
- ANN calorie prediction
- Fuzzy workout intensity
- Fitness goal selection
- Workout recommendation
- Session duration recommendation
- Weekly frequency recommendation
- Recovery guidance
- Protein guidance
- Hydration guidance
- Final fitness report

The project is designed to present the outputs of the ANN, Fuzzy Logic, and recommendation modules through a Streamlit interface. Intelligent_Fitness_Recommendat…

---

# 14. Project Structure

```text
Intelligent-Fitness-Recommendation-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── gym_members_exercise_tracking.csv
│
├── models/
│   ├── fitness_ann_model.keras
│   ├── fitness_scaler.pkl
│   ├── feature_columns.pkl
│   └── gender_encoder.pkl
│
├── notebooks/
│   └── Intelligent_Fitness_Recommendation_System.ipynb
│
└── outputs/
```

---

# 15. Technologies Used

| Category | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Neural Network | TensorFlow / Keras |
| Fuzzy Logic | scikit-fuzzy |
| Visualization | Matplotlib, Seaborn |
| Web Application | Streamlit |
| Model Storage | Joblib, Keras |
| Development | VS Code / Jupyter Notebook |
| Version Control | Git / GitHub |

The project technology stack is based on Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, scikit-fuzzy, Streamlit, Joblib/Keras, and Git/GitHub. Intelligent_Fitness_Recommendat…

---

# 16. Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Intelligent-Fitness-Recommendation-System.git
```

Navigate into the project:

```bash
cd Intelligent-Fitness-Recommendation-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 17. Run the Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

---

# 18. Example Workflow

### Standard Mode

```text
Enter User Information
        |
        v
Calculate BMI
        |
        v
Prepare ANN Features
        |
        v
Predict Calories
        |
        v
Apply Fuzzy Logic
        |
        v
Determine Intensity
        |
        v
Apply Goal-Based Rules
        |
        v
Generate Recommendation
```

### Approximation Mode

```text
Approximate Height
        +
Body Build
        +
Activity Level
        +
Workout Experience
        |
        v
Estimate BMI / Weight
        |
        v
ANN + Fuzzy Logic
        |
        v
Personalized Recommendation
```

---

# 19. Output

The application generates a fitness report containing information such as:

```text
BMI
Fitness Level
Workout Intensity
Intensity Score
Estimated Calories
Workout Type
Workout Duration
Weekly Frequency
Recovery Days
Protein Guidance
Hydration Guidance
Goal-Specific Guidance
```

The project synopsis describes the expected fitness report as including BMI, workout intensity, duration, workout type, calorie-related output, protein guidance, and fitness guidance. Fitness_Recommendation_Project\_…

---

# 20. Author

**Adesh Vishwakarma**

Project: **Intelligent Fitness Recommendation System using Soft Computing Techniques**
```

### After saving `README.md`

Run these commands:

```bash
git add README.md
git commit -m "Add project documentation"
git push
```
