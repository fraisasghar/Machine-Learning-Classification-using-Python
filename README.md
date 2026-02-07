<div align="center">
  
  ![Project Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&height=200&section=header&text=Machine+Learning+Classification+Using+Python&fontSize=40&fontColor=ffffff&animation=twinkling&desc=Data+Analysis+%7C+Visualization+%7C+Model+Evaluation&descSize=18&descAlignY=70)

  [![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
  [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
  [![Seaborn](https://img.shields.io/badge/Seaborn-4B8BBE?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)
  [![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
  [![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

</div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" alt="Bottom Line" width="100%" />
</div>




## Introduction

This project implements machine learning classification techniques using Python to analyze and interpret data patterns. The primary focus is on applying Decision Tree algorithms to classify data based on key features, with comprehensive evaluation through various performance metrics and visualization tools.

The project demonstrates end-to-end machine learning workflow including data preprocessing, exploratory data analysis, model training, hyperparameter tuning, and performance evaluation using confusion matrices and classification reports.

<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/34513b4f-ebff-48bb-be00-49c905b96c42" />


## Objectives

1. **Data Processing**: Clean, preprocess, and prepare datasets for machine learning algorithms
2. **Exploratory Data Analysis**: Visualize data distributions and identify patterns
3. **Model Implementation**: Build and train Decision Tree classification models
4. **Performance Optimization**: Apply hyperparameter tuning using GridSearchCV
5. **Evaluation**: Assess model performance through various metrics and visualizations
6. **Visualization**: Create insightful plots for data and model analysis

## Dataset Overview

The project utilizes a dataset containing:
- **Stock Price**: Numerical values representing stock prices
- **Trading Volume**: Numerical values representing trading volumes
- **Stock Name**: Categorical labels for different stocks (AAL, AAPL, AAP, ABBV, ABC, ABT, ACN)

**Dataset Characteristics**:
- Total samples: 1,833 entries
- Features: 2 numerical features (Stock Price, Trading Volume)
- Target: 7 classes (stock names)
- Data types: Float64, Int64, Object


### Key Insights
- **Stock Price Range**: Shows significant variation from 25 to 105
- **Trading Volume**: Remains constant across all quartiles (0.05)
- **Data Distribution**: Stock prices display normal distribution while trading volumes show uniform distribution


<img width="325" height="225" alt="image" src="https://github.com/user-attachments/assets/b3b591b2-d8c8-40cd-914f-10b09c7ce21b" /> <img width="330" height="225" alt="image" src="https://github.com/user-attachments/assets/89e9b621-7008-41fe-9623-5b9b5841ca6a" />   <img width="330" height="225" alt="image" src="https://github.com/user-attachments/assets/d88813f9-81fd-45b4-a144-30ef86261502" />
<img width="1000" height="100" alt="image" src="https://github.com/user-attachments/assets/90d48823-8b1b-41b2-ba83-443cd2408ffd" />


## Decision Tree Analysis

### Model Architecture
The Decision Tree classifier was implemented to analyze stock classification based on trading volume and stock price. The tree structure shows how the model makes splitting decisions at each node to classify different stock symbols.

### Tree Structure Explanation
The decision tree uses the following splitting criteria:
- **Primary Split**: Trading volume threshold at 3,209,598
- **Secondary Splits**: Stock price thresholds at various levels (86.73, 45.14, etc.)
- **Gini Index**: Measures impurity at each node (lower values indicate better splits)
- **Samples**: Number of data points at each node
- **Value Array**: Distribution of stock classes [AAL, AAP, AAPL, ABBV, ABC, ABT, ACN]





<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2c163f89-19ae-4fcc-ae59-11af5621ae7f" />


### Key Decision Points
1. **Root Node**: Initial split based on trading volume
2. **Branching**: Subsequent splits based on stock price thresholds
3. **Leaf Nodes**: Final classification decisions with class distributions
4. **Purity Metrics**: Gini index values indicate split effectiveness

### Visual Representation
The following tree structure illustrates the classification logic:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/28b7022c-dc23-4b35-bebd-0c5d5ebc7106" />




