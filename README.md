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


<img width="300" height="225" alt="image" src="https://github.com/user-attachments/assets/b3b591b2-d8c8-40cd-914f-10b09c7ce21b" /> <img width="300" height="225" alt="image" src="https://github.com/user-attachments/assets/89e9b621-7008-41fe-9623-5b9b5841ca6a" />   <img width="300" height="225" alt="image" src="https://github.com/user-attachments/assets/d88813f9-81fd-45b4-a144-30ef86261502" />
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





<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5aba300c-1dd2-4e2c-bbbe-36130c36471b" />



### Key Decision Points
1. **Root Node**: Initial split based on trading volume
2. **Branching**: Subsequent splits based on stock price thresholds
3. **Leaf Nodes**: Final classification decisions with class distributions
4. **Purity Metrics**: Gini index values indicate split effectiveness

### Visual Representation
The following tree structure illustrates the classification logic:

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/28b7022c-dc23-4b35-bebd-0c5d5ebc7106" />


## Model Performance Analysis

### Classification Results Comparison

This section presents the comprehensive evaluation of the Decision Tree classifier before and after hyperparameter tuning. The analysis includes precision, recall, F1-scores, and confusion matrices to assess model effectiveness.


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a5b30633-424e-4125-9059-316673f58c53" />




####  Pre-Tuning Performance
The initial model shows strong overall performance with 90% accuracy. Key observations:
- **AAL**: Perfect classification (1.00 across all metrics)
- **ABT**: High performance with 0.95 precision and 0.93 F1-score
- **ACN**: Lower precision (0.70) but good recall (0.88)
- **AAPL**: Perfect precision but lower recall (0.33) due to limited samples

####  Post-Tuning Performance
After hyperparameter optimization with GridSearchCV:
- **Overall Accuracy**: Improved to 91%
- **ABBV**: Enhanced performance (0.93 F1-score)
- **ABC**: Improved recall (0.93)
- **ACN**: Zero predictions due to class imbalance
- **Trade-offs**: Some classes show precision-recall trade-offs after tuning




<img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/44387ec1-7bae-4a6a-af65-954967ad8038" />
   <img width="200" height="250" alt="image" src="https://github.com/user-attachments/assets/aedf6a3b-b294-415a-ab5a-cce4eb5f68f5" />   <img width="400" height="250" alt="image" src="https://github.com/user-attachments/assets/134b47a0-1618-40c7-b941-fcd5784f2670" />



## Confusion Matrix Analysis

### Training Data Results
The training confusion matrix demonstrates:
- **Diagonal Dominance**: Strong diagonal values indicate correct predictions
- **Class Separation**: Clear distinction between most stock classes
- **Minor Misclassifications**: Some confusion between similar stock patterns
- **Training Fit**: Model shows good learning from training data

### Testing Data Results
The testing confusion matrix reveals:
- **Generalization**: Model maintains performance on unseen data
- **Consistency**: Similar patterns to training matrix
- **Real-world Performance**: Good applicability to new data
- **Robustness**: Stable predictions across different stock classes





<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/45f3b33f-b26a-48ce-9b9a-9a3808b5fc9e" />    <img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/ed389b1b-896e-4e62-aaa3-5c442990ed4f" />




## Decision Boundary Visualization

### Model Decision Analysis
The decision boundary plot provides a visual representation of how the Decision Tree classifier separates different stock classes in the feature space. This visualization helps understand the model's classification logic and feature importance.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f0c2b5cf-1fe0-4f13-9727-8486b19ca408" />




###  Plot Interpretation
The decision boundary shows:
- **X-axis**: Stock Price (Primary feature for classification)
- **Y-axis**: Trading Volume (Secondary feature influencing decisions)
- **Colored Regions**: Different areas representing classified stock symbols
- **Boundary Lines**: Decision thresholds learned by the model


<img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/6533de51-60a6-4f35-b29c-09077b3dab3d" />



## Conclusion

This project successfully demonstrates the application of machine learning classification techniques for stock market analysis using Python. Through comprehensive data preprocessing, exploratory analysis, and model development, the Decision Tree classifier achieved **91% accuracy** in classifying different stocks based on price and volume features.



<div align="center">

🎉 **Enjoy Your Machine Learning** 🎉

If this project helped or inspired you,  
give it a ⭐ **Star** on GitHub!

**Built with precision ❤️ for the Engineering Community**  
**Happy Designing!** ✨

</div>


<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif" alt="Bottom Line" width="100%" />
</div>
