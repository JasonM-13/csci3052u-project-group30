# Intial Exploratory Data Analysis

Basic tests were run to measure and see the struture of the data set. Sample of input and output reviews were looked at.

The label distribution in the training set is perfectly balanced. Each of the three sentiment classes contains 3,333 examples, representing 33.33% of the training data, therefore no apparent class imbalance in the training set so this balanced distribution provides a consistent basis for comparing model performance across the three sentiment classes.

Overlap was checked between the training and test sets, and no identical reveiw sentences were found so the proves no evidence of exact duplicate reviews causing train/test data leakage.

It was found that 17.2% of the training data or 1717 training reviews contained repeated punctutation which appears to reflect the informal nature of user generated Cantonese resturant reveiws than invalid data.

Overall, the initial EDA indicates that the dataset is well structured and has a balanced distribution across the three sentiment classes. The absence of exact train/test sentence overlap is also useful for reducing concerns about data leakage.