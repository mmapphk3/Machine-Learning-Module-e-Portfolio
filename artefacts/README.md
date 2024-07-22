# Development 

## This section includes artefacts demonstrating development over the duration of the module.

### Collaborative Discussion 1

#### Reflecting on Industry 4.0 and the 2017 Equifax data breach highlights complex challenges for machine learning professionals. Industry 4.0 integrates IoT, AI, robotics, and big data analytics, transforming the financial sector by enhancing data analytics, blockchain, cryptocurrencies, automation, and cybersecurity. These advancements enable personalised services, risk management, fraud detection, and streamlined operations. However, the Equifax breach, exposing the personal information of 147 million people, underscored the importance of robust cybersecurity (Schwab, 2016; Fruhlinger, 2020).

#### The breach highlighted the need for strict data protection laws, shown by the fines Equifax faced. It eroded public trust and showed how system failures impact consumer confidence. Ethically, it raised concerns about corporate responsibility and data protection neglect. Professionally, it emphasized the need for continuous learning in machine learning and cybersecurity. This incident taught me the importance of legal, social, ethical, and professional considerations in guiding responsible practices in Industry 4.0 (Fruhlinger, 2020).

### Exploratory Data Analysis

#### Exploratory Data Analysis (EDA) involves key steps to understand and prepare a dataset for machine learning. The first step is exploring the dataset’s features, looking at variable types, distributions, and summary statistics. This helps identify central tendencies, variances, and potential correlations between features. It’s also crucial to spot anomalies, such as missing values, outliers, or inconsistent data entries, as they can affect analysis and model performance. Using visual inspection, statistical tests, and domain knowledge helps in effectively detecting and handling these anomalies (panData, 2023).

#### Visual analysis is crucial in EDA, helping uncover patterns, trends, and relationships not obvious through numerical analysis. Visualizations like histograms, scatter plots, and correlation matrices provide a comprehensive inspection of the data (panData, 2023). Preparing the dataset for machine learning involves cleaning data, handling missing values and outliers, transforming variables if needed, and encoding categorical variables. This ensures the dataset is optimal for machine learning algorithms, enhancing model accuracy and efficiency (Ray, 2023).

### Experiment Analysis: Correlation and Covariance

<table>
<tr>
<td>
<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/Graph1.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig1: Correlation Plot 1</p>
</td>
<td>
<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/Graph2.png" alt="Corr Plot 2" style="width: 100%;">
<p>Fig2: Correlation Plot 2</p>
</td>
</tr>
</table>

#### I conducted an experiment to understand how changes in variability affect correlation and covariance between two variables. I generated two datasets using a normal distribution. The first dataset, data1, had a mean of 100 and a standard deviation of 20. For the second dataset, data2, I used the formula: data2 = data1 + (10 * randn(1000) + 50).

#### The mean and standard deviation for data1 were about 100 and 20, respectively. For data2, they were around 150 and 22.36. The covariance between data1 and data2 was 390.53, and the Pearson’s correlation coefficient was 0.888, indicating a strong positive relationship.

#### To see how increased variability affected these measures, I modified data2 to: data2 = data1 + (20 * randn(1000) + 50), doubling the standard deviation component. This resulted in a more dispersed scatter plot. Recalculating the statistical measures, I found that the covariance decreased and the Pearson’s correlation coefficient dropped, illustrating a weaker linear relationship as variability in data2 increased.


### Experiment Analysis: Linear Regression

<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/Data%201.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig3: Dataset</p>

#### In a recent analysis using linear regression, I explored the relationship between the independent variable x and the dependent variable y, as shown in Fig3. Using the `stats.linregress` method from the `scipy` library, I calculated the slope, intercept, r-value, p-value, and standard error. This enabled the creation of a function, `myfunc`, to compute predicted y-values based on x-values. Plotting the original scatter plot of x and y with the regression line revealed the direction and strength of their linear relationship, offering insights into how y changes with x (Debroy, 2023; Castiglioni, 2020). The model demonstrated practical predictive capabilities, with the function `myfunc` predicting y when x is 10, resulting in a value of approximately 85.58, highlighting the model’s utility in making predictions based on the established relationship.

<table>
<tr>
<td>
<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/LinReg1.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig4: Original Data </p>
</td>
<td>
<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/LinReg2.png" alt="Corr Plot 2" style="width: 100%;">
<p>Fig5: Modified Data</p>
</td>
</tr>
</table>

#### By increasing both x and y variables by 20%, the linear regression model was recalibrated. This adjustment significantly impacted the prediction for x=10. Originally, the predicted value for x=10 was approximately 85.58. After the 20% increase, the prediction for x=10 shifted to a higher value, reflecting the overall rise in y values. This change illustrates how scaling data affects regression coefficients, leading to proportional adjustments in predictions. The new regression line, plotted against the modified data, shows an upward shift, indicating a stronger positive relationship as data values increase.

### Analysis of Similarity in Pathological Test Results Using the Jaccard Coefficient

<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/TestResults.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig6: Pathological Test Results </p>

#### In this exercise, I analysed the pathological test results of three individuals—Jack, Mary, and Jim—to determine the similarity of their health profiles. By calculating the Jaccard coefficient for different pairs of individuals, I aim to understand the extent of shared characteristics in their test results. The Jaccard coefficient is a statistical measure used to compare the similarity and diversity of sample sets (Jadeja, 2022).

#### The Jaccard coefficient between Jack and Mary is 0.43, indicating 43% attribute similarity. Jack and Jim have a higher similarity, with a Jaccard coefficient of 0.67, meaning 67% of their attributes match. Jim and Mary show the least similarity, with a Jaccard coefficient of 0.14, indicating only 14% attribute similarity. These results can assist in medical diagnosis, treatment planning, and understanding disease characteristics within the group (Jadeja, 2022).

### Perceptron Activities

#### In this exercise, a simple perceptron model was tested with inputs [45, 25] and initial weights [0.7, 0.1]. This produced a weighted sum of 34, resulting in an output of 1, as it exceeded the threshold of 1. When the weights were changed to [-0.7, 0.1], the weighted sum became -29, leading to an output of 0. This exercise demonstrates how the perceptron’s classification output is influenced by weight adjustments, affecting whether the weighted sum meets the threshold for a given set of inputs, and illustrating the fundamental mechanism of binary classification in neural networks (Kılıç, 2023).

### Training a Simple Perceptron for Binary Classification of Logical AND Operation

#### In this exercise, a perceptron was implemented to perform binary classification using the logical AND function. The inputs included all combinations of two binary values: (0,0), (0,1), (1,0), and (1,1), with corresponding outputs (0, 0, 0, 1). The perceptron was initialised with zero weights and a learning rate of 0.1. Using a step function activation, it returned 1 if the weighted sum was ≥ 1, otherwise 0.

#### During training, the perceptron iteratively adjusted its weights to minimize errors by looping through each input instance, calculating predictions, determining errors, and updating weights if errors were non-zero. This process continued until the total error across all instances was zero, indicating that the perceptron had learned the correct weights to accurately classify the inputs. After training, the perceptron correctly classified all four input instances of the AND function: producing 0 for (0,0), (0,1), and (1,0), and 1 for (1,1). This demonstrated the perceptron’s successful learning of the logical AND operation (Viridi, 2023).

### Reflecting on AI Writers: Risks and Benefits of the use AI writers at different levels

#### Writing this piece has deepened my understanding of the implications of AI-generated writing, particularly legal and ethical issues. Legally, AI in writing raises questions about intellectual property and authorship rights, such as who owns the AI-generated content and how it should be attributed. Ethically, there's a significant risk of "bias" and "misinformation," as AI systems can reflect the prejudices in their training data, potentially "perpetuating stereotypes" and spreading "false" information. Additionally, the authenticity of AI-produced creative works poses ethical dilemmas about "originality" and the value of human "creativity." This experience has highlighted the need for ongoing scrutiny and regulation to ensure AI tools are used responsibly and ethically in writing (Nython, 2024).

References:

- Schwab, K. (2016) The Fourth Industrial Revolution: What It Means and how to respond, World Economic Forum. Available at: https://www.weforum.org/agenda/2016/01/the-fourth-industrial-revolution-what-it-means-and-how-to-respond/ (Accessed: 21 July 2024). 

- Fruhlinger, J. (2020) Equifax Data Breach FAQ: What happened, who was affected, what was the impact?, CSO Online. Available at: https://www.csoonline.com/article/567833/equifax-data-breach-faq-what-happened-who-was-affected-what-was-the-impact.html (Accessed: 21 July 2024). 

- panData (2023) 🐼 Exploratory Data Analysis (EDA): Techniques and methods for Effective ML models, Medium. Available at: https://medium.com/@panData/eda-pipeline-techniques-and-methods-for-effective-machine-learning-models-506d8ff0d664 (Accessed: 21 July 2024).

- Ray, S. (2023) Data preprocessing: Handling missing values, outliers, and categorical data, Medium. Available at: https://medium.com/@sonalika_07/data-preprocessing-handling-missing-values-outliers-and-categorical-data-5ee7b4d5c783 (Accessed: 21 July 2024).

- Debroy, S. (2023) Simple linear regression in python, Medium. Available at: https://medium.com/@shuv.sdr/simple-linear-regression-in-python-a0069b325bf8 (Accessed: 21 July 2024). 

- Castiglioni, A. (2020) Linear regression in python from scratch with scipy, statsmodels, sklearn, Medium. Available at: https://medium.com/analytics-vidhya/linear-regression-in-python-from-scratch-with-scipy-statsmodels-sklearn-da8e373cc89b (Accessed: 21 July 2024). 

- Jadeja, M. (2022) Jaccard similarity made simple: A beginner’s guide to data comparison, Medium. Available at: https://medium.com/@mayurdhvajsinhjadeja/jaccard-similarity-34e2c15fb524 (Accessed: 21 July 2024).

- Kılıç, İ. (2023) Perceptron model: The Foundation of Neural Networks, Medium. Available at: https://medium.com/@ilyurek/perceptron-model-the-foundation-of-neural-networks-4db25b0148d#:~:text=Learning%20in%20the%20Perceptron%20Model&text=This%20entails%20the%20following%20steps,function%20to%20produce%20an%20output. (Accessed: 21 July 2024).

- Viridi, S. (2023) Perceptron for binary classifier with unit step activation function, Medium. Available at: https://medium.com/@6unpnp/perceptron-for-binary-classifier-with-unit-step-activation-function-b9b143ae391f (Accessed: 21 July 2024).

- Nython, P. (2024) The ethical implications of using AI in content creation, Medium. Available at: https://medium.com/@Phannuman/the-ethical-implications-of-using-ai-in-content-creation-ccd81b26fabc (Accessed: 21 July 2024). 
