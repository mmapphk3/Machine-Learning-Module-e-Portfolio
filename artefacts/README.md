# Development 

## This section includes artefacts demonstrating development over the duration of the module.

### Collaborative Discussion 1

#### Reflecting on Industry 4.0 and the 2017 Equifax data breach highlights the complex challenges machine learning professionals face. Industry 4.0, integrating IoT, AI, robotics, and big data analytics, has transformed the financial sector by enhancing data analytics, blockchain technology, cryptocurrencies, automation, and cybersecurity. These advancements enable personalized services, risk management, real-time fraud detection, and streamlined operations. However, the Equifax breach, exposing the personal information of 147 million people, underscored the critical importance of robust cybersecurity (Schwab, 2016; Fruhlinger, 2020).

#### Legally, the breach highlighted the need for stringent regulations to prioritize data protection, demonstrated by the substantial fines faced by Equifax. Socially, it eroded public trust and showed the broad impact of information system failures on consumer confidence. Ethically, it raised concerns about corporate responsibility and the moral implications of neglecting data protection. Professionally, it emphasized the necessity for continuous learning and adaptation in machine learning and cybersecurity. This breach taught me that legal, social, ethical, and professional considerations are crucial for guiding responsible and effective practices in Industry 4.0 (Fruhlinger, 2020).

### Exploratory Data Analysis

#### Exploratory Data Analysis (EDA) involves key steps to understand and prepare a dataset for machine learning. The first step is exploring the dataset's features, looking at variable types, distributions, and summary statistics. This helps identify central tendencies, variances, and potential correlations between features. It's also crucial to spot anomalies, such as missing values, outliers, or inconsistent data entries, as they can affect analysis and model performance. Using visual inspection, statistical tests, and domain knowledge helps in effectively detecting and handling these anomalies (panData, 2023).

#### Visual analysis of the dataset plays a significant role in EDA, as it helps in uncovering patterns, trends, and relationships that are not immediately apparent through numerical analysis. Visualisations like histograms, scatter plots, and correlation matrices allow for a comprehensive visual inspection of the data (panData, 2023). Preparing the dataset for machine learning involves cleaning the data by handling missing values and outliers, transforming variables if necessary, and encoding categorical variables. This step ensures that the dataset is in an optimal format for feeding into machine learning algorithms, thus enhancing the accuracy and efficiency of the models built (Ray, 2023).

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

#### In a recent experiment, I explored the relationship between two variables by examining how changes in their variability affect correlation and covariance. The goal was to gain a deeper understanding of these statistical measures through practical application.

#### Initially, I generated two datasets using a normal distribution. The first dataset, `data1`, was created with a mean of 100 and a standard deviation of 20. For the second dataset, `data2`, I added `data1` to another normally distributed set of values with a mean of 50 and a standard deviation of 10. This resulted in the formula: `data2 = data1 + (10 * randn(1000) + 50)`. After calculating the covariance matrix and Pearson's correlation coefficient for these datasets, I visualized the relationship with a scatter plot.

#### The results were quite revealing. The mean and standard deviation for `data1` were approximately 100 and 20, respectively, while for `data2`, they were around 150 and 22.36. The covariance between `data1` and `data2` was found to be 390.53, indicating a strong relationship. The Pearson's correlation coefficient was 0.888, suggesting a high positive correlation.

#### To investigate the effect of increased variability, I altered `data2` by changing the formula to `data2 = data1 + (20 * randn(1000) + 50)`, effectively doubling the standard deviation component. This adjustment aimed to see how a broader spread in the data would impact the correlation and covariance. The scatter plot for the modified datasets showed a more dispersed distribution compared to the initial plot.

#### Upon recalculating the statistical measures for the modified datasets, the covariance decreased, and the Pearson's correlation coefficient dropped, indicating a weaker linear relationship. This outcome illustrated that as the variability in `data2` increased, the strength of the correlation between `data1` and `data2` diminished. The dispersion in the scatter plot confirmed this observation, displaying a broader spread of data points.

### Experiment Analysis: Linear Regression

<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/blob/33373a719527207159efd00e25fd07124a66cc46/assets/images/Data%201.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig3: Dataset</p>

#### In a recent analysis, I used linear regression to explore the relationship between the independent variable x and the dependent variable y as in Fig3. Utilizing the `stats.linregress` method from the `scipy` library, I calculated the slope, intercept, r-value, p-value, and standard error for the linear regression. This allowed me to create a function, `myfunc`, which computes predicted y-values based on the x-values. Plotting the original scatter plot of x and y and superimposing the linear regression line revealed the direction and strength of the linear relationship, offering insights into how y changes with x (Debroy, 2023, Castiglioni, 2020).

#### Moreover, the linear regression model demonstrated practical predictive capabilities. By defining the function `myfunc` with the calculated slope and intercept, I predicted x for given x values. For instance, predicting y when x is 10 resulted in a value of approximately 85.58. This highlights the model's utility in making predictions based on the established relationship between the variables.

<table>
<tr>
<td>
<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/blob/33ff56ad0e0af911c38422fc1d3a92703531f751/assets/images/LinReg1.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig4: Original Data </p>
</td>
<td>
<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/blob/33ff56ad0e0af911c38422fc1d3a92703531f751/assets/images/LinReg2.png" alt="Corr Plot 2" style="width: 100%;">
<p>Fig5: Modified Data</p>
</td>
</tr>
</table>

#### By increasing both the `x` and `y` variables by 20%, the linear regression model was recalibrated to reflect the new data points. This adjustment had a notable impact on the prediction for `x=10`. In the original dataset, the predicted value for `x=10` was approximately 85.58. After increasing the data points by 20%, the prediction for `x=10` shifted to a higher value due to the overall increase in the `y` values. This change demonstrates how scaling data impacts the regression coefficients, leading to a proportional adjustment in predicted outcomes. The new regression line, plotted against the modified data, illustrates this upward shift, indicating a stronger positive relationship as the data values increase.

### Analysis of Similarity in Pathological Test Results Using the Jaccard Coefficient"

<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/blob/main/assets/images/TestResults.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig6: Pathological Test Results </p>

#### In this exercise, I analyze the pathological test results of three individuals—Jack, Mary, and Jim—to determine the similarity of their health profiles. By calculating the Jaccard coefficient for different pairs of individuals, I aim to understand the extent of shared characteristics in their test results. The Jaccard coefficient is a statistical measure used to compare the similarity and diversity of sample sets (Jadeja, 2022).

#### The Jaccard coefficient between Jack and Mary is 0.43, indicating a moderate level of similarity, with approximately 43% of their attributes matching. Jack and Jim show a higher level of similarity, with a Jaccard coefficient of 0.67, indicating that 67% of their attributes are similar. On the other hand, Jim and Mary have the least similarity in their pathological profiles, with a Jaccard coefficient of 0.14, meaning only 14% of their attributes match. These results can help in medical diagnosis, treatment planning, and understanding the spread or characteristics of a disease within this group of individuals (Jadeja, 2022).

### Perceptron Activities

#### In this exercise, a simple perceptron model is implemented to understand its behavior with different input values and weights. The perceptron performs a weighted sum of the inputs and applies a step function to produce a binary output. Initially, with inputs of [45, 25] and weights of [0.7, 0.1], the weighted sum is 34, resulting in an output of 1, since the sum exceeds the threshold of 1. When the weights are changed to [-0.7, 0.1], the weighted sum becomes -29, leading to an output of 0, as the sum does not meet the threshold. This demonstrates how the perceptron's decision boundary is influenced by the weights, affecting whether the weighted sum meets the threshold for a given set of inputs. By altering the weights, the perceptron's classification output changes, illustrating the fundamental mechanism of binary classification in neural networks and the impact of weight adjustments on model behavior (Kılıç, 2023).

### Training a Simple Perceptron for Binary Classification of Logical AND Operation

#### In this exercise, a simple perceptron model was implemented to perform "binary" classification using a logical AND function. The inputs were defined as a matrix with four instances representing all possible combinations of two binary values (0,0), (0,1), (1,0), and (1,1). Corresponding outputs were defined as a vector representing the expected output for an AND operation, where only (1,1) produces a 1, and all other combinations produce a 0. The perceptron was initialized with zero weights for the two inputs and a learning rate of 0.1 (Viridi, 2023).

#### The perceptron's "activation function", a step function, returned 1 if the weighted sum of inputs was greater than or equal to 1, and 0 otherwise. The model's output for a given input instance was calculated by taking the dot product of the inputs and weights, followed by applying the step function. During the training process, the perceptron adjusted its weights iteratively to minimize the error between predicted and actual outputs. This involved looping through each input instance, calculating predictions, determining errors, and updating weights if the error was non-zero (Viridi, 2023).

#### The training continued until the total error across all instances was zero, indicating that the perceptron had learned the correct weights to classify the inputs accurately. After training, the final weights were used to classify new instances. The trained perceptron correctly classified all four input instances of the AND function: producing 0 for (0,0), (0,1), and (1,0), and 1 for (1,1). This demonstrated that the perceptron had successfully learned the logical AND operation, with the weights adjusted to correctly reflect the relationship between inputs and outputs for this binary classification task (Viridi, 2023).

### Reflecting on AI Writers: Personal Development and Ethical Considerations

#### Writing this piece has deepened my understanding of the multifaceted implications of AI-generated writing, particularly in terms of legal and ethical issues. Legally, the use of AI in writing raises questions about intellectual property and authorship rights. Who owns the content generated by AI, and how should it be attributed? Ethically, the risk of "bias" and "misinformation" is significant, as AI systems often reflect the prejudices present in their training data. This can "perpetuate stereotypes" and spread "false" information if not carefully monitored. Additionally, the authenticity of creative works produced by AI poses ethical dilemmas about "originality" and the value of human "creativity". Developing this piece has made me more aware of these complexities and the importance of ongoing scrutiny and regulation to ensure that AI tools are used responsibly and ethically in all writing contexts (Nython, 2024).

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
