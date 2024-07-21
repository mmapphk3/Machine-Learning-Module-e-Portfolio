# Development 

## This section includes artefacts demonstrating development over the duration of the module.

### Collaborative Discussion 1

#### Reflecting on Industry 4.0 and the Equifax data breach of 2017 has deepened my understanding of the complex issues faced by machine learning professionals. Industry 4.0, with its integration of IoT, AI, robotics, and big data analytics, transformed the financial sector by enhancing data analytics, blockchain technology, cryptocurrencies, automation, and cybersecurity. Financial institutions utilized these technologies to offer personalized services, manage risks, detect fraud in real-time, and streamline operations. However, the Equifax breach, which exposed the personal information of approximately 147 million people, underscored the critical importance of robust cybersecurity measures (Schwab, 2016, Fruhlinger, 2020).

#### Legally, the breach emphasized the need for stringent regulations to ensure companies prioritize data protection, as evidenced by the substantial fines and compensation costs faced by Equifax. Socially, it eroded public trust and demonstrated the broad impact of information system failures on consumer confidence. Ethically, it raised concerns about corporate responsibility and the moral implications of neglecting data protection. Professionally, the incident highlighted the necessity for continuous learning and adaptation in machine learning and cybersecurity. This breach taught me that legal, social, ethical, and professional considerations are crucial for guiding responsible and effective practices in the era of Industry 4.0 (Fruhlinger, 2020).

### Exploratory Data Analysis

#### Exploratory Data Analysis (EDA) involves key steps to understand and prepare a dataset for machine learning. The first step is exploring the dataset's features, looking at variable types, distributions, and summary statistics. This helps identify central tendencies, variances, and potential correlations between features. It's also crucial to spot anomalies, such as missing values, outliers, or inconsistent data entries, as they can affect analysis and model performance. Using visual inspection, statistical tests, and domain knowledge helps in effectively detecting and handling these anomalies (panData, 2023).

#### Visual analysis of the dataset plays a significant role in EDA, as it helps in uncovering patterns, trends, and relationships that are not immediately apparent through numerical analysis. Visualisations like histograms, scatter plots, and correlation matrices allow for a comprehensive visual inspection of the data (panData, 2023). Preparing the dataset for machine learning involves cleaning the data by handling missing values and outliers, transforming variables if necessary, and encoding categorical variables. This step ensures that the dataset is in an optimal format for feeding into machine learning algorithms, thus enhancing the accuracy and efficiency of the models built (Ray, 2023).

### Experiment Analysis: Correlation and Covariance

![Corr Plot 1](https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/blob/d5853a608cdd8edcf605acdf4d9fb78a8b848b65/assets/images/Graph1.png)
Fig1: Correlation Plot 1


![Corr Plot 2](assets/images/Graph2.png)
Fig2: Correlation Plot 2

#### In a recent experiment, I explored the relationship between two variables by examining how changes in their variability affect correlation and covariance. The goal was to gain a deeper understanding of these statistical measures through practical application.

#### Initially, I generated two datasets using a normal distribution. The first dataset, `data1`, was created with a mean of 100 and a standard deviation of 20. For the second dataset, `data2`, I added `data1` to another normally distributed set of values with a mean of 50 and a standard deviation of 10. This resulted in the formula: `data2 = data1 + (10 * randn(1000) + 50)`. After calculating the covariance matrix and Pearson's correlation coefficient for these datasets, I visualized the relationship with a scatter plot.

#### The results were quite revealing. The mean and standard deviation for `data1` were approximately 100 and 20, respectively, while for `data2`, they were around 150 and 22.36. The covariance between `data1` and `data2` was found to be 390.53, indicating a strong relationship. The Pearson's correlation coefficient was 0.888, suggesting a high positive correlation.

#### To investigate the effect of increased variability, I altered `data2` by changing the formula to `data2 = data1 + (20 * randn(1000) + 50)`, effectively doubling the standard deviation component. This adjustment aimed to see how a broader spread in the data would impact the correlation and covariance. The scatter plot for the modified datasets showed a more dispersed distribution compared to the initial plot.

#### Upon recalculating the statistical measures for the modified datasets, the covariance decreased, and the Pearson's correlation coefficient dropped, indicating a weaker linear relationship. This outcome illustrated that as the variability in `data2` increased, the strength of the correlation between `data1` and `data2` diminished. The dispersion in the scatter plot confirmed this observation, displaying a broader spread of data points.

#### In summary, this experiment demonstrated the significant impact of variability on correlation and covariance. By increasing the standard deviation in `data2`, the correlation with `data1` weakened, highlighting the sensitivity of these measures to changes in data variability. This practical exploration provided valuable insights into the behavior of statistical relationships in the presence of varying data spreads.

### Experiment Analysis: Linear Regression

### Linear Regression Analysis

#### In a recent analysis, I applied linear regression to understand the relationship between two variables. The independent variable `x` consisted of the values [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6], and the dependent variable `y` consisted of the values [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86].

#### Using the `stats.linregress` method from the `scipy` library, I calculated the slope, intercept, r-value, p-value, and standard error for the linear regression. The slope and intercept were used to create a function that computes the predicted y-values based on the x-values. This function, defined as `myfunc`, was then applied to the x-values to generate the corresponding y-values for the regression line.

#### To visualize the relationship, I plotted the original scatter plot of the x and y values and superimposed the linear regression line on the plot. The linear regression analysis revealed the direction and strength of the linear relationship between the variables, providing insights into how the dependent variable `y` changes with the independent variable `x`.

#### In summary, the linear regression model helped illustrate the statistical relationship between the two variables, highlighting the trend and potential predictive capability of the independent variable on the dependent variable.

#### Additionally, I used the linear regression model to predict values. By defining a function myfunc that uses the calculated slope and intercept, I was able to predict the dependent variable y for a given value of the independent variable x. For example, using this function, I predicted the value of y when x is 10, resulting in a predicted value of approximately 85.58. This demonstrates the practical application of the linear regression model for making predictions based on the established relationship between the variables.

#### By increasing both the `x` and `y` variables by 20%, the linear regression model was recalibrated to reflect the new data points. This adjustment had a notable impact on the prediction for `x=10`. In the original dataset, the predicted value for `x=10` was approximately 85.58. After increasing the data points by 20%, the prediction for `x=10` shifted to a higher value due to the overall increase in the `y` values. This change demonstrates how scaling data impacts the regression coefficients, leading to a proportional adjustment in predicted outcomes. The new regression line, plotted against the modified data, illustrates this upward shift, indicating a stronger positive relationship as the data values increase.

### Analysis of Similarity in Pathological Test Results Using the Jaccard Coefficient"

#### Based on the calculated Jaccard coefficients, we can draw conclusions about the similarity in pathological test results among the pairs of individuals.

#### Jack and Mary have a Jaccard coefficient of 0.43, indicating a moderate level of similarity in their pathological test results. This means that approximately 43% of their attributes match when ignoring 'A' (absent) values.

#### Jack and Jim have a Jaccard coefficient of 0.67, suggesting a higher level of similarity in their test results compared to the other pairs. About 67% of their attributes are similar, making Jack and Jim the most similar pair among the three.

#### Jim and Mary, with a Jaccard coefficient of 0.14, exhibit the lowest level of similarity in their pathological test results. Only about 14% of their attributes match, making them the least similar pair.

#### In summary, Jack and Jim share the most similar pathological profiles, while Jim and Mary have the least similarity. This information can be valuable for medical diagnosis, treatment planning, and understanding the spread or characteristics of a disease within this group of individuals.

### Perceptron Activities

#### In this exercise, a simple perceptron model is implemented to understand its behavior with different input values and weights. The perceptron performs a weighted sum of the inputs and applies a step function to produce a binary output. Initially, with inputs of [45, 25] and weights of [0.7, 0.1], the weighted sum is 34, resulting in an output of 1, since the sum exceeds the threshold of 1. When the weights are changed to [-0.7, 0.1], the weighted sum becomes -29, leading to an output of 0, as the sum does not meet the threshold. This demonstrates how the perceptron's decision boundary is influenced by the weights, affecting whether the weighted sum meets the threshold for a given set of inputs. By altering the weights, the perceptron's classification output changes, illustrating the fundamental mechanism of binary classification in neural networks and the impact of weight adjustments on model behavior.

### Training a Simple Perceptron for Binary Classification of Logical AND Operation

#### In this exercise, a simple perceptron model was implemented to perform binary classification using a logical AND function. The inputs were defined as a matrix with four instances representing all possible combinations of two binary values (0,0), (0,1), (1,0), and (1,1). Corresponding outputs were defined as a vector representing the expected output for an AND operation, where only (1,1) produces a 1, and all other combinations produce a 0. The perceptron was initialized with zero weights for the two inputs and a learning rate of 0.1.

#### The perceptron's activation function, a step function, returned 1 if the weighted sum of inputs was greater than or equal to 1, and 0 otherwise. The model's output for a given input instance was calculated by taking the dot product of the inputs and weights, followed by applying the step function. During the training process, the perceptron adjusted its weights iteratively to minimize the error between predicted and actual outputs. This involved looping through each input instance, calculating predictions, determining errors, and updating weights if the error was non-zero.

#### The training continued until the total error across all instances was zero, indicating that the perceptron had learned the correct weights to classify the inputs accurately. After training, the final weights were used to classify new instances. The trained perceptron correctly classified all four input instances of the AND function: producing 0 for (0,0), (0,1), and (1,0), and 1 for (1,1). This demonstrated that the perceptron had successfully learned the logical AND operation, with the weights adjusted to correctly reflect the relationship between inputs and outputs for this binary classification task.

### Reflecting on AI Writers: Personal Development and Ethical Considerations

#### Writing this piece has deepened my understanding of the multifaceted implications of AI-generated writing, particularly in terms of legal and ethical issues. Legally, the use of AI in writing raises questions about intellectual property and authorship rights. Who owns the content generated by AI, and how should it be attributed? Ethically, the risk of bias and misinformation is significant, as AI systems often reflect the prejudices present in their training data. This can perpetuate stereotypes and spread false information if not carefully monitored. Additionally, the authenticity of creative works produced by AI poses ethical dilemmas about originality and the value of human creativity. Developing this piece has made me more aware of these complexities and the importance of ongoing scrutiny and regulation to ensure that AI tools are used responsibly and ethically in all writing contexts.

References:

- Schwab, K. (2016) The Fourth Industrial Revolution: What It Means and how to respond, World Economic Forum. Available at: https://www.weforum.org/agenda/2016/01/the-fourth-industrial-revolution-what-it-means-and-how-to-respond/ (Accessed: 21 July 2024). 

- Fruhlinger, J. (2020) Equifax Data Breach FAQ: What happened, who was affected, what was the impact?, CSO Online. Available at: https://www.csoonline.com/article/567833/equifax-data-breach-faq-what-happened-who-was-affected-what-was-the-impact.html (Accessed: 21 July 2024). 

- panData (2023) 🐼 Exploratory Data Analysis (EDA): Techniques and methods for Effective ML models, Medium. Available at: https://medium.com/@panData/eda-pipeline-techniques-and-methods-for-effective-machine-learning-models-506d8ff0d664 (Accessed: 21 July 2024).

- Ray, S. (2023) Data preprocessing: Handling missing values, outliers, and categorical data, Medium. Available at: https://medium.com/@sonalika_07/data-preprocessing-handling-missing-values-outliers-and-categorical-data-5ee7b4d5c783 (Accessed: 21 July 2024). 
