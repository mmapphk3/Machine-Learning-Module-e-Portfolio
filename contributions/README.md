# Project Work

## This section details individual contributions to project work in unit 6 and unit 11

### Individual Contribution to Group Project: Analyzing High-End Airbnb Customer Preferences

#### As a key member of our analytics team, I played a crucial role in shaping the direction and execution of our project aimed at understanding the preferences of high-spending Airbnb customers. Our collective brainstorming session on May 26th was instrumental in defining our main business question: "What are the customers (booking the top range of Airbnb reservations) looking for?" During this session, each group member actively listened and contributed ideas, fostering a collaborative environment that enabled us to reach a consensus. We also formulated follow-up questions focusing on the top booking areas, potential geographic patterns, and amenities influencing booking preferences. My ability to facilitate open discussions and encourage diverse perspectives was pivotal in ensuring our team made well-rounded decisions.

#### On May 31st, as we each undertook separate exploratory data analyses (EDA), I focused specifically on K-means clustering to profile high-spending Airbnb customers. My analysis targeted the top 25% of spenders in Manhattan and Brooklyn, with the aim of understanding the variables influencing their spending behaviors. After meticulously cleaning the data by removing null values, I utilized the elbow method to determine the optimal number of clusters, resulting in the identification of four distinct customer segments. My findings revealed that Cluster 2, representing the highest spenders, preferred entire homes or apartments in Manhattan for extended stays, highlighting a demand for space and luxury. The insights derived from this clustering analysis were critical in shaping our marketing strategies, allowing us to tailor our approach to meet the needs of premium customers effectively.

#### Throughout the project, my contributions extended beyond data analysis. After each meeting, I actively participated in discussions to outline our next steps, ensuring our project remained organized and on track. In our final meeting, I completed the code for the analysis and meticulously documented my section in the appendix. My clustering methodology and proposed marketing strategies were integral parts of our final analytical report. The cohesive effort from all team members, combined with our ability to meet every mini-deadline we set, resulted in a comprehensive and well-rounded report. Our collective agreement on the final version before submission underscored the effectiveness of our teamwork and the thoroughness of our analysis.



### Evaluation of Neural Network Model for Object Recognition: Technical Insights and Presentation Experience

#### The process of creating and delivering this presentation on the Neural Network Model for Object Recognition was an invaluable learning experience, both technically and from a presentation skills perspective. Technically, I gained a deep understanding of the components and architecture of a convolutional neural network (CNN), specifically tailored for object recognition using the CIFAR-10 dataset. This involved hands-on experience with data preprocessing techniques such as normalization and one-hot encoding, understanding the significance of validation sets, and the intricate process of training the model while preventing overfitting. Additionally, I became proficient in implementing crucial elements of CNNs such as convolutional layers, max pooling layers, dropout layers, and activation functions like ReLU and softmax. The practical application of these concepts solidified my knowledge and underscored the importance of each component in achieving a well-performing model (Brownlee, 2022).


<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/TrainingandValidationLoss.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig1: Training and Validation Accuracy and Loss</p>



#### Fig1 shows the training and validation accuracy and loss over 25 epochs. This illustrates the model's learning process, with the training accuracy steadily increasing while the validation accuracy stabilizes, indicating potential overfitting after a certain point. The validation and training loss graphs further highlight this trend, with the validation loss beginning to increase after 10 epochs, suggesting diminishing returns on generalization performance. Through this, I learned the critical importance of monitoring both training and validation metrics to detect overfitting and adjust the training process accordingly to ensure better generalization (Lehn, 2023).


<img src="https://github.com/mmapphk3/Machine-Learning-Module-e-Portfolio/raw/main/assets/images/CNNResults.png" alt="Corr Plot 1" style="width: 100%;">
<p>Fig2: Feature maps from the First Convolutional Layer </p>

#### From a presentation perspective, the experience honed my ability to clearly communicate complex technical ideas through visual aids. Creating a PowerPoint required careful planning to ensure each slide succinctly conveyed key points. Visual elements like pie charts, line graphs, diagrams, bar charts, and heatmaps illustrated concepts and model performance effectively. For example, Fig2 shows feature maps from the first convolutional layer, highlighting how different filters capture various features. Recording the presentation improved my public speaking skills, helping me articulate thoughts clearly and maintain a steady pace. This experience underscored the importance of combining technical proficiency with effective communication to present complex information accessibly.






References

- Brownlee, J. (2022) Object classification with cnns using the Keras Deep Learning Library, MachineLearningMastery.com. Available at: 
  https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/ (Accessed: 14 July 2024).

- Lehn, F. vom (2023) Interpreting training/validation accuracy and loss, Medium. Available at: https://medium.com/@frederik.vl/interpreting- 
  training-validation-accuracy-and-loss-cf16f0d5329f (Accessed: 14 July 2024).
