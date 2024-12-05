# Hidden Markov Models for Real Estate Market Trend Classification

This project explores the use of Hidden Markov Models (HMMs) to classify real estate market trends, focusing on identifying patterns of price increases and decreases (0: 'decreasing', 1: 'increasing'):. The data includes historical real estate transactions from 2007–2019, featuring variables such as property type, number of bedrooms, and median moving averages. Due to the absence of sequential time series data for individual properties, HMMs were leveraged as a preprocessing tool to uncover latent states, which were then used as features in various classification models.

### Data
- House Property Sales Time Series Data in Australian Capital Territory (ACT) from 2007-2019
- Data Source: https://www.kaggle.com/datasets/htagholdings/property-sales?select=raw_sales.csv

### Methodology
- **Hidden Markov Models**: HMMs were used to partition the data into 5 latent market states (e.g. decreasing, increasing). These states were then added as a new feature, `hidden_state`, to enhance the classification task. The optimal number of hidden states was determined using Akaike and Bayesian Information Criteria (AIC/BIC).
- **Feature Engineering**: Variables such as normalized prices, price movement, and price change were engineered to capture trends. Train/Test Data was split 80/20 before and after 2017 to avoid data leakage.
- **Classification Models**: After adding the `hidden_state` feature, various classification models (Logistic Regression, SVM, Decision Trees, Random Forest, and Gradient Boosting) were trained to predict price movement trends. The best-performing model, Gradient Boosting, achieved an **AUC of 0.77** on cross-validated test data.
- **Goal**: Classifies whether or not a property is expcted to increase or decrease ( >1%) from quarter to quarter
- **Libraries**: `hmmlearn`, `scikit-learn`, `pandas`, and `matplotlib` for model development and evaluation

While HMMs added a slight improvement in predictive performance, they highlighted the potential of uncovering latent market states for sequential and temporal data analysis. The inclusion of the `hidden_state` feature improved the Gradient Boosting model's AUC from 0.78 to 0.80 on the original test data.

## Future Directions
- **Multi-Class Clasifiction**: Incorporate additional classes for a more robust model with deeper insights:
  - 0: Rapid Decrease: (price change <= -3% or < 2σ within a relatively short time period t, e.g. a month)
  - 1: Steady Decrease: (-1% <= price change <= -0.5% over several months m)
  - 2: Rapid Increase
  - 3: Steady Increase
  - 4: Positive Spike: (weekly price change >=  5% or a rapid increase followed by a rapid decrease)
  - 5: Negative Spike
- **Expanded Data**: Incorporating property-specific historical data, such as square footage or neighborhood details
- **Advanced Algorithms**: Exploring XGBoost and RNNs for deeper insights into latent trends.
- - **Hybrid Models**: Combining HMMs with ARIMA or RNNs for improved sequential trend analysis and more specific for price prediction
