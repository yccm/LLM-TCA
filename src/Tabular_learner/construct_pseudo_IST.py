import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

def load_and_check_data(path):
    # Load data
    df = pd.read_csv(f"{path}/IST_syn.csv")
    df_train = df.iloc[:int(0.8*len(df))]
    df_test = df.iloc[int(0.8*len(df)):]

    # Get training data
    X_train = df.drop(columns=['Y0', 'Y1', 'A'])
    Y0_train = df['Y0']
    Y1_train = df['Y1']
    A_train = df['A']

    # Get test data
    X_test = df_test.drop(columns=['Y0', 'Y1', 'A'])
    Y0_test = df_test['Y0']
    Y1_test = df_test['Y1']
    A_test = df_test['A']

    # Print data shapes and samples for verification
    print("Training data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"Y0_train: {Y0_train.shape}")
    print(f"Y1_train: {Y1_train.shape}")
    print(f"A_train: {A_train.shape}")
    
    print("\nTest data shapes:")
    print(f"X_test: {X_test.shape}")
    print(f"Y0_test: {Y0_test.shape}")
    print(f"Y1_test: {Y1_test.shape}")
    print(f"A_test: {A_test.shape}")

    return X_train, Y0_train, Y1_train, A_train, X_test, Y0_test, Y1_test, A_test

# # Randomly assign treatment
X_train, Y0_train, Y1_train, A_train, X_test, Y0_test, Y1_test, A_test = load_and_check_data(path="dataset/IST/IST_tabular")



class DRLearner:
    def __init__(self):
        # Initialize outcome predictors and propensity model
        self.mu0_model = LinearRegression()
        self.mu1_model = LinearRegression()
        self.prop_model = LogisticRegression(max_iter=1000)
        self.final_model = LinearRegression()
        self.pseudo_outcomes = None
    
    def fit(self, X, A, Y):
        # Fit propensity score model
        self.prop_model.fit(X, A)
        prop_scores = self.prop_model.predict_proba(X)[:, 1]
        print("prop_scores", prop_scores)
        
        # Split data into treatment and control groups
        treat_idx = A == 1
        control_idx = A == 0
        
        # Fit outcome models
        self.mu0_model.fit(X[control_idx], Y[control_idx])
        self.mu1_model.fit(X[treat_idx], Y[treat_idx])
        
        # Calculate pseudo-outcomes using DR formula
        mu1_pred = self.mu1_model.predict(X)
        mu0_pred = self.mu0_model.predict(X)
        
        # DR pseudo-outcome formula
        pseudo_outcomes = mu1_pred - mu0_pred + \
            A * (Y - mu1_pred) / (prop_scores + 1e-7) - \
            (1 - A) * (Y - mu0_pred) / (1 - prop_scores + 1e-7)
        
        # Store pseudo-outcomes
        self.pseudo_outcomes = pseudo_outcomes
        
        print("pseudo_outcomes", pseudo_outcomes)
        # Fit final CATE model on pseudo-outcomes
        self.final_model.fit(X, pseudo_outcomes)

        
    
    def predict_cate(self, X):
        # Predict CATE using the final model
        return self.final_model.predict(X)

    def get_pseudo_outcomes(self):
        """Return the stored pseudo-outcomes from the last training."""
        if self.pseudo_outcomes is None:
            raise ValueError("Model hasn't been fitted yet. Call fit() first.")
        return self.pseudo_outcomes





print("Train DRLearner")

# Train DRLearner
drlearner = DRLearner()
Y_train = Y1_train * A_train + Y0_train * (1 - A_train)
drlearner.fit(X_train, A_train, Y_train)

# Evaluate on test set
test_cate = drlearner.predict_cate(X_test)
print("test_cate", test_cate)
true_test_cate = Y1_test - Y0_test
print("true_test_cate", true_test_cate)
# Calculate MSE for CATE estimation
cate_mse = ((test_cate - true_test_cate) ** 2).mean()
print(f"CATE MSE: {cate_mse:.8f}")

# Get the pseudo-outcomes
pseudo_outcomes = drlearner.get_pseudo_outcomes()
print("pseudo_outcomes.shape", pseudo_outcomes.shape)
# save the pseudo-outcomes into a csv file
pd.DataFrame(pseudo_outcomes).to_csv("dataset/IST/IST_tabular/pseudo_outcomes.csv", index=False)
