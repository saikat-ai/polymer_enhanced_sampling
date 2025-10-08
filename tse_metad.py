## Import importants libraries
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyemma
from pyemma.util.contexts import settings

# Load order parameter time series data
op1 = np.loadtxt('OP1_Rg')[:, 1]
op2 = np.loadtxt('OP2_end_to_end_distance')[:, 1]
op3 = np.loadtxt('OP3_num_of_water')[:, 1]

# Combine all the time series data
#data = np.vstack((op1, op2, op3)).T

# Normalize the data
#scaler = MinMaxScaler()
#data_norm = scaler.fit_transform(data)

A = msm.metastable_sets[2] ## define source state
B = msm.metastable_sets[4] ## define sink state
flux = pyemma.msm.tpt(msm, A, B) ## calculate flux committor_probabilities = flux.committor[dtrajs_concatenated]

# Combine data with committor probabilities
combined_data = np.column_stack((op1, op2, op3, op4, committor_probabilities))
combined_df = pd.DataFrame(combined_data, columns=["COMDIST", "DIHEDRAL", "SASA", "Number_of_Water", "Committor_Probability"])

# Committor probability range
committor_range = (0.40, 0.60)  
filtered_df = combined_df[(combined_df["Committor_Probability"] >= committor_range[0]) & (combined_df["Committor_Probability"] <= committor_range[1])]
print(f"Number of data points in the range {committor_range}: {len(filtered_df)}")

X = filtered_df[["Rg", "d", "Nw"]]
y = filtered_df["Committor_Probability"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Apply Elastic Net with Cross-Validation
elastic_net_cv = ElasticNetCV(cv=3, l1_ratio=[0.1, 0.5, 0.9], max_iter=1000)  # Tune both alpha and l1_ratio
elastic_net_cv.fit(X_train, y_train)

# Best alpha and l1_ratio found by cross-validation
best_alpha = elastic_net_cv.alpha_
best_l1_ratio = elastic_net_cv.l1_ratio_
print(f"Best alpha selected by cross-validation: {best_alpha}")
print(f"Best l1_ratio selected by cross-validation: {best_l1_ratio}")

# Train final Elastic Net model with the best hyperparameters
elastic_net = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=1000)
elastic_net.fit(X_train, y_train)

# Get feature coefficients
feature_names = X.columns
coefficients = elastic_net.coef_

# Display coefficients
print("Elastic Net Regression Feature Coefficients:")
print(coef_df)
