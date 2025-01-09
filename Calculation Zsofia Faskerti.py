import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import det, inv
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from scipy.optimize import Bounds

# Set the random seed for reproducibility
np.random.seed(1)
###################################################
#Calibration of pre-COVID force of mortality
file_path = 'AG2024data.xlsx' 
data = pd.read_excel(file_path, sheet_name=None)

# Extracting specific sheets for male and female mortality and exposure data
deathNLmale = data["DeathsNetherlandsMale"]
deathNLfemale = data["DeathsNetherlandsFemale"]
exposureNLmale = data["ExposuresNetherlandsMale"]
exposureNLfemale = data["ExposuresNetherlandsFemale"]
deathEUfemale = data["DeathsEuropeFemale"]
deathEUmale = data["DeathsEuropeMale"]
exposureEUmale = data["ExposuresEuropeMale"]
exposureEUfemale = data["ExposuresEuropeFemale"]

# Drop the first column of each DataFrame (likely age or year headers), and clean missing values
deathNLmale = deathNLmale.set_index(deathNLmale.columns[0]).dropna(how='all')
deathNLfemale = deathNLfemale.set_index(deathNLfemale.columns[0]).dropna(how='all')
deathEUmale = deathEUmale.set_index(deathEUmale.columns[0]).dropna(how='all')
deathEUfemale = deathEUfemale.set_index(deathEUfemale.columns[0]).dropna(how='all')
exposureNLmale = exposureNLmale.set_index(exposureNLmale.columns[0]).dropna(how='all')
exposureNLfemale = exposureNLfemale.set_index(exposureNLfemale.columns[0]).dropna(how='all')
exposureEUmale = exposureEUmale.set_index(exposureEUmale.columns[0]).dropna(how='all')
exposureEUfemale = exposureEUfemale.set_index(exposureEUfemale.columns[0]).dropna(how='all')

# Initialize parameters based on data (Dutch deviation only uses data [1983,2019])
D_male_eu = deathEUmale.values 
D_female_eu = deathEUfemale.values  
E_male_eu = exposureEUmale.values  
E_female_eu = exposureEUfemale.values  
D_male_nl = deathNLmale.values[:, 13:]
D_female_nl = deathNLfemale.values[:, 13:]
E_male_nl = exposureNLmale.values[:, 13:]
E_female_nl = exposureNLfemale.values[:, 13:]
#########################################
# Part 1, Europe parameters
def log_likelihood_A(A, B, K, D, E):
    mu = np.exp(A[:, None] + B[:, None] * K[None, :])
    log_likelihood_value = np.sum(D * (A[:, None] + B[:, None] * K[None, :]) - E * mu)
    return -log_likelihood_value

def log_likelihood_B(B, A, K, D, E):
    mu = np.exp(A[:, None] + B[:, None] * K[None, :])
    log_likelihood_value = np.sum(D * (A[:, None] + B[:, None] * K[None, :]) - E * mu)
    return -log_likelihood_value

def log_likelihood_K(K, A, B, D, E):
    mu = np.exp(A[:, None] + B[:, None] * K[None, :])
    log_likelihood_value = np.sum(D * (A[:, None] + B[:, None] * K[None, :]) - E * mu)
    return -log_likelihood_value

def optimize_parametersEU(A, B, K, D, E, tol=1e-4, max_iter=5000):
    n_ages = len(A)
    n_times = len(K)

    # Constraint for B: sum(B) = 1
    B_constraint = LinearConstraint(np.ones(n_ages), 1, 1)

    # Constraint for K: sum(K) = 0
    K_constraint = LinearConstraint(np.ones(n_times), 0, 0)

    # Iterative optimization
    for iteration in range(max_iter):
        # Step 1: Minimize with respect to A
        result_A = minimize(log_likelihood_A, A, args=(B, K, D, E), method='L-BFGS-B', tol=tol)
        A = result_A.x

        # Step 2: Minimize with respect to B with constraint sum(B) = 1
        result_B = minimize(log_likelihood_B, B, args=(A, K, D, E), method='SLSQP', constraints=B_constraint, tol=tol)
        B = result_B.x

        # Step 3: Minimize with respect to K with constraint sum(K) = 0
        result_K = minimize(log_likelihood_K, K, args=(A, B, D, E), method='SLSQP', constraints=K_constraint, tol=tol)
        K = result_K.x

        # Check for convergence (change in log-likelihood values)
        if result_A.success and result_B.success and result_K.success:
            if (np.abs(result_A.fun - result_B.fun) < tol and
                np.abs(result_B.fun - result_K.fun) < tol):
                print(f'Convergence achieved after {iteration + 1} iterations.')
                break
    else:
        print("Maximum iterations reached without full convergence.")

    return A, B, K

# Setting initial values
A_initial_male = np.log((D_male_eu.sum(axis=1) + 1e-10) / (E_male_eu.sum(axis=1) + 1e-10)) #Log of death rates for initial A
B_initial_male = np.ones(D_male_eu.shape[0]) / D_male_eu.shape[0] # Uniform initial B
K_initial_male = np.zeros(D_male_eu.shape[1]) # Initial K set to zero

A_initial_female = np.log((D_female_eu.sum(axis=1) + 1e-10) / (E_female_eu.sum(axis=1) + 1e-10)) #Log of death rates for initial A
B_initial_female = np.ones(D_female_eu.shape[0]) / D_female_eu.shape[0] # Uniform initial B
K_initial_female = np.zeros(D_female_eu.shape[1]) # Initial K set to zero

# Optimizing the parameters A, B and K
A_opt_male, B_opt_male, K_opt_male = optimize_parametersEU(A_initial_male, B_initial_male, K_initial_male, D_male_eu, E_male_eu)
A_opt_female, B_opt_female, K_opt_female = optimize_parametersEU(A_initial_female, B_initial_female, K_initial_female, D_female_eu, E_female_eu)

# Printing A and B values by age, and K values by year
ages = deathEUmale.index
years = deathEUmale.columns

print("\n A values by age:")
for age, A_male_value, A_female_value in zip(ages, A_opt_male, A_opt_female):
    print(f"Age {age}: A male = {A_male_value}, A female = {A_female_value}")

print("\n B values by age:")
for age, B_male_value, B_female_value in zip(ages, B_opt_male, B_opt_female):
    print(f"Age {age}: B male = {B_male_value}, B female = {B_female_value}")

print("\n K values by year:")
for year, K_male_value, K_female_value in zip(years, K_opt_male, K_opt_female):
    print(f"Year {year}: K male = {K_male_value}, K female = {K_female_value}")

#################################
#PART2, Dutch parameters

mu_male_eu = np.exp(A_opt_male[:, None]+B_opt_male[:, None]* K_opt_male[None, :])
mu_male_eu = mu_male_eu[:, 13:]
mu_female_eu = np.exp(A_opt_female[:, None]+B_opt_female[:, None]* K_opt_female[None, :])
mu_female_eu = mu_female_eu[:, 13:]

def log_likelihood_alpha(alpha, beta, kappa, D, E, mu_eu):
    mu = mu_eu * np.exp(alpha[:, None] + beta[:, None] * kappa[None, :])
    log_likelihood_value = np.sum(D * (alpha[:, None] + beta[:, None] * kappa[None, :]) - E * mu)
    return -log_likelihood_value

def log_likelihood_beta(beta, alpha, kappa, D, E, mu_eu):
    mu = mu_eu *(np.exp(alpha[:, None] + beta[:, None] * kappa[None, :]))
    log_likelihood_value = np.sum(D * (alpha[:, None] + beta[:, None] * kappa[None, :]) - E * mu)
    return -log_likelihood_value

def log_likelihood_kappa(kappa, alpha, beta, D, E, mu_eu):
    mu = mu_eu * (np.exp(alpha[:, None] + beta[:, None] * kappa[None, :]))
    log_likelihood_value = np.sum(D * (alpha[:, None] + beta[:, None] * kappa[None, :]) - E * mu)
    return -log_likelihood_value

def optimize_parametersNL(alpha, beta, kappa, D, E, mu_eu, tol=1e-4, max_iter=5000):
    n_ages = len(alpha)
    n_times = len(kappa)

    # Constraint for beta: sum(beta) = 1
    beta_constraint = LinearConstraint(np.ones(n_ages), 1, 1)

    # Constraint for kappa: sum(kappa) = 0
    kappa_constraint = LinearConstraint(np.ones(n_times), 0, 0)

    # Iterative optimization
    for iteration in range(max_iter):
        # Step 1: Minimize with respect to alpha
        result_alpha = minimize(log_likelihood_alpha, alpha, args=(beta, kappa, D, E, mu_eu), method='L-BFGS-B', tol=tol)
        alpha = result_alpha.x

        # Step 2: Minimize with respect to beta with constraint sum(beta) = 1
        result_beta = minimize(log_likelihood_beta, beta, args=(alpha, kappa, D, E, mu_eu), method='SLSQP', constraints=beta_constraint, tol=tol)
        beta = result_beta.x

        # Step 3: Minimize with respect to kappa with constraint sum(kappa) = 0
        result_kappa = minimize(log_likelihood_kappa, kappa, args=(alpha, beta, D, E, mu_eu), method='SLSQP', constraints=kappa_constraint, tol=tol)
        kappa = result_kappa.x

        # Check for convergence (change in log-likelihood values)
        if result_alpha.success and result_beta.success and result_kappa.success:
            if (np.abs(result_alpha.fun - result_beta.fun) < tol and
                np.abs(result_beta.fun - result_kappa.fun) < tol):
                print(f'Convergence achieved after {iteration + 1} iterations.')
                break
    else:
        print("Maximum iterations reached without full convergence.")

    return alpha, beta, kappa


# Setting initial values
alpha_initial_male = np.log((D_male_nl.sum(axis=1) + 1e-10) / (E_male_nl.sum(axis=1) + 1e-10)) # Log of death rates for initial alpha
beta_initial_male = np.ones(D_male_nl.shape[0]) / D_male_nl.shape[0] # Uniform initial beta
kappa_initial_male = np.zeros(D_male_nl.shape[1]) # Initial kappa set to zero

alpha_initial_female = np.log((D_female_nl.sum(axis=1) + 1e-10) / (E_female_nl.sum(axis=1) + 1e-10)) # Log of death rates for initial alpha
beta_initial_female = np.ones(D_female_nl.shape[0]) / D_female_nl.shape[0] # Uniform initial beta
kappa_initial_female = np.zeros(D_female_nl.shape[1]) # Initial kappa set to zero

# Optimizing the parameters alpha, beta and kappa
alpha_opt_male, beta_opt_male, kappa_opt_male = optimize_parametersNL(alpha_initial_male, beta_initial_male, kappa_initial_male, D_male_nl, E_male_nl, mu_male_eu)
alpha_opt_female, beta_opt_female,kappa_opt_female = optimize_parametersNL(alpha_initial_female, beta_initial_female, kappa_initial_female, D_female_nl, E_female_nl, mu_female_eu)

# Print alpha and beta values by age, and kappa values by year
ages = deathEUmale.index
years = deathEUmale.columns[13:]

print("\n Alpha values by age:")
for age, alpha_male_value, alpha_female_value in zip(ages, alpha_opt_male, alpha_opt_female):
    print(f"Age {age}: Alpha male = {alpha_male_value}, Alpha female = {alpha_female_value}")

print("\n Beta values by age:")
for age, beta_male_value, beta_female_value in zip(ages, beta_opt_male, beta_opt_female):
    print(f"Age {age}: Beta male = {beta_male_value}, Beta female = {beta_female_value}")

print("\n Kappa values by year:")
for year, kappa_male_value, kappa_female_value in zip(years, kappa_opt_male, kappa_opt_female):
    print(f"Year {year}: Kappa male = {kappa_male_value}, Kappa female = {kappa_female_value}")


###########################
#Part 3
#Optimizing parameters theta, a, c and covariance matrix C

# t = 1970,...,1982
# Setting Y_t+1 for t = 1970,...,1982
K_male_pre = K_opt_male[:14]
K_female_pre=K_opt_female[:14]
T_pre = len(K_male_pre)  #number of time periods (13 years)
Y_t_plus_1_pre = []# Initialize an empty list to store Y_t+1 values

# Calculating Y_t+1 for each year t = 1970, ..., 1981 (since we are comparing with t+1)
for t in range(T_pre - 1):
    delta_K_male_pre = K_male_pre[t+1] - K_male_pre[t]
    delta_K_female_pre = K_female_pre[t+1] - K_female_pre[t]
    Y_t_next = [delta_K_male_pre, delta_K_female_pre]
    Y_t_plus_1_pre.append(Y_t_next)

# t = 1983,...,2018
# Setting Y_t+1 for t = 1983,...,2018
K_male_after = K_opt_male[13:]
K_female_after=K_opt_female[13:]
T_after = len(K_male_after)  # number of time periods (37 years)
Y_t_plus_1_after = []
# Calculate Y_t+1 for each year t = 1983, ..., 2018 (since we are comparing with t+1)
for t in range(T_after-1):
    delta_K_male_after = K_male_after[t+1] - K_male_after[t]
    delta_K_female_after = K_female_after[t+1] - K_female_after[t]
    kappa_M_after = kappa_opt_male[t+1]
    kappa_F_after = kappa_opt_female[t+1]
    
    # Combine the differences into a 4-element vector
    Y_t_next = [delta_K_male_after, delta_K_female_after, kappa_M_after, kappa_F_after]
    Y_t_plus_1_after.append(Y_t_next)

# Create the matrix X_t
# t = 1970,...,1982
X_t_pre = np.array([
    [1, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0, 0]   
])

# t = 1983,...,2018
X_t_aftery = np.array([
    [1, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0], 
    [0, 0, 0, 0, 0, 1]  
], dtype=float)

X_t_after = []

for t in range(T_after-1):
    X_t_new = X_t_aftery.copy()
    X_t_new[2,2] = float(kappa_opt_male[t])
    X_t_new[3,3] = float(kappa_opt_female[t])
    X_t_after.append(X_t_new)

# Log-likelihood function to optimize Psi and C together
def log_likelihood_combined(params, Y_pre, X_pre, Y_after, X_after):
    Psi = params[:6]  # First 6 elements are Psi
    C_elements = params[6:]  # Remaining elements correspond to C (4x4 symmetric)

    # Constructing C matrix (assuming a 4x4 symmetric matrix)
    C = np.array([[C_elements[0], C_elements[1], C_elements[2], C_elements[3]],
                  [C_elements[1], C_elements[4], C_elements[5], C_elements[6]],
                  [C_elements[2], C_elements[5], C_elements[7], C_elements[8]],
                  [C_elements[3], C_elements[6], C_elements[8], C_elements[9]]])
    
    C_tilde = C[:2, :2]  # Submatrix of C (for pre-1983)

    # Calculating log likelihood for pre-1983
    Z_t_pre = [Y - X_pre @ Psi for Y in Y_pre]
    Z_pre = [np.outer(z,z) for z in Z_t_pre]
    pre_term = np.trace(inv(C_tilde) @ sum(Z_pre))
    log_likelihood_pre = -0.5 * pre_term - (13 / 2) * np.log(det(C_tilde)) - (13) * np.log(2 * np.pi)

    # Calculating log likelihood for post-1982
    Z_t_after = [Y - X @ Psi for Y, X in zip(Y_after, X_after)]
    Z_after = [np.outer(z,z) for z in Z_t_after]
    post_term = np.trace(inv(C) @ sum(Z_after))
    log_likelihood_post = -0.5 * post_term - (36 / 2) * np.log(det(C)) - (36 * 4 / 2) * np.log(2 * np.pi)

    # Total log likelihood (to minimize, so return negative)
    return -(log_likelihood_pre + log_likelihood_post)


# Joint optimization of both Psi and C
def optimize_psi_and_C(parameter, Y_pre, X_pre, Y_after, X_after, tol=1e-4, max_iter=5000):

    # Optimization using minimizing
    result = minimize(log_likelihood_combined, parameter, args=(Y_pre, X_pre, Y_after, X_after), method='BFGS', tol=tol, options={'maxiter': max_iter, 'disp': True})
    Opt_para=result.x

    return Opt_para

#Setting initial values for Psi and C
#Computing Theta (difference between consecutive K values)
theta_male = np.diff(K_opt_male)
theta_female = np.diff(K_opt_female) 
mean_theta_male = np.mean(theta_male)
mean_theta_female = np.mean(theta_female)

#Computing c (difference between consecutive K values)(assuming a in both cases are 1)
c_male = np.diff(kappa_opt_male)
c_female = np.diff(kappa_opt_female)
mean_c_male = np.mean(c_male)
mean_c_female = np.mean(c_female)
Psi_initial = [mean_theta_male,mean_theta_female, 1, 1, mean_c_male, mean_c_female]

#Initial guess for C (covariance matrix)
Z_t_after = np.array([Y - X @ Psi_initial for Y, X in zip(Y_t_plus_1_after, X_t_after)]) # Calculate the residuals Z_t for post-1982 period
C_empirical = np.cov(Z_t_after, rowvar=False) # Compute the empirical covariance matrix
print("Empirical initial guess for C:", C_empirical)

# Flatten the C matrix (for optimization, we only need the upper triangular part since it's symmetric)
C_initial = C_empirical[np.triu_indices(4)]
parameter = np.hstack([Psi_initial, C_initial])
result = optimize_psi_and_C(parameter, Y_t_plus_1_pre, X_t_pre, Y_t_plus_1_after, X_t_after)

Psi_optimized = result[:6]
C_optimized_elements = result[6:]

# Constructing the optimized C matrix
C_optimized = np.array([[C_optimized_elements[0], C_optimized_elements[1], C_optimized_elements[2], C_optimized_elements[3]],
                        [C_optimized_elements[1], C_optimized_elements[4], C_optimized_elements[5], C_optimized_elements[6]],
                        [C_optimized_elements[2], C_optimized_elements[5], C_optimized_elements[7], C_optimized_elements[8]],
                        [C_optimized_elements[3], C_optimized_elements[6], C_optimized_elements[8], C_optimized_elements[9]]])

# Optimized results
print("Optimized Psi:", Psi_optimized)
print("Optimized Covariance Matrix C:", C_optimized)

theta_opt_male=Psi_optimized[0]
theta_opt_female=Psi_optimized[1]
a_opt_male=Psi_optimized[2]
a_opt_female=Psi_optimized[3]
c_opt_male=Psi_optimized[4]
c_opt_female=Psi_optimized[5]

##########################################
#Closure of the parameters of pre-COVID force of mortality

#Beta parameters for age 91 to 120
# Known values of beta for ages 80 to 90
B_male_last_11 = B_opt_male[-11:]  # Last 11 values for male
B_female_last_11 = B_opt_female[-11:]  # Last 11 values for female
ln_B_known_male = np.log(B_male_last_11)
ln_B_known_female = np.log(B_female_last_11)

# Ages corresponding to the known beta values (80 to 90)
used_period = np.arange(80, 91)

y_bar = 85 # Average of the known ages
n = len(used_period) # Number of ages used in regression
sum_squared_dev = 110

# Calculating the regression weights w_k(x)
def regression_weights(x, ages_known, y_bar, sum_squared_dev,n):
    weights = (1/n) + ((ages_known - y_bar) * (x - y_bar)) / sum_squared_dev
    return weights

# Extrapolating beta values for ages 91 to 120
def extrapolate_B(ages_extrapolate, ln_beta_known, ages_known, y_bar, sum_squared_dev,n=11):
    beta_extrapolated = []
    
    for x in ages_extrapolate:
        # Calculate the regression weights for the current age x
        weights = regression_weights(x, ages_known, y_bar, sum_squared_dev,n)
        # Extrapolate ln(beta_x) as a weighted sum of ln(beta_k)
        ln_beta_x = np.sum(weights * ln_beta_known)
        # Convert back to beta_x
        beta_x = np.exp(ln_beta_x)
        beta_extrapolated.append(beta_x)
    
    return np.array(beta_extrapolated)

# Ages 91 to 120 for which we want to extrapolate beta values
X_tilde = np.arange(91, 121)

# Extrapolate beta values for ages 91 to 120
B_extrapolated_male = extrapolate_B(X_tilde, ln_B_known_male, used_period, y_bar, sum_squared_dev)
B_extrapolated_female = extrapolate_B(X_tilde, ln_B_known_female, used_period, y_bar, sum_squared_dev)
B_opt_male_total= np.hstack([B_opt_male, B_extrapolated_male])
B_opt_female_total= np.hstack([B_opt_female, B_extrapolated_female])
print("\n Extrapolated B values for ages 91 to 120:")
for age, B_male_value, B_female_value in zip(X_tilde, B_extrapolated_male, B_extrapolated_female):
    print(f"Age {age}: B male = {B_male_value}, B female = {B_female_value}")

#Determining A_hat for ages 91 to 120
# Logistic function
def L(x):
    return 1 / (1 + np.exp(-x))

# Inverse logistic function
def L_inv(x):
    return np.log(x / (1 - x))


def determine_A_hat(A_yk, B_yk, B_extrapolated, K_2019, y_k_values,n=11, y_bar=85, sum_squared_dev=110):
    A_hat_values = []
    
    for age, x in  enumerate(range(91, 121)):  # For ages 91 to 120
        weighted_sum = 0
        
        for k, y_k in enumerate(y_k_values):
            # Calculate the regression weight w_k(x)
            weight = regression_weights(x, y_k, y_bar, sum_squared_dev,n)
            # Compute the weighted inverse logistic term
            weighted_sum += weight * L_inv(np.exp(A_yk[k] + B_yk[k] * K_2019))
        
        # Using B_x for ages 91 to 120
        A_hat_x = np.log(L(weighted_sum)) - B_extrapolated[age] * K_2019 
        A_hat_values.append(A_hat_x)
    
    return A_hat_values


A_yk_male = A_opt_male[-11:]  # A values for ages 80-90
A_yk_female = A_opt_female[-11:]
B_yk_male = B_male_last_11    # B values for ages 80-90
B_yk_female = B_female_last_11
K_2019_male = K_opt_male[-1]  # K value for 2019
K_2019_female= K_opt_female[-1]
y_k_values = np.array([80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90])  # Age range 80-90

# Determining A_hat values for ages 91 to 120
A_extrapolated_male = determine_A_hat(A_yk_male, B_yk_male, B_extrapolated_male, K_2019_male, y_k_values)
A_extrapolated_female = determine_A_hat(A_yk_female, B_yk_female, B_extrapolated_female, K_2019_female, y_k_values)
A_opt_male_total=np.hstack([A_opt_male,A_extrapolated_male])
A_opt_female_total=np.hstack([A_opt_female,A_extrapolated_female])
print("Extrapolated A values for ages 91 to 120:")
for age, A_male_value, A_female_value in zip(X_tilde, A_extrapolated_male, A_extrapolated_female):
    print(f"Age {age}: A male = {A_male_value}, A female = {A_female_value}")

#Determining alpha for ages 91 to 120
alpha_90_male = alpha_opt_male[-1]
alpha_90_female=alpha_opt_female[-1]
# Linear extrapolation of alpha from age 90 to 120, with alpha_120 set to 0
alpha_extrapolated_male = alpha_90_male * (120 - X_tilde) / (120 - 90)
alpha_extrapolated_female = alpha_90_female * (120 - X_tilde) / (120 - 90)
alpha_opt_male_total=np.hstack([alpha_opt_male,alpha_extrapolated_male])
alpha_opt_female_total=np.hstack([alpha_opt_female,alpha_extrapolated_female])
print("Extrapolated Alpha values for ages 91 to 120:")
for age, alpha_value_male, alpha_value_female in zip(X_tilde, alpha_extrapolated_male, alpha_extrapolated_female):
    print(f"Age {age}: Alpha male = {alpha_value_male}, Alpha female = {alpha_value_female}")

#Calculating Beta_hat for ages 91 to 120
def determine_beta_hat(A_yk, A_extrapolated,B_yk, B_extrapolated,K_2019, y_k_values,alpha_yk,alpha_extrapolated,beta_yk,kappa_2019, n=11,y_bar=85, sum_squared_dev=110):
    beta_hat_values = []
    
    for age, x in enumerate(range(91, 121)):  # For ages 91 to 120
        weighted_sum = 0
        
        for k, y_k in enumerate(y_k_values):
            # Calculate the regression weight w_k(x)
            weight = regression_weights(x, y_k, y_bar, sum_squared_dev,n)
            # Compute the weighted inverse logistic term
            weighted_sum += weight * L_inv(np.exp(A_yk[k] + B_yk[k] * K_2019 + alpha_yk[k] + beta_yk[k]*kappa_2019))
        
        # Calculate A_hat for age x using the logistic function
        beta_hat_x = (np.log(L(weighted_sum)) - A_extrapolated[age] - B_extrapolated[age] * K_2019 - alpha_extrapolated[age])/kappa_2019
        beta_hat_values.append(beta_hat_x)
    
    return beta_hat_values

kappa_2019_male = kappa_opt_male[-1]
kappa_2019_female = kappa_opt_female[-1]
alpha_yk_male = alpha_opt_male[-11:]  
alpha_yk_female = alpha_opt_female[-11:]  
beta_yk_male = beta_opt_male[-11:] 
beta_yk_female = beta_opt_female[-11:] 
# Determine A_hat values for ages 91 to 120
beta_extrapolated_male = determine_beta_hat(A_yk_male, A_extrapolated_male, B_yk_male, B_extrapolated_male, K_2019_male, y_k_values,alpha_yk_male,alpha_extrapolated_male,beta_yk_male,kappa_2019_male)
beta_extrapolated_female = determine_beta_hat(A_yk_female, A_extrapolated_female, B_yk_female, B_extrapolated_female, K_2019_female, y_k_values,alpha_yk_female,alpha_extrapolated_female,beta_yk_female,kappa_2019_female)
beta_opt_male_total=np.hstack([beta_opt_male,beta_extrapolated_male])
beta_opt_female_total=np.hstack([beta_opt_female,beta_extrapolated_female])
print("Extrapolated Beta values for ages 91 to 120:")
for age, beta_value_male, beta_value_female in zip(X_tilde, beta_extrapolated_male, beta_extrapolated_female):
    print(f"Age {age}: Beta male= {beta_value_male}, Beta female = {beta_value_female}")

#####################################
#Simulation of the pre-covid-time series
# Perform Cholesky decomposition to obtain matrix H such that H * H.T = C
H = np.linalg.cholesky(C_optimized)

# Number of time periods to simulate: 2020 to 2200
n_time_periods = 2200-2020+1

# Generate standard normal variables (Z_tilde) for each time period (4 variables for each time period)
Z_tilde = np.random.normal(size=(n_time_periods, 4))

# Calculate Z_t by multiplying H with Z_tilde (for each time period)
Z_t_future = Z_tilde @ H

Z_t_future_df = pd.DataFrame(Z_t_future, columns=['epsilon_male', 'epsilon_female', 'delta_male', 'delta_female'])

epsilon_male = Z_t_future_df['epsilon_male']
epsilon_female = Z_t_future_df['epsilon_female']
delta_male = Z_t_future_df['delta_male']
delta_female = Z_t_future_df['delta_female']

epsilon_male= epsilon_male.to_numpy()
epsilon_female = epsilon_female.to_numpy()
delta_male= delta_male.to_numpy()
delta_female = delta_female.to_numpy()

####################
# Creating the years array
years = np.arange(2019, 2201)
number_years=len(years)
# Initialize arrays to store K and kappa values for each future year
K_values_male = np.zeros(number_years)
K_values_female = np.zeros(number_years)
kappa_values_male = np.zeros(number_years)
kappa_values_female = np.zeros(number_years)
# Set initial values (for 2021)
K_values_male[0] = K_2019_male
K_values_female[0] = K_2019_female
kappa_values_male[0] = kappa_2019_male
kappa_values_female[0] = kappa_2019_female
length2=len(K_values_male)
# Simulate K_t and kappa_t for future years (2020 to 2200)
for t in range(1, number_years):
    K_values_male[t] = K_values_male[t - 1] + theta_opt_male
    K_values_female[t] = K_values_female[t - 1] + theta_opt_female
    kappa_values_male[t] = a_opt_male * kappa_values_male[t - 1] + c_opt_male
    kappa_values_female[t] = a_opt_female * kappa_values_female[t - 1] + c_opt_female

years_future =years[3:]
K_future_male=K_values_male[3:]
K_future_female=K_values_female[3:]
kappa_future_male=kappa_values_male[3:]
kappa_future_female=kappa_values_female[3:]
# Creating a DataFrame for the results without 2019,2020,2021
results_df = pd.DataFrame({
    "Year": years_future,
    "K_t_male": K_future_male,
    "K_t_female": K_future_female, 
    "kappa_t_male": kappa_future_male,
    "kappa_t_female": kappa_future_female})

# Calculating mu_pre_covid
def calculate_ln_mu_pre_covid(A_x, B_x, alpha_x, beta_x, K_future, kappa_future):
    # Calculating the EU component: ln(mu_x_pre_covid_EU(t)) = A_x + B_x * K_t
    ln_mu_pre_covid_EU = A_x[:, None] + B_x[:, None] * K_future[None, :]
    
    # Calculating the Ducth component: ln(mu_x_pre_covid_NL(t)) = alpha_x + beta_x * kappa_t
    ln_mu_pre_covid_NL = alpha_x[:, None] + beta_x[:, None] * kappa_future[None, :]
    
    # Combining the two components: ln(mu_x_pre_covid(t)) = ln(mu_x_pre_covid_EU(t)) + ln(mu_x_pre_covid_NL(t))
    ln_mu_pre_covid = ln_mu_pre_covid_EU + ln_mu_pre_covid_NL
    
    return ln_mu_pre_covid

# Calculating mu_pre_covid for males for each age and year
ln_mu_pre_covid_male = calculate_ln_mu_pre_covid(A_opt_male_total, B_opt_male_total, alpha_opt_male_total, beta_opt_male_total, K_future_male, kappa_future_male)
ln_mu_pre_covid_female = calculate_ln_mu_pre_covid(A_opt_female_total, B_opt_female_total, alpha_opt_female_total, beta_opt_female_total, K_future_female, kappa_future_female)


################################
#For Covid layer calculation
mu_pre_covid_male=np.exp(ln_mu_pre_covid_male)
mu_pre_covid_female=np.exp(ln_mu_pre_covid_female)
index = pd.MultiIndex.from_product([np.arange(0, 121), np.arange(2022, 2201)], names=["Age", "Year"])
mu_precovid_df = pd.DataFrame({
    'Mu precovid Male': mu_pre_covid_male.flatten(),
    'Mu precovid Female': mu_pre_covid_female.flatten()
}, index=index)
# Filtering for ages 55 to 90 and years 2022 and 2023
filtered_df = mu_precovid_df.loc[(slice(55, 90), [2022, 2023]), :]

mu_male_precovid_filtered = filtered_df['Mu precovid Male'].unstack(level='Year').values
mu_female_precovid_filtered = filtered_df['Mu precovid Female'].unstack(level='Year').values

# Printing the mu_pre covid arrays for males and females
print("Mu Male Pre-Covid Filtered:")
print(mu_male_precovid_filtered)

print("\nMu Female Pre-Covid Filtered:")
print(mu_female_precovid_filtered)

############################
#Covid layer

# Frak beta data (taken as given)
file_path = 'AG2024result.xlsx' 

parameterwaarden_df = pd.read_excel(file_path, sheet_name='Parameterwaarden 2024', header=None)

frak_beta_male_column = 5  # Column 6 for men (zero-indexed 5)
frak_beta_female_column = 17  # Column 18 for women (zero-indexed 17)

frak_beta_male = parameterwaarden_df.iloc[9:130, frak_beta_male_column].reset_index(drop=True)
frak_beta_female = parameterwaarden_df.iloc[9:130, frak_beta_female_column].reset_index(drop=True)

# Creating a DataFrame combining the frak-beta values for men and women
ages = pd.Series(range(0, 121), name='Age')
combined_frak_beta = pd.DataFrame({
    'Age': ages,
    'frak-beta(x) - Men': frak_beta_male,
    'frak-beta(x) - Women': frak_beta_female})

# Given values for x_t in 2022 and 2023
x_t_2022_male = 1.54678557447261
x_t_2023_male = 1.308375976276
x_t_2022_female = 3.45722029886781
x_t_2023_female = 3.363414048

eta = 0.75

# Create a range of years from 2022 to 2200
years = np.arange(2022, 2201)

# Initialize lists to store x_t and death probabilities by age and year
x_t_male = []
x_t_female = []
death_probability_male_by_age = []
death_probability_female_by_age = []

x_t_male.append(x_t_2022_male)  # x_t for men in 2022
x_t_male.append(x_t_2023_male)  # x_t for men in 2023
x_t_female.append(x_t_2022_female)  # x_t for women in 2022
x_t_female.append(x_t_2023_female)  # x_t for women in 2023

# Calculating x_t for t >= 2024 using the given formula for men and women
for t in years[2:]:
    x_t_men = x_t_2023_male * eta**(t - 2023)
    x_t_women = x_t_2023_female * eta**(t - 2023)
    x_t_male.append(x_t_men)
    x_t_female.append(x_t_women)

# Ensuring that frak_beta_male, frak_beta_female, x_t_values_male, and x_t_values_female are NumPy arrays
frak_beta_male = np.array(frak_beta_male)
frak_beta_female = np.array(frak_beta_female)

# Calculating ln(o_x(t)) for all years
ln_o_x_men = frak_beta_male[:, None] * x_t_male  # Shape (121, len(years))
ln_o_x_women = frak_beta_female[:, None] * x_t_female  # Shape (121, len(years))


ln_mu_male= ln_mu_pre_covid_male + ln_o_x_men
ln_mu_female= ln_mu_pre_covid_female + ln_o_x_women
ln_mu_male = ln_mu_male.astype(float)
ln_mu_female = ln_mu_female.astype(float)
mu_male=np.exp(ln_mu_male)
mu_female=np.exp(ln_mu_female)
# Calculating death probabilities for all ages and years
death_prob_male = 1 - np.exp(-mu_male)
death_prob_female = 1 - np.exp(-mu_female)

# Creating a DataFrame for both male and female probabilities
index = pd.MultiIndex.from_product([np.arange(0, 121), np.arange(2022, 2201)], names=["Age", "Year"])
death_df = pd.DataFrame({
    'Death Probability Male': death_prob_male.flatten(),
    'Death Probability Female': death_prob_female.flatten()
}, index=index)

# Separating male and female data
death_male_df = death_df['Death Probability Male'].unstack(level="Year")
death_female_df = death_df['Death Probability Female'].unstack(level="Year")

#####################################
# Calculating survival probabilities for all ages and years
survival_prob_male=np.exp(-mu_male)
survival_prob_female=np.exp(-mu_female)
survival_df = pd.DataFrame({
    'Death Probability Male': survival_prob_male.flatten(),
    'Death Probability Female': survival_prob_female.flatten()
}, index=index)

# Separating male and female data
survival_male_df = survival_df['Death Probability Male'].unstack(level="Year")
survival_female_df = survival_df['Death Probability Female'].unstack(level="Year")

#Filter columns for years 2025 onwards
survival_male_2025_df = survival_male_df.loc[:, 2025:]
survival_female_2025_df = survival_female_df.loc[:, 2025:]

#Calculating t-years survival rates
def calculate_survival_by_t(df, max_t):
    # Create a new DataFrame to store survival probabilities
    survival_probs_df = pd.DataFrame(index=df.index, columns=range(0, max_t + 1))  # t = 0 to max_t
    
    # Set 0-year survival probabilities to 1
    survival_probs_df[0] = 1
    
    # Calculate survival probabilities for t >= 1
    for age in df.index:  # Iterate over ages
        for t in range(1, max_t + 1):  # Iterate over t-years
            end_age = age + t - 1
            if end_age <= df.index[-1]:  # Ensure age does not exceed maximum age
                diagonal = [df.loc[age + j, df.columns[j]] for j in range(t)]
                survival_probs_df.loc[age, t] = np.prod(diagonal)
            else:
                survival_probs_df.loc[age, t] = np.nan  # Set NaN if t extends beyond the max age
    
    return survival_probs_df


# Calculating t-year survival probabilities for males and females
max_t = 120  # for t=0 to t=120
t_years_survival_male = calculate_survival_by_t(survival_male_2025_df, max_t)
t_years_survival_female = calculate_survival_by_t(survival_female_2025_df, max_t)


def t_p_x(x, t, survival_male_df, survival_female_df, gender):
    if gender not in ['male', 'female']:
        raise ValueError("Invalid gender. Choose from 'male' or 'female'.")

    # Retrieve survival probability based on gender
    if gender == 'male':
        try:
            return survival_male_df.loc[x, t]
        except KeyError:
            return None  # Handle case where age or t is out of bounds for males
    elif gender == 'female':
        try:
            return survival_female_df.loc[x, t]
        except KeyError:
            return None  # Handle case where age or t is out of bounds for females
#trial
# Specify the x and t (number of years)
x = 60
t = 5
male_prob = t_p_x(x, t, t_years_survival_male, t_years_survival_female, gender='male')
print(f"{t}-year survival probability for age {age} (Male): {male_prob}")
female_prob = t_p_x(x, t, t_years_survival_male,t_years_survival_female, gender='female')
print(f"{t}-year survival probability for age {age} (Female): {female_prob}")


#Calculating the Annuity factors for commenced retirement pension (OP)
def calculate_annuity_factor(r, t_p_x_df, ages, max_age=120):
    v = 1 / (1 + r)

    # Initialize a Series to store the results for the specified ages
    results = pd.Series(index=ages, dtype=float)

    # Iterate over the specified ages to calculate the formula
    for age in ages:
        if age not in t_p_x_df.index:
            results[age] = np.nan  # Handle missing ages
            continue

        # Extract the survival probabilities for the current age
        t_p_x_values = t_p_x_df.loc[age].values

        # Calculate the maximum valid `t` for this age
        max_t = min(t_p_x_df.columns.max(), max_age - age)

        # Precompute the discount factors for all valid `t`
        discount_factors = np.array([v**t for t in range(0, max_t + 1)])

        # Calculate the first summation (from t=1 to max_t)
        summation_1 = sum(t_p_x_values[t] * discount_factors[t] for t in range(1, max_t + 1))

        # Calculate the second summation (from t=0 to max_t)
        summation_2 = sum(t_p_x_values[t] * discount_factors[t] for t in range(0, max_t + 1))

        # Combine the two summations and divide by 2
        results[age] = 0.5 * (summation_1 + summation_2)

    return results



# Specific ages to calculate Commenced OP
specific_ages = [30, 40, 50, 60, 70, 80, 90]
interest_rate = 0.03
# Calculating Annuity Factors for males
annuity_factor_male = calculate_annuity_factor(interest_rate, t_years_survival_male, specific_ages)

# Calculating Annuity factors for females
annuity_factor_female = calculate_annuity_factor(interest_rate, t_years_survival_female, specific_ages)

# Combine results into a single DataFrame
annuity_factor_combined = pd.DataFrame({
    'Age': specific_ages,
    'Annuity factor Male': annuity_factor_male.values,
    'Annuity factor Female': annuity_factor_female.values
})

print("Annuity factor Combined Results:")
print(annuity_factor_combined)

#Annual payments (given in AG 2024)
data = {
    "Age": [30, 40, 50, 60, 70, 80, 90],
    "Male OP": [1500, 8500, 15000, 15000, 8500, 3500, 500],
    "Female OP": [2500, 7500, 12500, 10000, 7500, 5000, 1000]}
U_x_table = pd.DataFrame(data).set_index("Age")  # Set Age as the index

# Getting specific U_x values
def U_x(age, sex):
    if sex.lower() not in ['male', 'female']:
        raise ValueError("Invalid sex. Please choose 'male' or 'female'.")
    
    try:
        # Retrieve the value based on sex and age
        if sex.lower() == 'male':
            return U_x_table.loc[age, "Male OP"]
        elif sex.lower() == 'female':
            return U_x_table.loc[age, "Female OP"]
    except KeyError:
        return None  # Handle case where age is not in the table

# Example
ages_to_check = [30, 50, 90]
for age in ages_to_check:
    male_U_x = U_x(age, 'male')
    female_U_x = U_x(age, 'female')
    print(f"U_x for Age {age}, Male: {male_U_x}, Female: {female_U_x}")

print("\nU_x Table:")
print(U_x_table)


# Calculating the provision for old-age pension with immediate effect
def calculate_provision_op(U_x_table, annuity_factor, ages, sex):
    # Ensure valid gender input
    if sex.lower() not in ['male', 'female']:
        raise ValueError("Invalid gender. Please choose 'male' or 'female'.")
    
    # Retrieve the U_x^{op} column based on gender
    column_name = "Male OP" if sex.lower() == 'male' else "Female OP"
    
    # Multiply U_x^{op} by commenced OP for each age
    provisions = []
    for age in ages:
        try:
            U_x_op = U_x_table.loc[age, column_name]
            alpha_x = annuity_factor.loc[age]
            provisions.append(U_x_op * alpha_x)
        except KeyError:
            provisions.append(None)  # Handle missing data for ages
    
    return pd.Series(provisions, index=ages)

# Calculating the provision for males and females
provision_male = calculate_provision_op(U_x_table, annuity_factor_male, specific_ages, 'male')
provision_female = calculate_provision_op(U_x_table, annuity_factor_female, specific_ages, 'female')

# Combining results into one DataFrame
provision_combined = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male': provision_male.values,
    'Provision Female': provision_female.values})

print("Provision for Old-Age Pension (Immediate Effect):")
print(provision_combined)

##############################
#Model refinement

#One off scenario - scenario 1
# Calculating x_t for t >= 2024 for men and women
years_new = np.arange(2024, 2201)
x_t_male_v1 = [x_t_2022_male, x_t_2023_male]
x_t_female_v1 = [x_t_2022_female, x_t_2023_female]
num_years = 2200 - 2024 + 1

# Repeating the same values for the calculated number of years
for _ in range(num_years):
    x_t_male_v1.append(0)
    x_t_female_v1.append(0)

# Calculating ln(o_x(t)) for all years
ln_o_x_men_v1 = frak_beta_male[:, None] * x_t_male_v1  # Shape (121, len(years))
ln_o_x_women_v1 = frak_beta_female[:, None] * x_t_female_v1  # Shape (121, len(years))


ln_mu_male_v1= ln_mu_pre_covid_male + ln_o_x_men_v1
ln_mu_female_v1= ln_mu_pre_covid_female + ln_o_x_women_v1
ln_mu_male_v1 = ln_mu_male_v1.astype(float)
ln_mu_female_v1 = ln_mu_female_v1.astype(float)
mu_male_v1=np.exp(ln_mu_male_v1)
mu_female_v1=np.exp(ln_mu_female_v1)
# Calculating death probabilities for all ages and years
death_prob_male_v1 = 1 - np.exp(-mu_male_v1)
death_prob_female_v1 = 1 - np.exp(-mu_female_v1)

# Flatten the arrays and create a DataFrame for both male and female probabilities
index = pd.MultiIndex.from_product([np.arange(0, 121), np.arange(2022, 2201)], names=["Age", "Year"])
death_df_v1 = pd.DataFrame({
    'Death Probability Male': death_prob_male_v1.flatten(),
    'Death Probability Female': death_prob_female_v1.flatten()
}, index=index)

# Separating male and female data
death_male_df_v1 = death_df_v1['Death Probability Male'].unstack(level="Year")
death_female_df_v1 = death_df_v1['Death Probability Female'].unstack(level="Year")

###Survival probabilities
# Calculating survival probabilities for all ages and years
survival_prob_male_v1=np.exp(-mu_male_v1)
survival_prob_female_v1=np.exp(-mu_female_v1)
survival_df_v1 = pd.DataFrame({
    'Death Probability Male for One off scenario': survival_prob_male_v1.flatten(),
    'Death Probability Female for One off scenario': survival_prob_female_v1.flatten()
}, index=index)

# Separating male and female data
survival_male_v1_df = survival_df_v1['Death Probability Male for One off scenario'].unstack(level="Year")
survival_female_v1_df = survival_df_v1['Death Probability Female for One off scenario'].unstack(level="Year")

#Filtering columns for years 2025 onwards
survival_male_2025_v1_df = survival_male_v1_df.loc[:, 2025:]
survival_female_2025_v1_df = survival_female_v1_df.loc[:, 2025:]

t_years_survival_male_v1 = calculate_survival_by_t(survival_male_2025_v1_df, max_t)
t_years_survival_female_v1 = calculate_survival_by_t(survival_female_2025_v1_df, max_t)

# Calculating Annuity factor for males and females
annuity_factor_male_v1 = calculate_annuity_factor(interest_rate, t_years_survival_male_v1, specific_ages)
annuity_factor_female_v1 = calculate_annuity_factor(interest_rate, t_years_survival_female_v1, specific_ages)
annuity_factor_combined_v1 = pd.DataFrame({
    'Age': specific_ages,
    'Annuity factor Male for One off Scenario': annuity_factor_male_v1.values,
    'Annuity factor Female for One off Scenario': annuity_factor_female_v1.values})

print("Annuity factor for One off scenario:")
print(annuity_factor_combined_v1)

# Calculating provision for males and females
provision_male_v1 = calculate_provision_op(U_x_table, annuity_factor_male_v1, specific_ages, 'male')
provision_female_v1 = calculate_provision_op(U_x_table, annuity_factor_female_v1, specific_ages, 'female')
provision_combined_v1 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for One off scenario': provision_male_v1.values,
    'Provision Female for One off scenario': provision_female_v1.values})

print("Provision for Old-Age Pension (Immediate Effect) for One off scenario:")
print(provision_combined_v1)
################
#Structurual scenario - scenario 2
# Calculating x_t for t >= 2024 for men and women
x_t_male_v2 = [x_t_2022_male, x_t_2023_male]
x_t_female_v2 = [x_t_2022_female, x_t_2023_female]

# Calculating the number of years (2201 - 2024)
num_years = 2200 - 2024 + 1

# Repeating the same values for the calculated number of years
for _ in range(num_years):
    x_t_male_v2.append(x_t_2023_male)
    x_t_female_v2.append(x_t_2023_female)

# Calculating ln(o_x(t)) for all years
ln_o_x_men_v2 = frak_beta_male[:, None] * x_t_male_v2  # Shape (121, len(years))
ln_o_x_women_v2 = frak_beta_female[:, None] * x_t_female_v2  # Shape (121, len(years))


ln_mu_male_v2= ln_mu_pre_covid_male + ln_o_x_men_v2
ln_mu_female_v2= ln_mu_pre_covid_female + ln_o_x_women_v2
ln_mu_male_v2 = ln_mu_male_v2.astype(float)
ln_mu_female_v2 = ln_mu_female_v2.astype(float)
mu_male_v2=np.exp(ln_mu_male_v2)
mu_female_v2=np.exp(ln_mu_female_v2)
# Calculating death probabilities for all ages and years
death_prob_male_v2 = 1 - np.exp(-mu_male_v2)
death_prob_female_v2 = 1 - np.exp(-mu_female_v2)

# Flatten the arrays and create a DataFrame for both male and female probabilities
index = pd.MultiIndex.from_product([np.arange(0, 121), np.arange(2022, 2201)], names=["Age", "Year"])
death_df_v2 = pd.DataFrame({
    'Death Probability Male': death_prob_male_v2.flatten(),
    'Death Probability Female': death_prob_female_v2.flatten()
}, index=index)

# Separating male and female data
death_male_df_v2 = death_df_v2['Death Probability Male'].unstack(level="Year")
death_female_df_v2 = death_df_v2['Death Probability Female'].unstack(level="Year")

# Calculating survival probabilities for all ages and years
survival_prob_male_v2=np.exp(-mu_male_v2)
survival_prob_female_v2=np.exp(-mu_female_v2)
survival_df_v2 = pd.DataFrame({
    'Death Probability Male for Structural scenario': survival_prob_male_v2.flatten(),
    'Death Probability Female for Structural scenario': survival_prob_female_v2.flatten()
}, index=index)

# Separating male and female data
survival_male_v2_df = survival_df_v2['Death Probability Male for Structural scenario'].unstack(level="Year")
survival_female_v2_df = survival_df_v2['Death Probability Female for Structural scenario'].unstack(level="Year")

#Filtering columns for years 2025 onwards
survival_male_2025_v2_df = survival_male_v2_df.loc[:, 2025:]
survival_female_2025_v2_df = survival_female_v2_df.loc[:, 2025:]

t_years_survival_male_v2 = calculate_survival_by_t(survival_male_2025_v2_df, max_t)
t_years_survival_female_v2 = calculate_survival_by_t(survival_female_2025_v2_df, max_t)

# Calculating Annuity factor for males and females
annuity_factor_male_v2 = calculate_annuity_factor(interest_rate, t_years_survival_male_v2, specific_ages)
annuity_factor_female_v2 = calculate_annuity_factor(interest_rate, t_years_survival_female_v2, specific_ages)
annuity_factor_combined_v2 = pd.DataFrame({
    'Age': specific_ages,
    'Annuity factor Male for Structural Scenario': annuity_factor_male_v2.values,
    'Annuity factor Female for Structural Scenario': annuity_factor_female_v2.values})

print("Annuity factor for Structural scenario:")
print(annuity_factor_combined_v2)

# Calculating provision for males and females
provision_male_v2 = calculate_provision_op(U_x_table, annuity_factor_male_v2, specific_ages, 'male')
provision_female_v2 = calculate_provision_op(U_x_table, annuity_factor_female_v2, specific_ages, 'female')
provision_combined_v2 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Structural scenario': provision_male_v2.values,
    'Provision Female for Structural scenario': provision_female_v2.values})

print("Provision for Old-Age Pension (Immediate Effect) for Structural scenario:")
print(provision_combined_v2)

#New normal scenario - scenario 3
# Calculating x_t for t >= 2024 for men and women
x_t_male_v3 = [x_t_2022_male, x_t_2023_male]
x_t_female_v3 = [x_t_2022_female, x_t_2023_female]

# Calculating x_t for t < 2026
for t in range(2024, 2026):
    x_t_men_v3 = x_t_2023_male * eta ** (t - 2023)
    x_t_women_v3 = x_t_2023_female * eta ** (t - 2023)
    x_t_male_v3.append(x_t_men_v3)
    x_t_female_v3.append(x_t_women_v3)
fixed_level_male=sum(x_t_male_v3)/4
fixed_level_female=sum(x_t_female_v3)/4
# Calculating x_t for t >= 2026
# Calculating the number of years
num_years_new = 2200 - 2026 + 1
# Repeating the same values for the calculated number of years
for _ in range(num_years_new):
    x_t_male_v3.append(fixed_level_male)
    x_t_female_v3.append(fixed_level_female)


# Calculating ln(o_x(t)) for all years
ln_o_x_men_v3 = frak_beta_male[:, None] * x_t_male_v3  # Shape (121, len(years))
ln_o_x_women_v3 = frak_beta_female[:, None] * x_t_female_v3  # Shape (121, len(years))


ln_mu_male_v3= ln_mu_pre_covid_male + ln_o_x_men_v3
ln_mu_female_v3= ln_mu_pre_covid_female + ln_o_x_women_v3
ln_mu_male_v3 = ln_mu_male_v3.astype(float)
ln_mu_female_v3 = ln_mu_female_v3.astype(float)
mu_male_v3=np.exp(ln_mu_male_v3)
mu_female_v3=np.exp(ln_mu_female_v3)
# Calculating death probabilities for all ages and years
death_prob_male_v3 = 1 - np.exp(-mu_male_v3)
death_prob_female_v3 = 1 - np.exp(-mu_female_v3)

# Flatten the arrays and create a DataFrame for both male and female probabilities
index = pd.MultiIndex.from_product([np.arange(0, 121), np.arange(2022, 2201)], names=["Age", "Year"])
death_df_v3 = pd.DataFrame({
    'Death Probability Male': death_prob_male_v3.flatten(),
    'Death Probability Female': death_prob_female_v3.flatten()
}, index=index)

# Separating male and female data
death_male_df_v3 = death_df_v3['Death Probability Male'].unstack(level="Year")
death_female_df_v3 = death_df_v3['Death Probability Female'].unstack(level="Year")

# Calculating survival probabilities for all ages and years
survival_prob_male_v3=np.exp(-mu_male_v3)
survival_prob_female_v3=np.exp(-mu_female_v3)
survival_df_v3 = pd.DataFrame({
    'Death Probability Male for New Normal scenario': survival_prob_male_v3.flatten(),
    'Death Probability Female for New Normal scenario': survival_prob_female_v3.flatten()
}, index=index)

# Separating male and female data
survival_male_v3_df = survival_df_v3['Death Probability Male for New Normal scenario'].unstack(level="Year")
survival_female_v3_df = survival_df_v3['Death Probability Female for New Normal scenario'].unstack(level="Year")

#Filtering columns for years 2025 onwards
survival_male_2025_v3_df = survival_male_v3_df.loc[:, 2025:]
survival_female_2025_v3_df = survival_female_v3_df.loc[:, 2025:]

t_years_survival_male_v3 = calculate_survival_by_t(survival_male_2025_v3_df, max_t)
t_years_survival_female_v3 = calculate_survival_by_t(survival_female_2025_v3_df, max_t)

# Calculating Annuity factor for males and females
annuity_factor_male_v3 = calculate_annuity_factor(interest_rate, t_years_survival_male_v3, specific_ages)
annuity_factor_female_v3 = calculate_annuity_factor(interest_rate, t_years_survival_female_v3, specific_ages)
annuity_factor_combined_v3 = pd.DataFrame({
    'Age': specific_ages,
    'Annuity factor Male for New Normal Scenario': annuity_factor_male_v3.values,
    'Annuity factor Female for New Normal Scenario':annuity_factor_female_v3.values})

print("Annuity factor for New Normal scenario:")
print(annuity_factor_combined_v3)

# Calculating provision for males and females
provision_male_v3 = calculate_provision_op(U_x_table, annuity_factor_male_v3, specific_ages, 'male')
provision_female_v3 = calculate_provision_op(U_x_table,annuity_factor_female_v3, specific_ages, 'female')
provision_combined_v3 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for New Normal scenario': provision_male_v3.values,
    'Provision Female for New Normal scenario': provision_female_v3.values})

print("Provision for Old-Age Pension (Immediate Effect) for New Normal scenario:")
print(provision_combined_v3)


#Reintroduction scenario - scenario 4
# Calculating x_t for t >= 2024 for men and women
x_t_male_v4= [x_t_2022_male, x_t_2023_male]
x_t_female_v4 = [x_t_2022_female, x_t_2023_female]

# Calculating x_t for t < 2026
for t in range(2024, 2027):
    x_t_men_v4 = x_t_2023_male * eta ** (t - 2023)
    x_t_women_v4 = x_t_2023_female * eta ** (t - 2023)
    x_t_male_v4.append(x_t_men_v4)
    x_t_female_v4.append(x_t_women_v4)

# Calculating x_t for t >= 2026
# Calculating the number of years
num_years_new2 = 2200 - 2027 + 1
# Repeating the same values for the calculated number of years
for _ in range(num_years_new2):
    x_t_male_v4.append(x_t_2022_male)
    x_t_female_v4.append(x_t_2022_female)

# Calculating ln(o_x(t)) for all years at once
ln_o_x_men_v4 = frak_beta_male[:, None] * x_t_male_v4  # Shape (121, len(years))
ln_o_x_women_v4 = frak_beta_female[:, None] * x_t_female_v4  # Shape (121, len(years))


ln_mu_male_v4= ln_mu_pre_covid_male + ln_o_x_men_v4
ln_mu_female_v4= ln_mu_pre_covid_female + ln_o_x_women_v4
ln_mu_male_v4 = ln_mu_male_v4.astype(float)
ln_mu_female_v4 = ln_mu_female_v4.astype(float)
mu_male_v4=np.exp(ln_mu_male_v4)
mu_female_v4=np.exp(ln_mu_female_v4)
# Calculating death probabilities for all ages and years
death_prob_male_v4 = 1 - np.exp(-mu_male_v4)
death_prob_female_v4 = 1 - np.exp(-mu_female_v4)

# Flatten the arrays and create a DataFrame for both male and female probabilities
index = pd.MultiIndex.from_product([np.arange(0, 121), np.arange(2022, 2201)], names=["Age", "Year"])
death_df_v4 = pd.DataFrame({
    'Death Probability Male': death_prob_male_v4.flatten(),
    'Death Probability Female': death_prob_female_v4.flatten()}, index=index)

# Separating male and female data
death_male_df_v4 = death_df_v4['Death Probability Male'].unstack(level="Year")
death_female_df_v4 = death_df_v4['Death Probability Female'].unstack(level="Year")

# Calculating survival probabilities for all ages and years
survival_prob_male_v4=np.exp(-mu_male_v4)
survival_prob_female_v4=np.exp(-mu_female_v4)
survival_df_v4 = pd.DataFrame({
    'Death Probability Male for Reintroduction scenario': survival_prob_male_v4.flatten(),
    'Death Probability Female for Reintroduction scenario': survival_prob_female_v4.flatten()}, index=index)
# Separating male and female data
survival_male_v4_df = survival_df_v4['Death Probability Male for Reintroduction scenario'].unstack(level="Year")
survival_female_v4_df = survival_df_v4['Death Probability Female for Reintroduction scenario'].unstack(level="Year")

#Filtering columns for years 2025 onwards
survival_male_2025_v4_df = survival_male_v4_df.loc[:, 2025:]
survival_female_2025_v4_df = survival_female_v4_df.loc[:, 2025:]

t_years_survival_male_v4 = calculate_survival_by_t(survival_male_2025_v4_df, max_t)
t_years_survival_female_v4 = calculate_survival_by_t(survival_female_2025_v4_df, max_t)

# Calculating Annuity factor for males and females
annuity_factor_male_v4 = calculate_annuity_factor(interest_rate, t_years_survival_male_v4, specific_ages)
annuity_factor_female_v4 = calculate_annuity_factor(interest_rate, t_years_survival_female_v4, specific_ages)
annuity_factor_combined_v4 = pd.DataFrame({
    'Age': specific_ages,
    'Annuity factor Male for Reintroduction Scenario': annuity_factor_male_v4.values,
    'Annuity factor Female for Reintroduction Scenario': annuity_factor_female_v4.values})

print("Annuity factor for Reintroduction scenario:")
print(annuity_factor_combined_v4)

# Calculate provision for males and females
provision_male_v4 = calculate_provision_op(U_x_table, annuity_factor_male_v4, specific_ages, 'male')
provision_female_v4 = calculate_provision_op(U_x_table, annuity_factor_female_v4, specific_ages, 'female')
provision_combined_v4 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Reintroduction scenario': provision_male_v4.values,
    'Provision Female for Reintroduction scenario': provision_female_v4.values})

print("Provision for Old-Age Pension (Immediate Effect) for Reintroduction scenario:")
print(provision_combined_v4)
#########
#Combined values for all scenarios
# Combined Annuity factor values for males across all scenarios
annuity_factor_male_combined = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': annuity_factor_male.values,
    'One-off Scenario': annuity_factor_male_v1.values,
    'Structural Scenario': annuity_factor_male_v2.values,
    'New Normal Scenario': annuity_factor_male_v3.values,
    'Reintroduction Scenario': annuity_factor_male_v4.values})

# Combined Annuity factor values for females across all scenarios
annuity_factor_female_combined = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': annuity_factor_female.values,
    'One-off Scenario': annuity_factor_female_v1.values,
    'Structural Scenario': annuity_factor_female_v2.values,
    'New Normal Scenario': annuity_factor_female_v3.values,
    'Reintroduction Scenario': annuity_factor_female_v4.values})

print("Annuity factor for Males:")
print(annuity_factor_male_combined)

print("Annuity factor for Females:")
print(annuity_factor_female_combined)

#Provision values for all scenarios
# Combined provision values for males across all scenarios
provision_male_combined = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_male.values,
    'One-off Scenario': provision_male_v1.values,
    'Structural Scenario': provision_male_v2.values,
    'New Normal Scenario': provision_male_v3.values,
    'Reintroduction Scenario': provision_male_v4.values
})

# Combine provision values for females across all scenarios
provision_female_combined = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_female.values,
    'One-off Scenario': provision_female_v1.values,
    'Structural Scenario': provision_female_v2.values,
    'New Normal Scenario': provision_female_v3.values,
    'Reintroduction Scenario': provision_female_v4.values
})

print("Provision for Males:")
print(provision_male_combined)

print("\nProvision for Females:")
print(provision_female_combined)

# Calculation of percentage deviations
def calculate_percentage_deviation(df, base_column):
    deviation_df = df.copy()
    scenarios = [col for col in df.columns if col != 'Age' and col != base_column]
    for scenario in scenarios:
        deviation_df[scenario] = ((df[scenario] - df[base_column]) / df[base_column]) * 100
    deviation_df = deviation_df.drop(columns=[base_column])
    return deviation_df

# Provision values for all scenarios
# Combined provision values for males across all scenarios
provision_male_deviation = calculate_percentage_deviation(provision_male_combined, 'AG 2024')

# Combined provision values for females across all scenarios
provision_female_deviation = calculate_percentage_deviation(provision_female_combined, 'AG 2024')

print("Percentage Deviations for Provision (Males):")
print(provision_male_deviation)

print("\nPercentage Deviations for Provision (Females):")
print(provision_female_deviation)

#Calculation of cohort life expectancy
def cohort_life_expectancy(t, x_values, survival_probabilities):
    results = {}
    
    for x in x_values:
        e_x_coh = 0.5  # Start with 1/2 as per the formula
        
        # Iterate over years k and sum the cohort life expectancy
        for k in range(0, len(survival_probabilities.columns)):
            product_term = 1  # Initialize the product term
            for s in range(0, k + 1):
                # Get the survival probability for age (x + s) and year (t + s)
                age = x + s
                year = t + s
                if age in survival_probabilities.index and year in survival_probabilities.columns:
                    product_term *= survival_probabilities.loc[age, year]
                else:
                    product_term = 0  # If data is missing, assume probability is 0
                    break
            e_x_coh += product_term
        
        # Store the result for the current age
        results[x] = e_x_coh
    
    return results

t = 2025

# Converting results from dictionary to list for DataFrame creation
def get_ordered_results(results, ages):
    return [results[age] for age in ages]

# Calculating cohort life expectancy for Males
life_expectancy_AG2024_male = cohort_life_expectancy(t, specific_ages, survival_male_df)
life_expectancy_One_off_male = cohort_life_expectancy(t, specific_ages, survival_male_v1_df)
life_expectancy_Structural_male = cohort_life_expectancy(t, specific_ages, survival_male_v2_df)
life_expectancy_New_Normal_male = cohort_life_expectancy(t, specific_ages, survival_male_v3_df)
life_expectancy_Reintroduction_male = cohort_life_expectancy(t, specific_ages, survival_male_v4_df)

# Combined life expectancy values for males across all scenarios
life_expectancy_male_combined = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': get_ordered_results(life_expectancy_AG2024_male, specific_ages),
    'One-off Scenario': get_ordered_results(life_expectancy_One_off_male, specific_ages),
    'Structural Scenario': get_ordered_results(life_expectancy_Structural_male, specific_ages),
    'New Normal Scenario': get_ordered_results(life_expectancy_New_Normal_male, specific_ages),
    'Reintroduction Scenario': get_ordered_results(life_expectancy_Reintroduction_male, specific_ages)})

print("Life expectancy for Males:")
print(life_expectancy_male_combined)

# Calculating cohort life expectancy for Females
life_expectancy_AG2024_female = cohort_life_expectancy(t, specific_ages, survival_female_df)
life_expectancy_One_off_female = cohort_life_expectancy(t, specific_ages, survival_female_v1_df)
life_expectancy_Structural_female = cohort_life_expectancy(t, specific_ages, survival_female_v2_df)
life_expectancy_New_Normal_female = cohort_life_expectancy(t, specific_ages, survival_female_v3_df)
life_expectancy_Reintroduction_female = cohort_life_expectancy(t, specific_ages, survival_female_v4_df)

# Combined life expectancy values for females across all scenarios
life_expectancy_female_combined = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': get_ordered_results(life_expectancy_AG2024_female, specific_ages),
    'One-off Scenario': get_ordered_results(life_expectancy_One_off_female, specific_ages),
    'Structural Scenario': get_ordered_results(life_expectancy_Structural_female, specific_ages),
    'New Normal Scenario': get_ordered_results(life_expectancy_New_Normal_female, specific_ages),
    'Reintroduction Scenario': get_ordered_results(life_expectancy_Reintroduction_female, specific_ages)})


print("Life expectancy for Females:")
print(life_expectancy_female_combined)


# Function to calculate deviations in years from AG 2024
def calculate_deviations(base_column, comparison_df):
    deviations = {}
    for i, age in enumerate(comparison_df['Age']):  # Loop through the 'Age' column
        base_value = base_column[i]  # Get base value using index
        deviations[age] = {
            'One-off Scenario': comparison_df.loc[i, 'One-off Scenario'] - base_value,
            'Structural Scenario': comparison_df.loc[i, 'Structural Scenario'] - base_value,
            'New Normal Scenario': comparison_df.loc[i, 'New Normal Scenario'] - base_value,
            'Reintroduction Scenario': comparison_df.loc[i, 'Reintroduction Scenario'] - base_value,
        }
    return deviations

# Extracting base column (AG 2024 values) as a list
base_column_male = life_expectancy_male_combined['AG 2024'].values.tolist()

# Calculating deviations for Females
deviations_male = calculate_deviations(
    base_column=base_column_male,
    comparison_df=life_expectancy_male_combined)
# Combined deviations into DataFrame for Males
deviation_male_df = pd.DataFrame({
    'Age': specific_ages,
    'One-off Scenario': [deviations_male[age]['One-off Scenario'] for age in specific_ages],
    'Structural Scenario': [deviations_male[age]['Structural Scenario'] for age in specific_ages],
    'New Normal Scenario': [deviations_male[age]['New Normal Scenario'] for age in specific_ages],
    'Reintroduction Scenario': [deviations_male[age]['Reintroduction Scenario'] for age in specific_ages]
})

print("Deviations in Life Expectancy for Males (in years):")
print(deviation_male_df)

# Extracting base column (AG 2024 values) as a list
base_column_female = life_expectancy_female_combined['AG 2024'].values.tolist()
# Calculating deviations for Females
deviations_female = calculate_deviations(
    base_column=base_column_female,
    comparison_df=life_expectancy_female_combined)
# Combined deviations into DataFrame for Females
deviation_female_df = pd.DataFrame({
    'Age': specific_ages,
    'One-off Scenario': [deviations_female[age]['One-off Scenario'] for age in specific_ages],
    'Structural Scenario': [deviations_female[age]['Structural Scenario'] for age in specific_ages],
    'New Normal Scenario': [deviations_female[age]['New Normal Scenario'] for age in specific_ages],
    'Reintroduction Scenario': [deviations_female[age]['Reintroduction Scenario'] for age in specific_ages]
})

print("Deviations in Life Expectancy for Females (in years):")
print(deviation_female_df)

###########################################################
#Sensitivity analysis with different interest rates
#First version: interest  rate= 1%
interest_rate_v1=0.01
#AG 2024
annuity_factor_male_r1 = calculate_annuity_factor(interest_rate_v1,t_years_survival_male, specific_ages)
annuity_factor_female_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_female, specific_ages)

provision_male_r1 = calculate_provision_op(U_x_table, annuity_factor_male_r1, specific_ages, 'male')
provision_female_r1 = calculate_provision_op(U_x_table, annuity_factor_female_r1, specific_ages, 'female')

provision_combined_r1 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male': provision_male_r1.values,
    'Provision Female': provision_female_r1.values})


#One off scenario - scenario 1
annuity_factor_male_v1_r1 = calculate_annuity_factor(interest_rate_v1,t_years_survival_male_v1, specific_ages)
annuity_factor_female_v1_r1 = calculate_annuity_factor(interest_rate_v1,t_years_survival_female_v1, specific_ages)

provision_male_v1_r1 = calculate_provision_op(U_x_table, annuity_factor_male_v1_r1, specific_ages, 'male')
provision_female_v1_r1 = calculate_provision_op(U_x_table, annuity_factor_female_v1_r1, specific_ages, 'female')
provision_combined_v1_r1 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for One off scenario': provision_male_v1_r1.values,
    'Provision Female for One off scenario': provision_female_v1_r1.values})

################
#Structurual scenario - scenario 2
annuity_factor_male_v2_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_male_v2, specific_ages)
annuity_factor_female_v2_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_female_v2, specific_ages)

# Calculating provision for males and females
provision_male_v2_r1 = calculate_provision_op(U_x_table, annuity_factor_male_v2_r1, specific_ages, 'male')
provision_female_v2_r1 = calculate_provision_op(U_x_table, annuity_factor_female_v2_r1, specific_ages, 'female')
provision_combined_v2_r1 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Structural scenario': provision_male_v2_r1.values,
    'Provision Female for Structural scenario': provision_female_v2_r1.values})

#New normal scenario - scenario 3
annuity_factor_male_v3_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_male_v3, specific_ages)
annuity_factor_female_v3_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_female_v3, specific_ages)

provision_male_v3_r1 = calculate_provision_op(U_x_table, annuity_factor_male_v3_r1, specific_ages, 'male')
provision_female_v3_r1 = calculate_provision_op(U_x_table,annuity_factor_female_v3_r1, specific_ages, 'female')
provision_combined_v3_r1 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for New Normal scenario': provision_male_v3_r1.values,
    'Provision Female for New Normal scenario': provision_female_v3_r1.values})

#Reintroduction scenario - scenario 4
annuity_factor_male_v4_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_male_v4, specific_ages)
annuity_factor_female_v4_r1 = calculate_annuity_factor(interest_rate_v1, t_years_survival_female_v4, specific_ages)

provision_male_v4_r1 = calculate_provision_op(U_x_table, annuity_factor_male_v4_r1, specific_ages, 'male')
provision_female_v4_r1 = calculate_provision_op(U_x_table, annuity_factor_female_v4_r1, specific_ages, 'female')
provision_combined_v4_r1 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Reintroduction scenario': provision_male_v4_r1.values,
    'Provision Female for Reintroduction scenario': provision_female_v4_r1.values})

#Provision values for all scenarios
provision_male_combined_r1 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_male_r1.values,
    'One-off Scenario': provision_male_v1_r1.values,
    'Structural Scenario': provision_male_v2_r1.values,
    'New Normal Scenario': provision_male_v3_r1.values,
    'Reintroduction Scenario': provision_male_v4_r1.values
})

# Combine provision values for females across all scenarios
provision_female_combined_r1 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_female_r1.values,
    'One-off Scenario': provision_female_v1_r1.values,
    'Structural Scenario': provision_female_v2_r1.values,
    'New Normal Scenario': provision_female_v3_r1.values,
    'Reintroduction Scenario': provision_female_v4_r1.values
})

provision_male_deviation_r1 = calculate_percentage_deviation(provision_male_combined_r1, 'AG 2024')
provision_female_deviation_r1 = calculate_percentage_deviation(provision_female_combined_r1, 'AG 2024')

print("Percentage Deviations for Provision (Males) for r=1%:")
print(provision_male_deviation_r1)

print("\nPercentage Deviations for Provision (Females) for r=1%:")
print(provision_female_deviation_r1)


#Second version: interest rate=2%
interest_rate_v2=0.02
#AG 2024
annuity_factor_male_r2 = calculate_annuity_factor(interest_rate_v2,t_years_survival_male, specific_ages)
annuity_factor_female_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_female, specific_ages)

provision_male_r2 = calculate_provision_op(U_x_table, annuity_factor_male_r2, specific_ages, 'male')
provision_female_r2 = calculate_provision_op(U_x_table, annuity_factor_female_r2, specific_ages, 'female')

provision_combined_r2 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male': provision_male_r2.values,
    'Provision Female': provision_female_r2.values})


#One off scenario - scenario 1
annuity_factor_male_v1_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_male_v1, specific_ages)
annuity_factor_female_v1_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_female_v1, specific_ages)

provision_male_v1_r2 = calculate_provision_op(U_x_table, annuity_factor_male_v1_r2, specific_ages, 'male')
provision_female_v1_r2 = calculate_provision_op(U_x_table, annuity_factor_female_v1_r2, specific_ages, 'female')
provision_combined_v1_r2 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for One off scenario': provision_male_v1_r2.values,
    'Provision Female for One off scenario': provision_female_v1_r2.values})

################
#Structurual scenario - scenario 2
annuity_factor_male_v2_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_male_v2, specific_ages)
annuity_factor_female_v2_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_female_v2, specific_ages)

# Calculating provision for males and females
provision_male_v2_r2 = calculate_provision_op(U_x_table, annuity_factor_male_v2_r2, specific_ages, 'male')
provision_female_v2_r2 = calculate_provision_op(U_x_table, annuity_factor_female_v2_r2, specific_ages, 'female')
provision_combined_v2_r2 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Structural scenario': provision_male_v2_r2.values,
    'Provision Female for Structural scenario': provision_female_v2_r2.values})

#New normal scenario - scenario 3
annuity_factor_male_v3_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_male_v3, specific_ages)
annuity_factor_female_v3_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_female_v3, specific_ages)

provision_male_v3_r2 = calculate_provision_op(U_x_table, annuity_factor_male_v3_r2, specific_ages, 'male')
provision_female_v3_r2 = calculate_provision_op(U_x_table,annuity_factor_female_v3_r2, specific_ages, 'female')
provision_combined_v3_r2 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for New Normal scenario': provision_male_v3_r2.values,
    'Provision Female for New Normal scenario': provision_female_v3_r2.values})

#Reintroduction scenario - scenario 4
annuity_factor_male_v4_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_male_v4, specific_ages)
annuity_factor_female_v4_r2 = calculate_annuity_factor(interest_rate_v2, t_years_survival_female_v4, specific_ages)

provision_male_v4_r2 = calculate_provision_op(U_x_table, annuity_factor_male_v4_r2, specific_ages, 'male')
provision_female_v4_r2 = calculate_provision_op(U_x_table, annuity_factor_female_v4_r2, specific_ages, 'female')
provision_combined_v4_r2 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Reintroduction scenario': provision_male_v4_r2.values,
    'Provision Female for Reintroduction scenario': provision_female_v4_r2.values})

#Provision values for all scenarios
provision_male_combined_r2 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_male_r2.values,
    'One-off Scenario': provision_male_v1_r2.values,
    'Structural Scenario': provision_male_v2_r2.values,
    'New Normal Scenario': provision_male_v3_r2.values,
    'Reintroduction Scenario': provision_male_v4_r2.values
})

# Combine provision values for females across all scenarios
provision_female_combined_r2 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_female_r2.values,
    'One-off Scenario': provision_female_v1_r2.values,
    'Structural Scenario': provision_female_v2_r2.values,
    'New Normal Scenario': provision_female_v3_r2.values,
    'Reintroduction Scenario': provision_female_v4_r2.values
})

provision_male_deviation_r2 = calculate_percentage_deviation(provision_male_combined_r2, 'AG 2024')
provision_female_deviation_r2 = calculate_percentage_deviation(provision_female_combined_r2, 'AG 2024')

print("Percentage Deviations for Provision (Males) for r=2%:")
print(provision_male_deviation_r2)

print("\nPercentage Deviations for Provision (Females) for r=2%:")
print(provision_female_deviation_r2)

#Third version: interest rate= 4%
interest_rate_v3=0.04
#AG 2024
annuity_factor_male_r3 = calculate_annuity_factor(interest_rate_v3,t_years_survival_male, specific_ages)
annuity_factor_female_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_female, specific_ages)

provision_male_r3 = calculate_provision_op(U_x_table, annuity_factor_male_r3, specific_ages, 'male')
provision_female_r3 = calculate_provision_op(U_x_table, annuity_factor_female_r3, specific_ages, 'female')

provision_combined_r3 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male': provision_male_r3.values,
    'Provision Female': provision_female_r3.values})


#One off scenario - scenario 1
annuity_factor_male_v1_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_male_v1, specific_ages)
annuity_factor_female_v1_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_female_v1, specific_ages)

provision_male_v1_r3 = calculate_provision_op(U_x_table, annuity_factor_male_v1_r3, specific_ages, 'male')
provision_female_v1_r3 = calculate_provision_op(U_x_table, annuity_factor_female_v1_r3, specific_ages, 'female')
provision_combined_v1_r3 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for One off scenario': provision_male_v1_r3.values,
    'Provision Female for One off scenario': provision_female_v1_r3.values})

################
#Structurual scenario - scenario 2
annuity_factor_male_v2_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_male_v2, specific_ages)
annuity_factor_female_v2_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_female_v2, specific_ages)

# Calculating provision for males and females
provision_male_v2_r3 = calculate_provision_op(U_x_table, annuity_factor_male_v2_r3, specific_ages, 'male')
provision_female_v2_r3 = calculate_provision_op(U_x_table, annuity_factor_female_v2_r3, specific_ages, 'female')
provision_combined_v2_r3 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Structural scenario': provision_male_v2_r3.values,
    'Provision Female for Structural scenario': provision_female_v2_r3.values})

#New normal scenario - scenario 3
annuity_factor_male_v3_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_male_v3, specific_ages)
annuity_factor_female_v3_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_female_v3, specific_ages)

provision_male_v3_r3 = calculate_provision_op(U_x_table, annuity_factor_male_v3_r3, specific_ages, 'male')
provision_female_v3_r3 = calculate_provision_op(U_x_table,annuity_factor_female_v3_r3, specific_ages, 'female')
provision_combined_v3_r3 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for New Normal scenario': provision_male_v3_r3.values,
    'Provision Female for New Normal scenario': provision_female_v3_r3.values})

#Reintroduction scenario - scenario 4
annuity_factor_male_v4_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_male_v4, specific_ages)
annuity_factor_female_v4_r3 = calculate_annuity_factor(interest_rate_v3, t_years_survival_female_v4, specific_ages)

provision_male_v4_r3 = calculate_provision_op(U_x_table, annuity_factor_male_v4_r3, specific_ages, 'male')
provision_female_v4_r3 = calculate_provision_op(U_x_table, annuity_factor_female_v4_r3, specific_ages, 'female')
provision_combined_v4_r3 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Reintroduction scenario': provision_male_v4_r3.values,
    'Provision Female for Reintroduction scenario': provision_female_v4_r3.values})

#Provision values for all scenarios
provision_male_combined_r3 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_male_r3.values,
    'One-off Scenario': provision_male_v1_r3.values,
    'Structural Scenario': provision_male_v2_r3.values,
    'New Normal Scenario': provision_male_v3_r3.values,
    'Reintroduction Scenario': provision_male_v4_r3.values
})

# Combine provision values for females across all scenarios
provision_female_combined_r3 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_female_r3.values,
    'One-off Scenario': provision_female_v1_r3.values,
    'Structural Scenario': provision_female_v2_r3.values,
    'New Normal Scenario': provision_female_v3_r3.values,
    'Reintroduction Scenario': provision_female_v4_r3.values
})

provision_male_deviation_r3 = calculate_percentage_deviation(provision_male_combined_r3, 'AG 2024')
provision_female_deviation_r3 = calculate_percentage_deviation(provision_female_combined_r3, 'AG 2024')

print("Percentage Deviations for Provision (Males) for r=4%:")
print(provision_male_deviation_r3)

print("\nPercentage Deviations for Provision (Females) for r=4%:")
print(provision_female_deviation_r3)

#Fourth version: interest rate= 5%
interest_rate_v4=0.05
#AG 2024
annuity_factor_male_r4 = calculate_annuity_factor(interest_rate_v4,t_years_survival_male, specific_ages)
annuity_factor_female_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_female, specific_ages)

provision_male_r4 = calculate_provision_op(U_x_table, annuity_factor_male_r4, specific_ages, 'male')
provision_female_r4 = calculate_provision_op(U_x_table, annuity_factor_female_r4, specific_ages, 'female')

provision_combined_r4 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male': provision_male_r4.values,
    'Provision Female': provision_female_r4.values})


#One off scenario - scenario 1
annuity_factor_male_v1_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_male_v1, specific_ages)
annuity_factor_female_v1_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_female_v1, specific_ages)

provision_male_v1_r4 = calculate_provision_op(U_x_table, annuity_factor_male_v1_r4, specific_ages, 'male')
provision_female_v1_r4 = calculate_provision_op(U_x_table, annuity_factor_female_v1_r4, specific_ages, 'female')
provision_combined_v1_r4 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for One off scenario': provision_male_v1_r4.values,
    'Provision Female for One off scenario': provision_female_v1_r4.values})

################
#Structurual scenario - scenario 2
annuity_factor_male_v2_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_male_v2, specific_ages)
annuity_factor_female_v2_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_female_v2, specific_ages)

# Calculating provision for males and females
provision_male_v2_r4 = calculate_provision_op(U_x_table, annuity_factor_male_v2_r4, specific_ages, 'male')
provision_female_v2_r4 = calculate_provision_op(U_x_table, annuity_factor_female_v2_r4, specific_ages, 'female')
provision_combined_v2_r4 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Structural scenario': provision_male_v2_r4.values,
    'Provision Female for Structural scenario': provision_female_v2_r4.values})

#New normal scenario - scenario 3
annuity_factor_male_v3_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_male_v3, specific_ages)
annuity_factor_female_v3_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_female_v3, specific_ages)

provision_male_v3_r4 = calculate_provision_op(U_x_table, annuity_factor_male_v3_r4, specific_ages, 'male')
provision_female_v3_r4 = calculate_provision_op(U_x_table,annuity_factor_female_v3_r4, specific_ages, 'female')
provision_combined_v3_r4= pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for New Normal scenario': provision_male_v3_r4.values,
    'Provision Female for New Normal scenario': provision_female_v3_r4.values})

#Reintroduction scenario - scenario 4
annuity_factor_male_v4_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_male_v4, specific_ages)
annuity_factor_female_v4_r4 = calculate_annuity_factor(interest_rate_v4, t_years_survival_female_v4, specific_ages)

provision_male_v4_r4 = calculate_provision_op(U_x_table, annuity_factor_male_v4_r4, specific_ages, 'male')
provision_female_v4_r4 = calculate_provision_op(U_x_table, annuity_factor_female_v4_r4, specific_ages, 'female')
provision_combined_v4_r4 = pd.DataFrame({
    'Age': specific_ages,
    'Provision Male for Reintroduction scenario': provision_male_v4_r4.values,
    'Provision Female for Reintroduction scenario': provision_female_v4_r4.values})

#Provision values for all scenarios
provision_male_combined_r4 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_male_r4.values,
    'One-off Scenario': provision_male_v1_r4.values,
    'Structural Scenario': provision_male_v2_r4.values,
    'New Normal Scenario': provision_male_v3_r4.values,
    'Reintroduction Scenario': provision_male_v4_r4.values
})

# Combine provision values for females across all scenarios
provision_female_combined_r4 = pd.DataFrame({
    'Age': specific_ages,
    'AG 2024': provision_female_r4.values,
    'One-off Scenario': provision_female_v1_r4.values,
    'Structural Scenario': provision_female_v2_r4.values,
    'New Normal Scenario': provision_female_v3_r4.values,
    'Reintroduction Scenario': provision_female_v4_r4.values
})

provision_male_deviation_r4 = calculate_percentage_deviation(provision_male_combined_r4, 'AG 2024')
provision_female_deviation_r4 = calculate_percentage_deviation(provision_female_combined_r4, 'AG 2024')

print("Percentage Deviations for Provision (Males) for r=5%:")
print(provision_male_deviation_r4)

print("\nPercentage Deviations for Provision (Females) for r=5%:")
print(provision_female_deviation_r4)


###################################xx
#Covid calculation
# Excel file with population data
file_path_male_population = 'Population male of age 55 to 90.xlsx'
file_path_female_population = 'Population female of age 55 to 90.xlsx'
male_population_df = pd.read_excel(file_path_male_population)
female_population_df = pd.read_excel(file_path_female_population)

# Part 1: Interpolation of population data to calculate weekly exposures
days_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
days_2023 = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

# Interpolation across the entire combined dataset (for both years)
def interpolate_daily_population(monthly_data_2022, monthly_data_2023, days_2022, days_2023):
    combined_monthly_data = pd.concat([monthly_data_2022, monthly_data_2023], ignore_index=True)
    first_of_each_month = pd.date_range(start=days_2022[0], end=days_2023[-1], freq='MS') # Get the first of each month from both 2022 and 2023 ('MS' is Month Start)
    total_days = np.concatenate([days_2022, days_2023])
    daily_population = pd.DataFrame(index=total_days)

    # Loop through each age group and interpolate the population between months
    for age in combined_monthly_data.columns[1:]: 
        population_by_month = combined_monthly_data[age].values # Extract monthly population values for the 1st of each month
        interpolated_daily = np.full(len(total_days), np.nan)
       
        interpolated_daily = np.interp(
            np.arange(len(total_days)),
            np.searchsorted(total_days, first_of_each_month[:-1]),  # Positions of the 1st of each month in total_days
            population_by_month[:-1]
        )
    
   # Handle December 2023 by extrapolating based on November's rate of change
        november_population = population_by_month[-2]
        december_population = population_by_month[-1]  
        days_in_november = 30
        days_in_december = 31
        december_start_idx = len(total_days) - days_in_december
        november_step = (december_population - november_population) / days_in_november # Calculate the daily change (step) for November
        
        for i in range(days_in_december):
            interpolated_daily[december_start_idx + i] = december_population + (i * november_step)

        daily_population[age] = interpolated_daily
    
    daily_population_2022 = daily_population.loc[days_2022]
    daily_population_2023 = daily_population.loc[days_2023]
    
    return daily_population_2022, daily_population_2023

daily_population_2022_male, daily_population_2023_male = interpolate_daily_population(male_population_df[male_population_df['Periods'].str.contains('2022')], male_population_df[male_population_df['Periods'].str.contains('2023')], days_2022, days_2023)
daily_population_2022_female, daily_population_2023_female = interpolate_daily_population(female_population_df[female_population_df['Periods'].str.contains('2022')], female_population_df[female_population_df['Periods'].str.contains('2023')], days_2022, days_2023)

# Defining the number of days in each week for 2022 and 2023
W_t_2022 = list(range(53))  # Weeks for year 2022
W_t_2023 = list(range(53))  # Weeks for year 2023
days_in_week_2022 = pd.Series([2] + [7] * 51 + [6], W_t_2022)
days_in_week_2023 = pd.Series([1] + [7] * 52, index=W_t_2023)


def calculate_weekly_exposures_by_year(daily_population, year, days_in_week, year_days):
    total_days_in_year = days_in_week.sum()
    age_groups = daily_population.columns
    weekly_exposures_by_year = {}
    year_mask = year_days.year == year
    weekly_dates_by_week = {} 

    # Week 0: Special case for the start of the year
    week_zero_start = pd.Timestamp(year, 1, 1)
    if year == 2022:
        week_zero_end = week_zero_start + pd.Timedelta(days=1)  # Capture first two days
    elif year == 2023:
        week_zero_end = week_zero_start + pd.Timedelta(days=0)  # Capture only Jan 1
    
    week_zero_mask = (year_days >= week_zero_start) & (year_days <= week_zero_end) & year_mask
    weekly_dates_by_week[0] = year_days[week_zero_mask]

    # Now group dates into weeks 1 to 53
    for week in range(1, 54):
        if year == 2022:
            week_start = pd.Timestamp(year, 1, 3) + pd.DateOffset(weeks=week - 1)  # First monday of year
        elif year == 2023:
            week_start = pd.Timestamp(year, 1, 2) + pd.DateOffset(weeks=week - 1)  # First monday of year
        
        week_end = week_start + pd.Timedelta(days=6)  # Sunday of the week
        
        if week_end.year != year: # Adjust week_end if it exceeds December 31st of the year
            week_end = pd.Timestamp(year, 12, 31)
        
        # Creating a mask for the current week
        week_mask = (year_days >= week_start) & (year_days <= week_end) & year_mask
        current_dates = year_days[week_mask]
       
        if len(current_dates) > 0:
            weekly_dates_by_week[week] = current_dates

    # Calculating exposures per age group
    for age in age_groups:
        # Calculating Week 0 exposure
        week_0_dates = weekly_dates_by_week[0]
        if len(week_0_dates) > 0:
            P_xd_t_sum_0 = daily_population.loc[week_zero_mask, age].sum()  # Sum of daily population for Week 0
            exposure_xw_0 = (len(week_0_dates) / total_days_in_year) * P_xd_t_sum_0
            weekly_exposures_by_year[(age, 0, year)] = exposure_xw_0  # Store Week 0 exposure

        # Calculating exposures for weeks 1 to 53
        for week, week_dates in weekly_dates_by_week.items():
            if week == 0:
                continue  # Skip Week 0 as it's already calculated
            week_mask = (year_days.isin(week_dates))  # Mask for current week dates
            P_xd_t_sum = daily_population.loc[week_mask, age].sum()  # Sum of daily population for this week
            exposure_xw = (days_in_week[week] / total_days_in_year) * P_xd_t_sum # Calculate exposure for the current week
            weekly_exposures_by_year[(age, week, year)] = exposure_xw

    return weekly_exposures_by_year

weekly_exposures_male_2022 = calculate_weekly_exposures_by_year(daily_population_2022_male, 2022, days_in_week_2022, days_2022)
weekly_exposures_male_2023 = calculate_weekly_exposures_by_year(daily_population_2023_male, 2023, days_in_week_2023, days_2023)
weekly_exposures_male = {**weekly_exposures_male_2022, **weekly_exposures_male_2023}

weekly_exposures_female_2022 = calculate_weekly_exposures_by_year(daily_population_2022_female, 2022, days_in_week_2022, days_2022)
weekly_exposures_female_2023 = calculate_weekly_exposures_by_year(daily_population_2023_female, 2023, days_in_week_2023, days_2023)
weekly_exposures_female = {**weekly_exposures_female_2022, **weekly_exposures_female_2023}
# Function to access male and female exposure values
def E_x_w_t_male(age, week, year):
    return weekly_exposures_male.get((age, week, year), 0)
def E_x_w_t_female(age, week, year):
    return weekly_exposures_female.get((age, week, year), 0)

# Part 2: Fitting a cyclic cubic spline to mortality data and combining with exposures
file_path = 'weekly mortality data 2016-2019 new.xlsx'
death_data = pd.read_excel(file_path)

# Exclude week 0 from the death data
death_data_filtered = death_data[(death_data['week'] != 0)]

# Aggregate deaths by week across all ages and sexes for 2016-2019
weekly_deaths_tot = death_data_filtered.groupby(['week'])[[2016, 2017, 2018, 2019]].sum().sum(axis=1)

# Add an extra value for week 53 which is the same as the first week (index 1, assuming week 0 is excluded)
weekly_deaths_tot = pd.concat([weekly_deaths_tot, pd.Series(weekly_deaths_tot.iloc[0], index=[53])])

# Normalize the weekly death counts by the total deaths across all weeks
D_tot_w= weekly_deaths_tot.values
D_tot_w_normalized= D_tot_w / np.sum(D_tot_w)


lambda_param = 0.03

# Weeks range from 1 to 53
weeks = np.arange(1, 54)
weeks_total=np.arange(0,54)
# Spline function to be optimized
def smoothing_objective(phi_values, weeks, D_tot_w_normalized, lambda_param):
    phi_0_53 = (phi_values[-1] + phi_values[0]) / 2  # Use Phi(52) and Phi(1)
    
    # Creating a new array for phi_values to account for Phi(0) and Phi(53)
    phi_adjusted = np.zeros(54)
    phi_adjusted[1:53] = phi_values 
    phi_adjusted[0] = phi_0_53  # Phi(0)
    phi_adjusted[-1] = phi_0_53  # Phi(53)
    # Fit term (residuals between data and the spline approximation)
    fit_term = np.sum((D_tot_w_normalized - phi_adjusted[1:]) ** 2)
    
    # Creating a cyclic cubic spline with periodic boundary conditions
    spline = CubicSpline(weeks_total,  phi_adjusted, bc_type='periodic')
    # Smoothness term (penalizes the second derivative)
    second_derivative = spline(weeks, 2)  # Evaluate the second derivative
    smoothness_term = np.trapz(second_derivative ** 2, weeks)  # Integrate over weeks
    
    # Objective function: balance between fit and smoothness
    objective = lambda_param * fit_term + (1 - lambda_param) * smoothness_term
    return objective

# Initial guess for the cyclic spline (could start with normalized deaths)
initial_phi_values = D_tot_w_normalized[:-1].copy()

# Performing optimization to minimize the objective function
result = minimize(smoothing_objective, initial_phi_values, 
                  args=(weeks, D_tot_w_normalized, lambda_param),
                  method='L-BFGS-B')

# Extracting the optimized values (for weeks 1 to 52)
optimized_phi = result.x

# Computing Phi(0) and Phi(53) using the boundary condition
phi_0_53 = (optimized_phi[-1] + optimized_phi[0]) / 2  # (Phi(52) + Phi(1)) / 2

# Creating the full array with Phi(0), Phi(1) to Phi(52), and Phi(53)
phi_adjusted = np.zeros(len(optimized_phi) + 2)  # Array of length 54 (weeks 0 to 53)
phi_adjusted[1:53] = optimized_phi               # Fill in Phi(1) to Phi(52)
phi_adjusted[0] = phi_0_53                       # Phi(0)
phi_adjusted[-1] = phi_0_53                      # Phi(53)

for week, phi_value in enumerate(phi_adjusted):
    print(f"Week {week}: Phi = {phi_value}")



# Function to calculate phi_w_t (seasonal effect), excluding week 53
def calculate_seasonal_effect(phi_values, days_in_week, year):
    # Include weeks 0 to 52, exclude week 53
    valid_weeks = np.arange(0, 53)  # Weeks from 0 to 52 (we exclude week 53)
    filtered_phi_values = phi_values[:53]  # Corresponding phi_values for weeks 0 to 52 (exclude week 53)
    total_days_in_year=np.sum(days_in_week)
    # Initializing a dictionary to store seasonal effects for each week
    seasonal_effect_w_t = {}

    # Calculating phi_w_t for each week w (from 0 to 52)
    for week in valid_weeks:
        numerator = filtered_phi_values[week] * total_days_in_year
        denominator = np.sum(filtered_phi_values * days_in_week)
        seasonal_effect_w_t[(week,year)] = numerator / denominator

    return seasonal_effect_w_t
# Example usage for 2022 and 2023
seasonal_effect_w_2022 = calculate_seasonal_effect(phi_adjusted, days_in_week_2022, 2022)
seasonal_effect_w_2023 = calculate_seasonal_effect(phi_adjusted, days_in_week_2023, 2023)

print("\nSeasonal Effect for 2022:")
print(seasonal_effect_w_2022)

print("\nSeasonal Effect for 2023:")
print(seasonal_effect_w_2023)

# Combining both dictionaries into a single one
seasonal_effects_dict = {**seasonal_effect_w_2022, **seasonal_effect_w_2023}

# Function to access seasonal effect based on week and year
def seasonal_effect_w_t(week, year):
    result = seasonal_effects_dict.get((week, year), None)
    return result

# Example usage
print(seasonal_effect_w_t(10, 2022))
print(seasonal_effect_w_t(15, 2023))
# File path to the Excel file
file_path = 'weekly mortality data 2022-2023_new.xlsx' 
excel_data = pd.ExcelFile(file_path)

# Load data from each sheet
male_2022 = excel_data.parse('Male_2022')
male_2023 = excel_data.parse('Male_2023')
female_2022 = excel_data.parse('Female_2022')
female_2023 = excel_data.parse('Female_2023')

# Combining data from all sheets into a single DataFrame
data_frames = [male_2022, male_2023, female_2022, female_2023]
combined_data = pd.concat(data_frames, ignore_index=True)

cleaned_data = combined_data[['Year', 'Week', 'Gender', 'Age', 'Death']]

# Splitting the data into male and female datasets
male_data = cleaned_data[cleaned_data['Gender'] == 'Male']
female_data = cleaned_data[cleaned_data['Gender'] == 'Female']

# Creating dictionaries where keys are (Age, Week, Year) tuples and values are Death counts
death_dict_male = male_data.set_index(['Age', 'Week', 'Year'])['Death'].to_dict()
death_dict_female = female_data.set_index(['Age', 'Week', 'Year'])['Death'].to_dict()

# Pre-COVID mortality rate values for ages 55 to 90 and years 2022 and 2023
mu_male_precovid = np.array([[0.00398893, 0.00391459],
       [0.00439543, 0.00431313],
       [0.00494567, 0.00484719],
       [0.00544526, 0.00533949],
       [0.00596992, 0.00585081],
       [0.00664532, 0.00651355],
       [0.00726744, 0.00712482],
       [0.00804489, 0.00788242],
       [0.00889854, 0.00872099],
       [0.00973851, 0.00954665],
       [0.01077074, 0.01055152],
       [0.01187372, 0.01163778],
       [0.01294962, 0.0126868 ],
       [0.01432013, 0.01402958],
       [0.01581729, 0.01550282],
       [0.01730717, 0.01696493],
       [0.01928447, 0.01892346],
       [0.02148028, 0.02108137],
       [0.02390757, 0.02347384],
       [0.02673499, 0.02625468],
       [0.02996914, 0.0294363 ],
       [0.03354333, 0.03298069],
       [0.03760943, 0.03702762],
       [0.04252385, 0.04190063],
       [0.04798209, 0.04728023],
       [0.05412305, 0.05340517],
       [0.06146126, 0.0607333 ],
       [0.06900906, 0.06823054],
       [0.07813149, 0.07734655],
       [0.08860147, 0.08775515],
       [0.10093382, 0.09994737],
       [0.11356311, 0.11264761],
       [0.12853292, 0.12757335],
       [0.145528  , 0.1447082 ],
       [0.16345586, 0.16263183],
       [0.18357043, 0.18286587]])
mu_female_precovid = np.array([[0.0031087 , 0.00306805],
       [0.00336505, 0.00331962],
       [0.00366646, 0.00361718],
       [0.00401425, 0.00395971],
       [0.00433541, 0.00427683],
       [0.00481372, 0.0047487 ],
       [0.00521946, 0.00514727],
       [0.00561924, 0.00554175],
       [0.00602206, 0.00593464],
       [0.00649419, 0.00639616],
       [0.00705251, 0.00694284],
       [0.00767764, 0.00755659],
       [0.00826785, 0.00813167],
       [0.00918089, 0.00902806],
       [0.01002853, 0.00985859],
       [0.01100818, 0.01081705],
       [0.01218103, 0.01196495],
       [0.01333385, 0.01309405],
       [0.01486036, 0.01458854],
       [0.01646114, 0.01615418],
       [0.0185079 , 0.01816167],
       [0.02080996, 0.02042213],
       [0.0235074 , 0.02306693],
       [0.02670443, 0.0262129 ],
       [0.0302278 , 0.02967798],
       [0.03512949, 0.03451277],
       [0.04041376, 0.03972514],
       [0.04679172, 0.04602393],
       [0.05406755, 0.0532188 ],
       [0.06209041, 0.06115547],
       [0.07215132, 0.07114166],
       [0.08409745, 0.08299463],
       [0.09760613, 0.0964259 ],
       [0.11162177, 0.11037147],
       [0.12889871, 0.12757176],
       [0.14949565, 0.14814223]])


# Age range for these values is from 55 to 90 (36 values)
ages = range(55, 91)

# Creating dictionaries to store mu_hat_pre_covid_x_t separately for male and female
mu_hat_pre_covid_male = {(age, year): mu_male_precovid[age_index, year_index] 
                         for age_index, age in enumerate(ages) 
                         for year_index, year in enumerate([2022, 2023])}

mu_hat_pre_covid_female = {(age, year): mu_female_precovid[age_index, year_index] 
                           for age_index, age in enumerate(ages) 
                           for year_index, year in enumerate([2022, 2023])}

########################################

# Defining ages, weeks, and years for reference
ages = range(55, 91)  # 36 ages
weeks = range(53)     # 0 to 52 (53 weeks)
years = [2022, 2023]  # Two years

# Initializing arrays for deaths (D) and exposures (E) with zeros
D_male = np.zeros((len(ages), len(weeks), len(years)))
E_male = np.zeros((len(ages), len(weeks), len(years)))
D_female = np.zeros((len(ages), len(weeks), len(years)))
E_female = np.zeros((len(ages), len(weeks), len(years)))

# Assuming `death_dict_male` and `exposure_dict_male` have (age, week, year) keys:
for age_index, age in enumerate(ages):
    for week_index, week in enumerate(weeks):
        for year_index, year in enumerate(years):
            D_male[age_index, week_index, year_index] = death_dict_male.get((age, week, year),0)
            E_male[age_index, week_index, year_index] = weekly_exposures_male.get((age, week, year),0)
            D_female[age_index, week_index, year_index] = death_dict_female.get((age, week, year),0)
            E_female[age_index, week_index, year_index] = weekly_exposures_female.get((age, week, year),0)
# Verify specific entries
age = 57  # Example age
week = 3  # Example week
year = 2023  # Example year
# Convert age, week, year to indices
age_index = ages.index(age)
week_index = week
year_index = years.index(year)
print(f"D_male value for Age {age}, Week {week}, Year {year}: {D_male[age_index, week_index, year_index]}")

# Initializing phi array (53 weeks, 2 years)
phi = np.zeros((53, 2))

for week_index, week in enumerate(weeks):
    for year_index, year in enumerate(years):
        phi[week_index, year_index] = seasonal_effects_dict.get((week, year),1) # Default to 1 if not found
def get_mu_pre_covid_array(mu_hat_pre_covid, ages, years):
    # Shape of (ages, weeks, years)
    mu_pre_covid = np.zeros((len(ages), 53, len(years)))
    for age_index,age in enumerate(ages):
        for year_index,year in enumerate(years):
            mu_pre_covid[age_index, :,year_index] = mu_hat_pre_covid.get((age,year),0)
    return mu_pre_covid

# Converting male and female pre-COVID mortality rates to 3D arrays
mu_pre_covid_male = get_mu_pre_covid_array(mu_hat_pre_covid_male, ages, years)
mu_pre_covid_female = get_mu_pre_covid_array(mu_hat_pre_covid_female, ages, years)


def vectorized_log_likelihood_B(B, k_combined, mu_pre_covid, phi, D, E):
    k_w_2022 = k_combined[:53]  # First 53 values for 2022
    k_w_2023 = k_combined[53:]  # Last 53 values for 2023

    k = np.stack([k_w_2022, k_w_2023], axis=-1)

    # Reshaping `B` to have dimensions (36, 1, 1), so it can broadcast with `k` and other terms
    B = B.reshape(-1, 1, 1)

    mu = mu_pre_covid * phi * np.exp(B * k)

    #Log-likelihood calculation
    log_likelihood_value = np.sum(D * np.log(mu) - E * mu)
    return -log_likelihood_value  # Return negative because we want to minimize

# Vectorized log-likelihood function for optimizing combined `k_w_2022` and `k_w_2023`
def vectorized_log_likelihood_combined_k(k_combined, B, mu_pre_covid, phi, D, E):
    k_w_2022 = k_combined[:53]  # First 53 values for 2022
    k_w_2023 = k_combined[53:]  # Last 53 values for 2023

    # Create a 3D array by stacking 2022 and 2023 `k_w_t` arrays
    k = np.stack([k_w_2022, k_w_2023], axis=-1)

    B = B.reshape(-1, 1, 1)

    mu = mu_pre_covid * phi * np.exp(B * k)
    
    # Log-likelihood calculation
    log_likelihood_value = np.sum(D * np.log(mu) - E * mu)
    return -log_likelihood_value  # Return negative because we want to minimize

# Function to optimize B, k_w_2022, and k_w_2023 simultaneously
def optimize_parameters_combined(B, k, mu_pre_covid, phi, D, E, max_iterations=500, tol=1e-6):
    n_ages = len(B)

    # Constraints for B
    Beta_constraint = LinearConstraint(np.ones(n_ages), 1, 1)  # Sum of B must equal 1
    Beta_bounds = Bounds(0, 1)  # Each B value must be between 0 and 1


    for iteration in range(max_iterations):
        # Step 1: Optimize B values
        opt_result_B = minimize(
            vectorized_log_likelihood_B, 
            B, 
            args=(k, mu_pre_covid, phi, D, E),
            method='SLSQP', 
            constraints=Beta_constraint, 
            bounds=Beta_bounds,
            tol=tol
        )
        B = opt_result_B.x

        # Step 2: Optimize combined k_w_t values
        opt_result_k_combined = minimize(
            vectorized_log_likelihood_combined_k, 
            k, 
            args=(B, mu_pre_covid, phi, D, E),
            method='L-BFGS-B',
            tol=tol
        )
        k = opt_result_k_combined.x
        if not opt_result_k_combined.success:
            print(f"Optimization failed for k_combined at iteration {iteration + 1}: {opt_result_k_combined.message}")

        # Calculate log-likelihood to track convergence
        current_log_likelihood = vectorized_log_likelihood_combined_k(k, B, mu_pre_covid, phi, D, E)

        print(f"Iteration {iteration + 1}: Log-Likelihood = {current_log_likelihood}")

        # Check for convergence (change in log-likelihood values)
        if opt_result_B.success and opt_result_k_combined.success:
            if np.abs(opt_result_B.fun - opt_result_k_combined.fun) < tol:
                print(f'Convergence achieved after {iteration + 1} iterations.')
                break


    # Splitting the combined k back into 2022 and 2023 components
    k_w_2022_final = k[:53]
    k_w_2023_final = k[53:]

    return B, k_w_2022_final, k_w_2023_final

# Initial guesses
n_age_groups = 36
B_initial= np.full(n_age_groups, 1 / n_age_groups)  # Random initial values for B that sum to 1
k_values_2022 = np.random.uniform(-1, 6, 53)
k_values_2023 = np.random.uniform(-1, 6, 53)
k_initial = np.concatenate((k_values_2022, k_values_2023))
# Optimizing parameters for males
B_optimized_male, k_w_2022_optimized_male, k_w_2023_optimized_male = optimize_parameters_combined(B_initial,k_initial, mu_pre_covid_male, phi, D_male, E_male)
# Check if the sum of B values equals 1
sum_B_male = np.sum(B_optimized_male)
if np.isclose(sum_B_male, 1.0, atol=1e-6):
    print("The sum of optimized B values for males is approximately 1.")
else:
    print(f"The sum of optimized B values for males is {sum_B_male}, which is not equal to 1.")

ages = range(55, 91)  # Ages 55 to 90

#B values by age
print("Final Optimized B values by Age (Male):")
for age, B_value in zip(ages, B_optimized_male):
    print(f"Age {age}: B = {B_value:.6f}")

#k_w values for each week in 2022 and 2023
print("\nFinal Optimized k_w values for each week in 2022 (Male):")
for week, k_value_2022 in enumerate(k_w_2022_optimized_male):
    print(f"Week {week}: k_w_2022 = {k_value_2022:.6f}")

print("\nFinal Optimized k_w values for each week in 2023 (Male):")
for week, k_value_2023 in enumerate(k_w_2023_optimized_male):
    print(f"Week {week}: k_w_2023 = {k_value_2023:.6f}")

# Optimizing parameters for females
B_optimized_female, k_w_2022_optimized_female, k_w_2023_optimized_female = optimize_parameters_combined(B_initial,k_initial,mu_pre_covid_female, phi, D_female, E_female)
# Checking if the sum of B values equals 1
sum_B_female = np.sum(B_optimized_female)
if np.isclose(sum_B_female, 1.0, atol=1e-6):
    print("The sum of optimized B values for females is approximately 1.")
else:
    print(f"The sum of optimized B values for females is {sum_B_female}, which is not equal to 1.")
# B values by age
print("Final Optimized B values by Age (Female):")
for age, B_value in zip(ages, B_optimized_female):
    print(f"Age {age}: B = {B_value:.6f}")

# k_w values for each week in 2022 and 2023
print("\nFinal Optimized k_w values for each week in 2022 (Female):")
for week, k_value_2022 in enumerate(k_w_2022_optimized_female):
    print(f"Week {week}: k_w_2022 = {k_value_2022:.6f}")

print("\nFinal Optimized k_w values for each week in 2023 (Female):")
for week, k_value_2023 in enumerate(k_w_2023_optimized_female):
    print(f"Week {week}: k_w_2023 = {k_value_2023:.6f}")
###########
#Determining frak x tilde
# Function to calculate frak X_tilde for a given year and gender
def calculate_frak_x_tilde_t(phi_w_t, B_x_g, k_w_t, days_in_week, year):
    xi_t = 0 
    k_w_values = k_w_t[year]
    
    # Loop over ages from 55 to 90
    for age in range(55, 91):
        inner_sum = 0  # Inner sum over weeks
        
        for week in days_in_week.index:
            # Seasonal effect
            phi = phi_w_t(week, year)
            
            # Getting the number of days in the current week (N_{w,t})
            days = days_in_week[week]
            
            # Total days calculation
            total_days = days_in_week.sum()
            # Fraction of days in the week
            fraction = days / total_days
            
            # Retrieving B_x and corresponding kappa
            B_x = B_x_g[age - 55] 
            k_w = k_w_values[week]
            exp_term = np.exp(B_x * k_w)

            # Update inner sum
            inner_sum += phi * fraction * exp_term
        
        # Add the log of the inner sum to xi_t
        xi_t += np.log(inner_sum)
    
    return xi_t

# Defining k_w_t dictionaries for male and female using optimized values from previous code
k_w_t_male = {
    2022: k_w_2022_optimized_male,
    2023: k_w_2023_optimized_male
}
k_w_t_female = {
    2022: k_w_2022_optimized_female,
    2023: k_w_2023_optimized_female
}

# Calculating frak X_tilde for male and female in 2022 and 2023
frak_x_tilde_2022_male = calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_male, k_w_t_male, days_in_week_2022, 2022)
frak_x_tilde_2023_male = calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_male, k_w_t_male, days_in_week_2023, 2023)

frak_x_tilde_2022_female = calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_female, k_w_t_female, days_in_week_2022, 2022)
frak_x_tilde_2023_female = calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_female, k_w_t_female, days_in_week_2023, 2023)


print("Calculated frak_x_tilde for 2022:")
print(f"Male: {frak_x_tilde_2022_male}, Female: {frak_x_tilde_2022_female}")

print("\nCalculated frak_x_tilde for 2023:")
print(f"Male: {frak_x_tilde_2023_male}, Female: {frak_x_tilde_2023_female}")

frak_x_tilde_male = {
    2022: calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_male, k_w_t_male, days_in_week_2022, 2022),
    2023: calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_male, k_w_t_male, days_in_week_2023, 2023)
}

frak_x_tilde_female = {
    2022: calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_female, k_w_t_female, days_in_week_2022, 2022),
    2023: calculate_frak_x_tilde_t(seasonal_effect_w_t, B_optimized_female, k_w_t_female, days_in_week_2023, 2023)
}

#Calculating frak x beta using an equation
#Defining the left hand side of the equation
def calculate_lhs(frak_tilde_beta, mu_pre_covid_x, frak_x_tilde_year):
    lhs_product = 1.0
    for year in [2022, 2023]:
        lhs_year_term = np.exp(-mu_pre_covid_x[year] * np.exp(frak_tilde_beta * frak_x_tilde_year[year]))
        lhs_product *= lhs_year_term
    return lhs_product
 #Defining the right hand side of the equation
def calculate_rhs(frak_beta_x, mu_pre_covid_x, phi_w_t, k_w_t, days_in_week, year):
    rhs_product = 1.0
    total_days = days_in_week.sum()
    
    for week, N_w_t in days_in_week.items():
        fraction = N_w_t / total_days  # Calculate N_w,t / sum_u N_u,t
        phi = phi_w_t(week, year)  # Seasonal effect for the week
        k_w_value = k_w_t[year][week]
        
        # Weekly term inside the exponential
        exp_term = -fraction * mu_pre_covid_x[year] * phi * np.exp(frak_beta_x * k_w_value)
        rhs_week_term = np.exp(exp_term)
        
        # Multiply this week's term into the product
        rhs_product *= rhs_week_term
    
    return rhs_product

def equation_to_solve(frak_tilde_beta, frak_beta_x, mu_pre_covid_x, phi_w_t, frak_x_tilde_year, k_w_t, days_in_week_2022, days_in_week_2023):
    # Computing the left-hand side for both years
    lhs = calculate_lhs(frak_tilde_beta, mu_pre_covid_x, frak_x_tilde_year)

    # Computing the right-hand side for both years
    rhs_2022 = calculate_rhs(frak_beta_x, mu_pre_covid_x, phi_w_t, k_w_t, days_in_week_2022, 2022)
    rhs_2023 = calculate_rhs(frak_beta_x, mu_pre_covid_x, phi_w_t, k_w_t, days_in_week_2023, 2023)
    rhs = rhs_2022 * rhs_2023

    # Return the difference (we aim to find the root of this difference)
    return lhs - rhs

def solve_for_frak_tilde_beta(frak_beta_x, mu_pre_covid_x, phi_w_t, frak_x_tilde_year, k_w_t, days_in_week_2022, days_in_week_2023):
    initial_guess = 0.01
    
    # Call the root-finding function
    solution = root(
        equation_to_solve,
        initial_guess,
        args=(frak_beta_x, mu_pre_covid_x, phi_w_t, frak_x_tilde_year, k_w_t, days_in_week_2022, days_in_week_2023),
        method='hybr',
        tol=1e-6
    )
    
    # Check if solution was found
    if solution.success:
        return solution.x[0]
    else:
        print(f"Root finding failed: {solution.message}")
        return None

# Defining parameters
ages = range(55, 91)
frak_tilde_beta_male_values = []
frak_tilde_beta_female_values = []


# Computing frak_tilde_beta for each age for males
print("Calculating frak_tilde_beta for males:")
for age_index, age in enumerate(ages):
    frak_beta_x_male = B_optimized_male[age_index]
    mu_pre_covid_x_male = {
        2022: mu_hat_pre_covid_male.get((age, 2022), 0),
        2023: mu_hat_pre_covid_male.get((age, 2023), 0)
    }
    frak_x_tilde_year_male = {
        2022: frak_x_tilde_male[2022],
        2023: frak_x_tilde_male[2023]
    }
    
    # Solve for frak_tilde_beta for this age (male)
    frak_tilde_beta_male = solve_for_frak_tilde_beta(
        frak_beta_x_male, mu_pre_covid_x_male, seasonal_effect_w_t, frak_x_tilde_year_male, k_w_t_male, days_in_week_2022, days_in_week_2023
    )
    
    # Store the result for male
    frak_tilde_beta_male_values.append(frak_tilde_beta_male)
    print(f"Age {age} (Male): frak_tilde_beta = {frak_tilde_beta_male}")

# Computing frak_tilde_beta for each age for females
print("\nCalculating frak_tilde_beta for females:")
for age_index, age in enumerate(ages):
    frak_beta_x_female = B_optimized_female[age_index]
    mu_pre_covid_x_female = {
        2022: mu_hat_pre_covid_female.get((age, 2022), 0),
        2023: mu_hat_pre_covid_female.get((age, 2023), 0)
    }
    frak_x_tilde_year_female = {
        2022: frak_x_tilde_female[2022],
        2023: frak_x_tilde_female[2023]
    }
    
    # Solve for frak_tilde_beta for this age (female)
    frak_tilde_beta_female = solve_for_frak_tilde_beta(
        frak_beta_x_female, mu_pre_covid_x_female, seasonal_effect_w_t, frak_x_tilde_year_female, k_w_t_female, days_in_week_2022, days_in_week_2023
    )
    
    # Store the result for female
    frak_tilde_beta_female_values.append(frak_tilde_beta_female)
    print(f"Age {age} (Female): frak_tilde_beta = {frak_tilde_beta_female}")

#Final frak beta values for ages 55 to 90
# Step 1: Normalize frak tilde_B_x values to obtain final frak_B_x
def normalize_frak_B_x(frak_tilde_B_x_solutions, ages):
    # Calculate the sum of all frak_tilde_B_x values
    total_sum = sum(frak_tilde_B_x_solutions)
    
    # Dictionary to store the normalized frak_B_x values with ages as keys
    normalized_frak_B_x = {}
    
    # Normalize each frak_tilde_B_x value by dividing by the total sum
    for age, value in zip(ages, frak_tilde_B_x_solutions):
        # Apply normalization formula
        normalized_frak_B_x[age] = value / total_sum
    
    return normalized_frak_B_x

ages = range(55, 91)


final_frak_B_x_male = normalize_frak_B_x(frak_tilde_beta_male_values, ages)
final_frak_B_x_female = normalize_frak_B_x(frak_tilde_beta_female_values, ages)

print("Final frak B_x values by Age (Male):")
for age, B_value in final_frak_B_x_male.items():
    print(f"Age {age}: B = {B_value:.6f}")

print("\nFinal frak B_x values by Age (Female):")
for age, B_value in final_frak_B_x_female.items():
    print(f"Age {age}: B = {B_value:.6f}")


total_normalized_sum_male = sum(final_frak_B_x_male.values())
print("Total sum of normalized frak_B_x for male:", total_normalized_sum_male)
total_normalized_sum_female = sum(final_frak_B_x_female.values())
print("Total sum of normalized frak_B_x for female:", total_normalized_sum_female)


# Step 2: Calculate frak x_t^g
def calculate_x_t_g(frak_B_x_solutions, frak_x_tilde_values):
    # Sum of double_frak_B_x values across the ages (55 to 90)
    total_sum_frak_B_x = sum(frak_B_x_solutions)
    
    # Compute x_t^g using the provided formula
    x_t_g = frak_x_tilde_values * total_sum_frak_B_x
    
    return x_t_g


# Compute x_t^g for both years
frak_x_t_2022_male = calculate_x_t_g(frak_tilde_beta_male_values, frak_x_tilde_2022_male)
frak_x_t_2023_male = calculate_x_t_g(frak_tilde_beta_male_values, frak_x_tilde_2023_male)
frak_x_t_2022_female = calculate_x_t_g(frak_tilde_beta_female_values, frak_x_tilde_2022_female)
frak_x_t_2023_female = calculate_x_t_g(frak_tilde_beta_female_values, frak_x_tilde_2023_female)

print("Calculated frak_x for 2022:")
print(f"Male: {frak_x_t_2022_male}, Female: {frak_x_t_2022_female}")

print("\nCalculated frak_x for 2023:")
print(f"Male: {frak_x_t_2023_male}, Female: {frak_x_t_2023_female}")


def full_frak_B_x(final_frak_B_x):
    # Creating a new dictionary to store the full range from ages 0 to 120
    full_frak_B_x = {}

    # Setting values to 0 for ages 0 to 54
    for age in range(0, 55):
        full_frak_B_x[age] = 0

    # Beta values for ages 55 to 90
    for age in range(55, 91):
        full_frak_B_x[age] = final_frak_B_x.get(age, 0)

    # Setting values equal to the value at age 90 for ages 91 to 120
    value_at_90 = final_frak_B_x.get(90, 0)
    for age in range(91, 121):
        full_frak_B_x[age] = value_at_90

    return full_frak_B_x

full_frak_B_x_male = full_frak_B_x(final_frak_B_x_male)
full_frak_B_x_female = full_frak_B_x(final_frak_B_x_female)

print("Full frak B_x values by Age (Male):")
for age in range(0, 121):
    print(f"Age {age}: B = {full_frak_B_x_male[age]:.6f}")

print("Full frak B_x values by Age (Female):")
for age in range(0, 121):
    print(f"Age {age}: B = {full_frak_B_x_female[age]:.6f}")
