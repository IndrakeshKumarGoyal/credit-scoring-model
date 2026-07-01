import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_credit(n_samples=1500, random_state=42):
    """
    Generates a realistic synthetic credit scoring dataset.
    
    Features:
    - Age (21 - 75)
    - Annual_Income (in USD)
    - Total_Debt (in USD)
    - Monthly_Payment (current debt obligations per month)
    - Credit_Utilization_Ratio (0.0 to 1.0)
    - Num_Open_Credit_Lines (1 - 20)
    - Num_Late_Payments_1Yr (0 - 10)
    - Employment_Duration_Years (0 - 40)
    - Housing_Status ('Own', 'Rent', 'Mortgage')
    - Savings_Balance (in USD)
    
    Target:
    - Credit_Default (0 = No Default/Good, 1 = Default/Bad)
    """
    np.random.seed(random_state)
    
    # 1. Generate core features
    age = np.random.randint(21, 76, size=n_samples)
    
    # Annual income: lognormal distribution to make it right-skewed and realistic
    annual_income = np.random.lognormal(mean=11.0, sigma=0.5, size=n_samples).astype(int)
    # Clamp to reasonable values ($15k to $250k)
    annual_income = np.clip(annual_income, 15000, 250000)
    
    # Employment duration: correlated with age
    emp_years = np.zeros(n_samples)
    for i in range(n_samples):
        max_emp = max(0, age[i] - 18)
        emp_years[i] = np.random.beta(a=2, b=5) * max_emp
    emp_years = np.round(emp_years, 1)
    
    # Housing status
    housing_opts = ['Rent', 'Mortgage', 'Own']
    # Rent is more common for younger/lower income, Own is more common for older/higher income
    housing_prob_matrix = []
    for i in range(n_samples):
        if annual_income[i] < 45000:
            probs = [0.65, 0.25, 0.10]
        elif annual_income[i] < 90000:
            probs = [0.30, 0.55, 0.15]
        else:
            probs = [0.10, 0.60, 0.30]
        housing_prob_matrix.append(probs)
        
    housing_status = [np.random.choice(housing_opts, p=probs) for probs in housing_prob_matrix]
    housing_status = np.array(housing_status)
    
    # Debt: Correlated with income
    dti_target = np.random.beta(a=2, b=4, size=n_samples) * 0.8  # Target DTI between 0 and 0.8
    total_debt = (annual_income * dti_target).astype(int)
    
    # Monthly payments: standard mortgage/loan amortization estimate (approx 1% of total debt per month)
    monthly_payment = (total_debt * np.random.uniform(0.008, 0.015, size=n_samples)).astype(int)
    # If debt is 0, payment is 0
    monthly_payment = np.where(total_debt == 0, 0, monthly_payment)
    
    # Credit utilization ratio: Beta distribution, centered around 0.30 but with some high risk users
    credit_util = np.random.beta(a=1.5, b=3.5, size=n_samples)
    
    # Number of open credit lines: related to age and income
    num_lines = np.random.poisson(lam=5 + (annual_income/30000) + (age/15)).astype(int)
    num_lines = np.clip(num_lines, 1, 20)
    
    # Late payments in last year: Poisson distribution, higher utilization -> higher rate of late payments
    late_lam = np.clip(credit_util * 2.5 + np.random.uniform(0, 1, size=n_samples) - 0.5, 0, None)
    num_late_payments = np.random.poisson(lam=late_lam).astype(int)
    
    # Savings balance: lognormal, correlated with income and age
    savings = (np.random.lognormal(mean=8.5, sigma=1.2, size=n_samples) * (annual_income / 50000) * (age / 35)).astype(int)
    savings = np.clip(savings, 0, 500000)
    
    # 2. Generate target probability (logistic model representation)
    # We construct a logit score representing risk of default
    # Higher value = Higher probability of default
    logit = (
        -1.8  # Intercept
        + 4.5 * credit_util                          # High utilization increase default risk
        + 0.8 * num_late_payments                     # Late payments strongly increase default risk
        + 1.5 * (total_debt / annual_income)         # High DTI ratio increases default risk
        - 0.04 * emp_years                            # More employment years decreases risk
        - 0.00001 * annual_income                     # Higher income decreases risk
        - 0.00002 * savings                           # Savings buffer decreases risk
        + 0.5 * (housing_status == 'Rent')            # Renters have slightly higher risk in synthetic model
        - 0.3 * (housing_status == 'Own')             # Owners have lower risk
        + 0.05 * (num_lines > 12)                     # Too many open lines slightly increases risk
        - 0.01 * (age - 35)                           # Age effect (middle-aged slightly lower risk)
    )
    
    # Sigmoid function
    prob = 1 / (1 + np.exp(-logit))
    
    # Add random noise to probability to prevent deterministic relationship
    prob = np.clip(prob + np.random.normal(0, 0.05, size=n_samples), 0, 1)
    
    # Generate binary default outcome (Bernoulli trial)
    credit_default = np.random.binomial(n=1, p=prob)
    
    # 3. Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': annual_income,
        'Total_Debt': total_debt,
        'Monthly_Payment': monthly_payment,
        'Credit_Utilization_Ratio': np.round(credit_util, 3),
        'Num_Open_Credit_Lines': num_lines,
        'Num_Late_Payments_1Yr': num_late_payments,
        'Employment_Duration_Years': emp_years,
        'Housing_Status': housing_status,
        'Savings_Balance': savings,
        'Credit_Default': credit_default
    })
    
    logging.info(f"Generated synthetic dataset with {df.shape[0]} samples and {df.shape[1]} features.")
    return df

def load_german_credit():
    """
    Fetches the German Credit dataset from OpenML.
    Returns a pandas DataFrame, or None if downloading fails.
    """
    try:
        from sklearn.datasets import fetch_openml
        logging.info("Attempting to fetch German Credit dataset from OpenML...")
        # openml ID 31 is the German Credit dataset ('credit-g')
        bunch = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
        df = bunch.frame
        
        # Clean target: class has values 'good' and 'bad'.
        # Let's map 'good' to 0 (no default) and 'bad' to 1 (default)
        if 'class' in df.columns:
            df['Credit_Default'] = df['class'].map({'good': 0, 'bad': 1}).astype(int)
            df = df.drop(columns=['class'])
        
        # Rename columns to standard CamelCase for visual consistency
        rename_map = {
            'checking_status': 'Checking_Status',
            'duration': 'Duration_Months',
            'credit_history': 'Credit_History',
            'purpose': 'Purpose',
            'credit_amount': 'Credit_Amount',
            'savings_status': 'Savings_Status',
            'employment': 'Employment_Duration',
            'installment_commitment': 'Installment_Rate',
            'personal_status': 'Personal_Status',
            'other_parties': 'Other_Parties',
            'residence_since': 'Residence_Since_Years',
            'property_magnitude': 'Property_Magnitude',
            'age': 'Age',
            'other_payment_plans': 'Other_Payment_Plans',
            'housing': 'Housing_Status',
            'existing_credits': 'Num_Existing_Credits',
            'job': 'Job_Type',
            'num_dependents': 'Num_Dependents',
            'own_telephone': 'Has_Telephone',
            'foreign_worker': 'Is_Foreign_Worker'
        }
        df = df.rename(columns=rename_map)
        logging.info(f"Successfully loaded German Credit dataset from OpenML: {df.shape[0]} samples, {df.shape[1]} columns.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch German Credit from OpenML: {e}")
        return None

if __name__ == "__main__":
    # Quick visual check
    df_syn = generate_synthetic_credit(5)
    print("Synthetic sample:\n", df_syn.head())
