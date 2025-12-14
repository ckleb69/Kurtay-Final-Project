import os
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.integrate import quad
from scipy.optimize import minimize
import lmoments3 as lm
import warnings

# Suppress integration warnings for cleaner output
warnings.filterwarnings("ignore")

# ===============================================================
# CONFIGURATION & PATHS
# ===============================================================
# Matches your existing folder structure
BASE_DIR = r"C:\Kurtay Finance Project\Data"
RESID_DIR = os.path.join(BASE_DIR, "Student_Residuals")
OUT_FILE  = os.path.join(BASE_DIR, "PearsonVII_Lmoments_Fit.csv")

# ===============================================================
# CORE LOGIC: THEORETICAL L-MOMENTS FOR PEARSON VII
# ===============================================================
def pearson7_iso_student_t(m, scale):
    """
    Converts Pearson VII parameters (m, scale) to 
    Student-t degrees of freedom (nu) and scaling factor.
    
    Pearson VII PDF: f(x) ~ (1 + (x/scale)^2)^(-m)
    Student-t PDF:   f(y) ~ (1 + y^2/nu)^(-(nu+1)/2)
    
    Mapping:
      - nu = 2*m - 1
      - x = y * (scale / sqrt(nu))
    """
    nu = 2 * m - 1
    # Scale factor to convert Student-t quantile to Pearson quantile
    # Q_pearson(u) = Q_student(u, df=nu) * factor
    factor = scale / np.sqrt(nu)
    return nu, factor

def get_theoretical_lmoms(m, scale):
    """
    Calculates Theoretical L2 (L-scale) and Tau4 (L-kurtosis) 
    for a Pearson Type VII distribution using numerical integration 
    of the Student-t Quantile Function.
    """
    # Constraint: Mean must exist for L-moments (nu > 1 => m > 1)
    if m <= 1.01 or scale <= 0:
        return np.inf, np.inf

    nu, factor = pearson7_iso_student_t(m, scale)

    # --- L2 (L-Scale) ---
    # Formula: Integral[0,1] of (2F - 1) * Q(F) dF
    # Since distribution is symmetric zero-mean, we can integrate 
    # Q(F)*(2F-1) from 0 to 1.
    def integrand_l2(f):
        return t.ppf(f, df=nu) * (2*f - 1)
    
    # Integrate
    raw_l2, _ = quad(integrand_l2, 0, 1)
    theo_l2 = raw_l2 * factor

    # --- L4 (for L-Kurtosis) ---
    # Formula: Integral[0,1] of (20F^3 - 30F^2 + 12F - 1) * Q(F) dF
    def integrand_l4(f):
        poly = 20*f**3 - 30*f**2 + 12*f - 1
        return t.ppf(f, df=nu) * poly

    raw_l4, _ = quad(integrand_l4, 0, 1)
    theo_l4 = raw_l4 * factor

    # Tau4 = L4 / L2
    if theo_l2 == 0: return np.inf, np.inf
    theo_tau4 = theo_l4 / theo_l2

    return theo_l2, theo_tau4

# ===============================================================
# OPTIMIZATION OBJECTIVE
# ===============================================================
def lmom_objective(params, sample_l2, sample_tau4):
    """
    Objective function to minimize:
    Squared Error between Sample L-moments and Theoretical L-moments.
    """
    m, scale = params
    
    # Soft constraints via penalty
    if m <= 1.01 or scale <= 1e-6:
        return 1e9

    theo_l2, theo_tau4 = get_theoretical_lmoms(m, scale)
    
    # Calculate weighted squared error
    # We weight them roughly equally (Tau4 is small, L2 is small)
    err_l2   = (theo_l2 - sample_l2)**2
    err_tau4 = (theo_tau4 - sample_tau4)**2
    
    # You can tune weights if needed, but 1:1 usually works for these magnitudes
    return err_l2 + err_tau4

# ===============================================================
# MAIN EXECUTION LOOP
# ===============================================================
def main():
    if not os.path.exists(RESID_DIR):
        print(f"Error: Directory not found: {RESID_DIR}")
        return

    files = [f for f in os.listdir(RESID_DIR) if f.endswith("_resid.csv")]
    results = []

    print(f"\n🚀 Starting L-Moment Estimation for {len(files)} tickers...")
    print(f"{'Ticker':<10} | {'Sample L2':<10} | {'Sample t4':<10} | {'Est m':<8} | {'Est Scale':<10}")
    print("-" * 65)

    for filename in files:
        ticker = filename.replace("_resid.csv", "")
        filepath = os.path.join(RESID_DIR, filename)
        
        try:
            # 1. Load Data
            df = pd.read_csv(filepath)
            data = df["std_resid"].dropna()
            
            if len(data) < 20:
                print(f"{ticker}: Not enough data")
                continue

            # 2. Calculate Sample L-Moments
            # lmom_ratios returns [L1, L2, Tau3, Tau4, ...]
            lm_vals = lm.lmom_ratios(data.tolist(), nmom=4)
            sample_l2 = lm_vals[1]
            sample_tau4 = lm_vals[3]

            # 3. Optimize (Method of L-Moments)
            # Initial guess: m=4 (similar to nu=7), scale=1 (standard)
            initial_guess = [4.0, 1.0]
            bounds = [(1.02, 50.0), (0.01, 10.0)] # m > 1.02, scale > 0

            res = minimize(
                lmom_objective, 
                initial_guess, 
                args=(sample_l2, sample_tau4),
                bounds=bounds,
                method='L-BFGS-B'
            )

            # 4. Store Results
            m_est, scale_est = res.x
            success_flag = res.success
            final_obj = res.fun

            # Calculate equivalent Student-t Nu for reference
            equiv_nu = 2 * m_est - 1

            results.append({
                "ticker": ticker,
                "n_obs": len(data),
                "sample_L2": sample_l2,
                "sample_tau4": sample_tau4,
                "est_m_lmom": m_est,
                "est_scale_lmom": scale_est,
                "equivalent_nu": equiv_nu,
                "optimization_error": final_obj,
                "converged": success_flag
            })

            print(f"{ticker:<10} | {sample_l2:<10.4f} | {sample_tau4:<10.4f} | {m_est:<8.4f} | {scale_est:<10.4f}")

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # ===============================================================
    # SAVE & SUMMARY
    # ===============================================================
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)
    
    print("-" * 65)
    print(f"\n✅ Processing complete. Results saved to:")
    print(f"   {OUT_FILE}")

if __name__ == "__main__":
    main()
    