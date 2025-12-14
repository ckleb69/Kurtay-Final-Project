import os
import numpy as np
import pandas as pd
from math import log, pi
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.stats import t as student_t_dist
from scipy.integrate import quad
import lmoments3 as lm
import warnings

warnings.filterwarnings("ignore")

# ===============================================================
# PATHS
# ===============================================================
BASE = r"C:\Kurtay Finance Project\Data"
RESID_DIR = os.path.join(BASE, "Student_Residuals")
OUT_FILE  = os.path.join(BASE, "ResidualOnly_Comparison_Lmoments.csv")

# ===============================================================
# L-MOMENT LOGIC (The New "Engine")
# ===============================================================
def get_theoretical_lmoms(m, scale):
    if m <= 1.01 or scale <= 0: return np.inf, np.inf
    nu = 2 * m - 1
    factor = scale / np.sqrt(nu)
    
    # L2
    lam2_raw, _ = quad(lambda f: student_t_dist.ppf(f, df=nu) * (2*f - 1), 0, 1)
    # L4
    lam4_raw, _ = quad(lambda f: student_t_dist.ppf(f, df=nu) * (20*f**3 - 30*f**2 + 12*f - 1), 0, 1)
    
    return lam2_raw * factor, (lam4_raw * factor) / (lam2_raw * factor)

def fit_pearson7_lmom(data):
    # 1. Calc Sample L-moments
    try:
        lm_vals = lm.lmom_ratios(data, nmom=4)
        sample_l2, sample_tau4 = lm_vals[1], lm_vals[3]
    except:
        return np.nan, np.nan, False

    # 2. Optimize
    def objective(params):
        m, s = params
        if m <= 1.01 or s <= 1e-6: return 1e9
        tl2, tt4 = get_theoretical_lmoms(m, s)
        return (tl2 - sample_l2)**2 + (tt4 - sample_tau4)**2

    # Guess: m=4, scale=1
    res = minimize(objective, [4.0, 1.0], bounds=[(1.02, 50), (0.01, 10)], method='L-BFGS-B')
    return res.x[0], res.x[1], res.success

# ===============================================================
# LIKELIHOOD FUNCTIONS (For AIC Calculation)
# ===============================================================
def ll_normal(z):
    return -0.5 * np.sum(z**2 + log(2*pi))

def ll_student_t(z, nu):
    c = gamma((nu + 1) / 2) / (np.sqrt(nu * pi) * gamma(nu / 2))
    return np.sum(np.log(c) - ((nu + 1) / 2) * np.log(1 + z**2 / nu))

def ll_pearson7(z, m, scale):
    # Log-Likelihood used ONLY for AIC score, not for estimation
    c = gamma(m) / (scale * np.sqrt(pi) * gamma(m - 0.5))
    return np.sum(np.log(c) - m * np.log(1 + (z / scale)**2))

def mle_student_t(z):
    # Keep Student-t as MLE benchmark
    def nll(nu):
        if nu <= 2: return 1e20
        return -ll_student_t(z, nu)
    res = minimize(lambda x: nll(x[0]), x0=[8.0], bounds=[(2.01, 200)], method="L-BFGS-B")
    return res.x[0]

# ===============================================================
# MAIN LOOP
# ===============================================================
files = [f for f in os.listdir(RESID_DIR) if f.endswith("_resid.csv")]
rows = []

print(f"\n🚀 Running L-Moment Comparison on {len(files)} tickers...\n")

for f in files:
    tkr = f.replace("_resid.csv", "")
    try:
        z = pd.read_csv(os.path.join(RESID_DIR, f))["std_resid"].dropna().to_numpy()
        n = len(z)
        if n < 50: continue

        # 1. Normal (Benchmark)
        llN = ll_normal(z)
        aicN = 2*0 - 2*llN 
        bicN = 0*log(n) - 2*llN

        # 2. Student-t (Benchmark - MLE)
        nu_t = mle_student_t(z)
        llT = ll_student_t(z, nu_t)
        aicT = 2*1 - 2*llT
        bicT = 1*log(n) - 2*llT

        # 3. Pearson VII (L-MOMENTS)
        m_p7, s_p7, ok_p7 = fit_pearson7_lmom(z)
        
        # Calculate AIC based on L-moment parameters
        if ok_p7:
            llP = ll_pearson7(z, m_p7, s_p7)
            aicP = 2*2 - 2*llP
            bicP = 2*log(n) - 2*llP
        else:
            llP, aicP, bicP = np.nan, np.nan, np.nan

        # Determine Winner
        aics = {"Normal": aicN, "StudentT": aicT, "PearsonVII": aicP}
        best_model = min(aics, key=aics.get)

        rows.append({
            "ticker": tkr, "n": n,
            "Best_Model": best_model,
            "AIC_Normal": aicN, "AIC_StudentT": aicT, "AIC_PearsonVII": aicP,
            "m_lmom": m_p7, "scale_lmom": s_p7, "nu_mle": nu_t
        })
        
        print(f"✅ {tkr}: Best={best_model} | m={m_p7:.2f} (L-Mom)")

    except Exception as e:
        print(f"❌ {tkr}: {e}")

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"\n💾 Saved L-Moment comparison to: {OUT_FILE}")