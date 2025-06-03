import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_csv(filepath):
    df = pd.read_csv(filepath, sep="\t")
    df = df.drop(index=0).reset_index(drop=True)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Set Point"] = pd.to_numeric(df["Set Point"])
    df["Feedback"] = pd.to_numeric(df["Feedback"])
    return df

def detect_step(df, threshold=5):
    sp_diff = df["Set Point"].diff().abs()
    step_idx = sp_diff[sp_diff > threshold].index
    if not step_idx.empty:
        i = step_idx[0]
        return i, df.iloc[i]["Set Point"]
    return None, None

def first_order_model(t, K, tau, delay):
    t_shifted = np.maximum(0, t - delay)
    return K * (1 - np.exp(-t_shifted / tau))

def simulate_pid_response(t, setpoint, Kp, Ki, plant_K, plant_tau, plant_delay):
    dt = t[1] - t[0]
    output = []
    feedback = 0
    integral = 0

    for i in range(len(t)):
        error = setpoint - feedback
        integral += error * dt
        control_signal = Kp * error + Ki * integral

        time_shift = max(0, t[i] - plant_delay)
        y = plant_K * (1 - np.exp(-time_shift / plant_tau))
        feedback = y * control_signal
        output.append(feedback)

    return np.array(output)

def analyze_and_plot(filepath):
    df = load_csv(filepath)
    i, new_sp = detect_step(df)
    if i is None:
        print("No step change detected.")
        return

    df = df.iloc[i:].copy().reset_index(drop=True)
    df["time_s"] = (df["Timestamp"] - df["Timestamp"].iloc[0]).dt.total_seconds()
    t = df["time_s"].values
    y = df["Feedback"].values
    u = new_sp - df["Feedback"].iloc[0]

    y_norm = (y - y[0]) / u
    try:
        popt, _ = curve_fit(first_order_model, t, y_norm, bounds=([0.1, 1, 0], [2.0, 1000, 300]))
    except RuntimeError:
        print("Model fitting failed.")
        return

    K_fit, tau_fit, delay_fit = popt
    print(f"Estimated system gain K: {K_fit:.2f}")
    print(f"Estimated time constant tau: {tau_fit:.1f} s")
    print(f"Estimated delay: {delay_fit:.1f} s")

    desired_tc = tau_fit
    Kp = (tau_fit / (K_fit * desired_tc))
    Ki = 1 / desired_tc

    print(f"Suggested PID values (IMC-based):")
    print(f"  Kp = {Kp:.4f}")
    print(f"  Ki = {Ki:.6f}")

    try:
        kp_user = input("Enter your Kp (or press Enter to use suggested): ")
        ki_user = input("Enter your Ki (or press Enter to use suggested): ")
        Kp = float(kp_user) if kp_user.strip() else Kp
        Ki = float(ki_user) if ki_user.strip() else Ki
    except ValueError:
        print("Invalid input, using suggested values.")

    y_sim_model = first_order_model(t, K_fit, tau_fit, delay_fit) * u + y[0]
    y_sim_pid = simulate_pid_response(t, new_sp, Kp, Ki, K_fit, tau_fit, delay_fit)

    plt.figure(figsize=(12, 6))
    plt.plot(t, y, label="Measured Feedback")
    plt.plot(t, df["Set Point"], label="Set Point", linestyle="--")
    plt.plot(t, y_sim_model, label="Fitted First-Order Model", linestyle=":")
    plt.plot(t, y_sim_pid, label=f"Simulated PID (Kp={Kp:.3f}, Ki={Ki:.6f})", linestyle="-.")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title("PID Step Response Analysis & Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pid_tuner_v2.py <your_csv_file>")
    else:
        analyze_and_plot(sys.argv[1])
