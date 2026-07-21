import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm_paths(
    S0,
    mu,
    sigma,
    T,
    n_steps,
    n_paths,
    seed = None):

    # input validation

    if not isinstance(n_steps, int):
        raise TypeError("n_steps must be an integer.")
    if not isinstance(n_paths, int):
        raise TypeError("n_paths must be an integer.")
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if sigma < 0:
        raise ValueError("sigma cannot be negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive.")
    
    # actual code

    dt = T / n_steps

    rng = np.random.default_rng(seed)

    z = rng.standard_normal((n_steps, n_paths))

    prices = np.empty((n_steps + 1, n_paths))
    prices[0, :] = S0

    for t in range(n_steps):
        prices[t + 1, :] = prices[t, :] * np.exp((mu - (0.5 * sigma**2)) * dt + (sigma * np.sqrt(dt) * z[t, :]))

    time_axis = np.linspace(0, T, n_steps + 1)

    return time_axis, prices


if __name__ == "__main__":
    time_axis, prices = simulate_gbm_paths(S0=100, mu=0.08, sigma=0.20, T=1, n_steps=252, n_paths=10000, seed=67)

    plt.plot(time_axis, prices[:, :20])
    plt.xlabel("Time")
    plt.ylabel("Stock price")
    plt.title("First 20 of 10,000 GBM Monte Carlo Paths")
    plt.tight_layout()
    plt.savefig("gbm_paths.png")

    print(prices.shape)

    # tests

    simulated_mean = prices[-1, :].mean()
    theoretical_mean = 100 * np.exp(0.08 * 1)
    absolute_error = abs(simulated_mean - theoretical_mean)
    relative_error = absolute_error / theoretical_mean

    print("Simulated mean terminal price:", simulated_mean)
    print("Theoretical mean terminal price:", theoretical_mean)
    print("Absolute error:", absolute_error)
    print("Relative error:", relative_error)