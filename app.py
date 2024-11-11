from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t

app = Flask(__name__)
app.secret_key = "Hola"  # Replace with your own secret key, needed for session management


def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate data and initial plots

    # Generate a random dataset X of size N with values between 0 and 1
    X = np.random.uniform(0, 1, N)

    # Generate a random dataset Y using the specified beta0, beta1, mu, and sigma2
    # Y = beta0 + beta1 * X + mu + error term
    sigma = np.sqrt(sigma2)
    error = np.random.normal(0, sigma, N)
    Y = beta0 + beta1 * X + mu + error

    # Fit a linear regression model to X and Y
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate a scatter plot of (X, Y) with the fitted regression line
    plot1_path = "static/plot1.png"
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    # Regression line
    X_line = np.linspace(0, 1, 100)
    Y_line = model.predict(X_line.reshape(-1, 1))
    plt.plot(X_line, Y_line, color='red', label='Regression Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Scatter Plot with Regression Line')
    plt.savefig(plot1_path)
    plt.close()

    # Run S simulations to generate slopes and intercepts
    # We won't store the slopes and intercepts here anymore
    # Instead, we'll generate them in the hypothesis test and confidence interval functions

    # Return data needed for further analysis
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Generate data and initial plots
        X, Y, slope, intercept, plot1 = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store only necessary parameters in session
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S
        session["slope"] = slope
        session["intercept"] = intercept

        # Generate and save histograms for the initial data generation (if needed)
        # We'll omit this step here and move histogram generation to the hypothesis test route

        # Return render_template with variables
        return render_template(
            "index.html",
            plot1=plot1,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    # This route handles data generation (same as above)
    return index()


@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve parameters from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    observed_slope = float(session.get("slope"))
    observed_intercept = float(session.get("intercept"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Re-run simulations to generate slopes and intercepts
    sigma = np.sqrt(sigma2)
    simulated_stats = []

    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(0, sigma, N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)

        if parameter == "slope":
            sim_stat = sim_model.coef_[0]
        else:
            sim_stat = sim_model.intercept_

        simulated_stats.append(sim_stat)

    simulated_stats = np.array(simulated_stats)

    # Use the observed statistic
    if parameter == "slope":
        observed_stat = observed_slope
        hypothesized_value = beta1
    else:
        observed_stat = observed_intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type
    if test_type == '>':
        p_value = np.sum(simulated_stats >= observed_stat) / S
    elif test_type == '<':
        p_value = np.sum(simulated_stats <= observed_stat) / S
    elif test_type == '!=':
        diff_observed = abs(observed_stat - hypothesized_value)
        diffs_simulated = abs(simulated_stats - hypothesized_value)
        p_value = np.sum(diffs_simulated >= diff_observed) / S
    else:
        p_value = None  # Invalid test type

    # Display a fun message if the p-value is very small
    fun_message = None
    if p_value is not None and p_value <= 0.0001:
        fun_message = "Wow! You've encountered a rare event!"

    # Plot histogram of simulated statistics
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(observed_stat, color='red', linestyle='dashed', linewidth=2, label='Observed Statistic')
    plt.axvline(hypothesized_value, color='green', linestyle='dashed', linewidth=2, label='Hypothesized Value')
    plt.xlabel(parameter.capitalize())
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Simulated {parameter.capitalize()}s')
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )


@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve parameters from session
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    observed_slope = float(session.get("slope"))
    observed_intercept = float(session.get("intercept"))

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    # Re-run simulations to generate estimates
    sigma = np.sqrt(sigma2)
    estimates = []

    for _ in range(S):
        X_sim = np.random.uniform(0, 1, N)
        error_sim = np.random.normal(0, sigma, N)
        Y_sim = beta0 + beta1 * X_sim + mu + error_sim

        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)

        if parameter == "slope":
            estimate = sim_model.coef_[0]
            true_param = beta1
        else:
            estimate = sim_model.intercept_
            true_param = beta0

        estimates.append(estimate)

    estimates = np.array(estimates)

    # Calculate mean and standard deviation of the estimates
    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)

    # Calculate confidence interval
    lower_percentile = (100 - confidence_level) / 2
    upper_percentile = 100 - lower_percentile
    ci_lower = np.percentile(estimates, lower_percentile)
    ci_upper = np.percentile(estimates, upper_percentile)

    # Check if confidence interval includes true parameter
    includes_true = (ci_lower <= true_param) and (true_param <= ci_upper)

    # Plot the individual estimates and confidence interval
    plot4_path = "static/plot4.png"
    plt.figure(figsize=(10, 2))
    plt.scatter(estimates, np.zeros_like(estimates), color='gray', alpha=0.5, label='Simulated Estimates')
    # Mean estimate
    if includes_true:
        color = 'green'
    else:
        color = 'red'
    plt.scatter(mean_estimate, 0, color=color, s=100, label='Mean Estimate')
    # Confidence interval
    plt.hlines(0, ci_lower, ci_upper, colors='blue', linestyles='-', linewidth=4, label='Confidence Interval')
    # True parameter value
    plt.scatter(true_param, 0, color='black', marker='x', s=100, label='True Parameter')

    plt.xlabel(parameter.capitalize())
    plt.yticks([])
    plt.title(f'{confidence_level}% Confidence Interval for {parameter.capitalize()}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot4_path)
    plt.close()

    # Return results to template
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=int(confidence_level),
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_slope if parameter == "slope" else observed_intercept,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )


if __name__ == "__main__":
    app.run(debug=True)
