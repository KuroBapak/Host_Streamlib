import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

st.title("Normal Distribution Visualization Using Teacher's Values")

# Parameters based on teacher's data:
mean = 6.0         # Mean study hours per day
std_dev = 0.75     # Standard deviation in hours (45 minutes)
# Part (a)
threshold_a = 7.0  # x = 7 hours for part (a)
# Part (b)
threshold_b = 6.5  # x = 6.5 hours for part (b)
# Part (c)
lower_bound = 5.0  # Lower bound for part (c)
upper_bound = 7.0  # Upper bound for part (c)
# Part (d)
perc = 0.30        # 30% of students study more than the threshold for part (d)

# Calculate probabilities:
# a) P(X > 7)
p_a = 1 - stats.norm.cdf(threshold_a, mean, std_dev)  # should be approximately 0.0918 (9.18%)
# b) P(X < 6.5)
p_b = stats.norm.cdf(threshold_b, mean, std_dev)        # should be approximately 0.7486 (74.86%)
# c) P(5 < X < 7)
p_c = stats.norm.cdf(upper_bound, mean, std_dev) - stats.norm.cdf(lower_bound, mean, std_dev)  # ≈ 0.8164 (81.64%)
# d) Find threshold such that 30% study more than that value, i.e., P(X > t) = 0.30
# This is equivalent to P(X < t) = 0.70
t_value = stats.norm.ppf(0.70, mean, std_dev)           # ≈ 6.39 hours

st.markdown("### Calculated Results")
st.write(f"**a)** Percentage of students studying more than {threshold_a} hours: **{p_a*100:.2f}%**")
st.write(f"**b)** Percentage of students studying less than {threshold_b} hours: **{p_b*100:.2f}%**")
st.write(f"**c)** Percentage of students studying between {lower_bound} and {upper_bound} hours: **{p_c*100:.2f}%**")
st.write(f"**d)** 30% of students study more than **{t_value:.2f}** hours.")

# Generate x values for plotting the normal distribution
x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)
y = stats.norm.pdf(x, mean, std_dev)

# Plot for part (a): Shade area for x > 7 hours
fig_a, ax_a = plt.subplots(figsize=(8, 4))
ax_a.plot(x, y, color="blue", label="Normal Distribution")
x_fill_a = np.linspace(threshold_a, mean + 4*std_dev, 1000)
ax_a.fill_between(x_fill_a, stats.norm.pdf(x_fill_a, mean, std_dev), color="red", alpha=0.5,
                  label=f"Area: x > {threshold_a} (≈{p_a*100:.2f}%)")
ax_a.axvline(threshold_a, color="red", linestyle="--")
ax_a.set_title("Part (a): x > 7 hours")
ax_a.set_xlabel("Study Hours")
ax_a.set_ylabel("Probability Density")
ax_a.legend()
st.pyplot(fig_a)

# Plot for part (b): Shade area for x < 6.5 hours
fig_b, ax_b = plt.subplots(figsize=(8, 4))
ax_b.plot(x, y, color="blue", label="Normal Distribution")
x_fill_b = np.linspace(mean - 4*std_dev, threshold_b, 1000)
ax_b.fill_between(x_fill_b, stats.norm.pdf(x_fill_b, mean, std_dev), color="green", alpha=0.5,
                  label=f"Area: x < {threshold_b} (≈{p_b*100:.2f}%)")
ax_b.axvline(threshold_b, color="green", linestyle="--")
ax_b.set_title("Part (b): x < 6.5 hours")
ax_b.set_xlabel("Study Hours")
ax_b.set_ylabel("Probability Density")
ax_b.legend()
st.pyplot(fig_b)

# Plot for part (c): Shade area for 5 < x < 7 hours
fig_c, ax_c = plt.subplots(figsize=(8, 4))
ax_c.plot(x, y, color="blue", label="Normal Distribution")
x_fill_c = np.linspace(lower_bound, upper_bound, 1000)
ax_c.fill_between(x_fill_c, stats.norm.pdf(x_fill_c, mean, std_dev), color="purple", alpha=0.5,
                  label=f"Area: {lower_bound} < x < {upper_bound} (≈{p_c*100:.2f}%)")
ax_c.axvline(lower_bound, color="purple", linestyle="--")
ax_c.axvline(upper_bound, color="purple", linestyle="--")
ax_c.set_title("Part (c): 5 < x < 7 hours")
ax_c.set_xlabel("Study Hours")
ax_c.set_ylabel("Probability Density")
ax_c.legend()
st.pyplot(fig_c)

# Plot for part (d): Shade area for x > threshold (≈6.39 hours)
fig_d, ax_d = plt.subplots(figsize=(8, 4))
ax_d.plot(x, y, color="blue", label="Normal Distribution")
x_fill_d = np.linspace(t_value, mean + 4*std_dev, 1000)
ax_d.fill_between(x_fill_d, stats.norm.pdf(x_fill_d, mean, std_dev), color="orange", alpha=0.5,
                  label=f"Area: x > {t_value:.2f} (30% of students)")
ax_d.axvline(t_value, color="orange", linestyle="--")
ax_d.set_title("Part (d): 30% of students study more than this value")
ax_d.set_xlabel("Study Hours")
ax_d.set_ylabel("Probability Density")
ax_d.legend()
st.pyplot(fig_d)
