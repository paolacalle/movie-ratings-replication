
def is_p_drop(statistic_name, stat, p, stat_testing="mean", alpha=0.005):
    print(f"{statistic_name}: {stat:.3f}, p-value: {p:.4f}")
    if p < alpha:
        print(f"Drop H0 --> accept H1")
    else:
        print(f"Fail to drop H0 --> No significant difference among {stat_testing}s.")