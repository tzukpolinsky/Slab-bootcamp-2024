def norm_test_res(is_normal: bool, JB: float, p_res: float, skewness: float, kurtosis: float,
                               alpha: float, skewness_threshold: float, kurtosis_limits: Tuple[float, float]):
    """
    Print the conclusion from the normality test.
    """
    if is_normal:
        print("Conclusion: The residuals appear to be normally distributed.")
    else:
        print("Conclusion: The residuals do not appear to be normally distributed.")
        if p_res <= alpha:
            print(f"  - The Jarque-Bera test indicates non-normality (p-value <= {alpha}).")
        if abs(skewness) >= skewness_threshold:
            print(f"  - The distribution is skewed (|skewness| >= {skewness_threshold}).")
        if kurtosis <= kurtosis_limits[0] or kurtosis >= kurtosis_limits[1]:
            print(f"  - The distribution has abnormal kurtosis (outside range {kurtosis_limits}).")