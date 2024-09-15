def linear_test_res(is_linear: bool, alpha: float):
    """
    Print the conclusion from the linearity test.
    """
    if is_linear:
        print("Conclusion: The relationship appears to be linear.")
    else:
        print(f"Conclusion: The relationship may not be linear (p-value <= {alpha}).")