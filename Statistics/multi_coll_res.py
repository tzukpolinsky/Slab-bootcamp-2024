def multi_coll_res(no_multicollinearity: bool, threshold: float):
    """
    Print the conclusion from the multicollinearity test.
    """
    if no_multicollinearity:
        print("Conclusion: No multicollinearity detected.")
    else:
        print("Conclusion: Multicollinearity detected.")
        print(f"  Features with VIF > {threshold:.1f} may be problematic.")