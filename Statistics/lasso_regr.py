def lasso_regr(data_tbl: pd.DataFrame, in_features: [str], out_features: [str]):
    input = data_tbl[in_features]
    output = data_tbl[out_features]
    model = MultiTaskLassoCV().fit(input, output)
    return model
