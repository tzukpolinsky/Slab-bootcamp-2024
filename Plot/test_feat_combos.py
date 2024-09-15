import numpy as np
import statsmodels.api as sm
from itertools import combinations

def test_feat_combos(data_set, feature_columns, target_column, num_features=2, with_conclusion_print=False):
    for feature_combo in combinations(feature_columns, num_features):
        inp = data_set[list(feature_combo)]
        y = data_set[target_column]
        mdl = sm.OLS(y, sm.add_constant(inp.to_numpy())).fit()
        residuals = np.array(mdl.resid)
        y_pred = mdl.predict(sm.add_constant(inp.to_numpy()))
        is_normal, JB, p_res, skewness, kurtosis = check_normality(residuals,
                                                                     with_conclusion_print=with_conclusion_print)
        plot_normality_test(residuals,feature_combo,target_column ,is_normal, JB, p_res, skewness, kurtosis)
        is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue = check_homoscedasticity(y, y_pred,
                                                                                   with_conclusion_print=with_conclusion_print)
        homo_test_plot(y, y_pred,feature_combo,target_column, is_homoscedastic, lm, lm_pvalue, fvalue, f_pvalue)

        is_linear, lin_p_res, fstat = check_linearity(inp, y, with_conclusion_print=with_conclusion_print)
        plot_linearity_test(inp, y.values,feature_combo, target_column, is_linear, lin_p_res, fstat)

        no_multicollinearity, vif_values = multi_coll_check(inp, with_conclusion_print=with_conclusion_print)
        plot_multicollinearity_test(vif_values, threshold=5.0)

        no_autocorrelation, lb_p_res, dw_statistic = check_autocorrelation(residuals,
                                                                             with_conclusion_print=with_conclusion_print)
        auto_corr_plot(residuals,feature_combo,target_column, no_autocorrelation, lb_p_res, dw_statistic)