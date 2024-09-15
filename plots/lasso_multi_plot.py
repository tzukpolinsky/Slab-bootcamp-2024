import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV


def lasso_multi_plot(data_tbl: pd.DataFrame, in_features: [str], out_features: [str], sns=None):
    input = data_tbl[in_features]
    output = data_tbl[out_features]
    model = MultiTaskLassoCV().fit(input, output)
    output_pred = model.predict(input)
    data = pd.DataFrame(input, columns=in_features)
    data[out_features] = output
    data[[f'{col}_pred' for col in out_features]] = output_pred

    sns.pairplot(data)
    import matplotlib.pyplot as plt
    plt.suptitle('Pairplot of Predictors and Responses')
    import matplotlib.pyplot as plt
    plt.show()
