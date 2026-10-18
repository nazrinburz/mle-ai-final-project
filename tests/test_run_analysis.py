import numpy as np
import pandas as pd

from src.run_all import analyze_saved_results


def test_analysis_high_price_band(tmp_path):
    pd.DataFrame([dict(Model="Tree regression",Task="A",RMSE=1,MAE=1,R2=0)]).to_csv(
        tmp_path/"metrics.csv",index=False
    )
    pd.DataFrame(dict(true=[500000,500001,600000],absolute_error=[999,10,30])).to_csv(
        tmp_path/"errors_Tree_regression.csv",index=False
    )
    pd.DataFrame(dict(true=[100],absolute_error=[0])).to_csv(
        tmp_path/"errors_Forest.csv",index=False
    )
    for name in ("Tree_classification","Tree_regression","Forest"):
        pd.DataFrame(dict(feature=["area"],importance=[1])).to_csv(
            tmp_path/f"importance_{name}.csv",index=False
        )
    pd.DataFrame(dict(feature=["area"],weight=[1])).to_csv(
        tmp_path/"svm_standardized_weights.csv",index=False
    )
    pd.DataFrame(columns=["model","validation_score"]).to_csv(
        tmp_path/"validation_search.csv",index=False
    )
    analyze_saved_results(tmp_path,tmp_path/"figures")
    summary=pd.read_csv(tmp_path/"high_price_error_summary.csv").set_index("model")
    assert summary.loc["Tree regression","count"]==2
    assert summary.loc["Tree regression","MAE"]==20
    assert summary.loc["Forest","count"]==0
    assert np.isnan(summary.loc["Forest","MAE"])
