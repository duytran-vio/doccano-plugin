# Retrain reproduce model
1. Copy neccessary files from tdsvm folder (which is in another source)
    # These files should be brought to services/retrain
    - X_test.csv
    - X_train.csv
    - y_test.pkl
    - y_train.pkl
    - report_dis.pkl
    - report.pkl
    # These files should be brought to services/models
    - hungne
    - tfidf.pickle

2. Run init_retrain.py to initialize:
    # These files are in services/retrain
    - f1_history.pkl
    - f1_logs.pkl
    - new_data_test.csv
    - new_label_test.pkl
    - accum_data.csv
    - accum_label.csv

3. Change status of lines (to retrain) in services/retrain/retrain_project.csv 