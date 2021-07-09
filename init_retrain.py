import pickle
import datetime as dt

def write_text_to_file(file_path, data):
    f = open(file_path, 'w',encoding='utf-8')
    for i in data:
        f.write(str(i))
        f.write('\n')
    f.close()

RETRAIN_DIR = 'services/retrain'

### init f1 files
report = pickle.load(open(RETRAIN_DIR + '/report.pkl','rb'))
report_1 = {'macro avg': {'f1-score':0}, 'weighted avg': {'f1-score': 0}}
pickle.dump([(report, report_1)], open(RETRAIN_DIR + '/f1_history.pkl','wb'))

report_disp = pickle.load(open(RETRAIN_DIR+ '/report_dis.pkl','rb'))
pickle.dump([(report_disp, None, str(dt.datetime.now()) )], open(RETRAIN_DIR + '/f1_logs.pkl','wb'))

### init new test files
write_text_to_file(RETRAIN_DIR + '/new_data_test.txt', [])
pickle.dump([], open(RETRAIN_DIR + '/new_label_test.pkl', 'wb'))

### init accumulate files
write_text_to_file(RETRAIN_DIR + '/accum_data.txt', [])
write_text_to_file(RETRAIN_DIR + '/accum_label.txt', [])
