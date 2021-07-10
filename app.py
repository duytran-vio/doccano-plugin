import os
import logging
from flask import Flask, jsonify, render_template, request as flask_request, send_file

from services.client import refresh_client, Client
from services.sample import handle_request as handle_sample_request
from services.evaluate import handle_request as handle_evaluate_request
from services.download import handle_request as handle_download_request
from services.summary import handle_request as handle_summary_request
from services.create import handle_request as handle_create_request
from services.add import handle_request as handle_add_request
from services.retrain import handle_request as handle_retrain_request

from services.address.prepare import preparation



HOST = '0.0.0.0'
PORT = 5500
debug = False
app = Flask(__name__,
    template_folder='template',
    static_url_path='/static',
    static_folder='static')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'doccano_project_data')
DATA_SAMPLE_PATH = os.path.join(DATA_PATH, 'sample')
DATA_DOWNLOAD_PATH = os.path.join(DATA_PATH, 'download')
DATA_CREATE_PATH = os.path.join(DATA_PATH, 'create')

# enc, dec, id2w, disown = prepare_enc(), prepare_dec(), prepare_id2w(), prepare_disown()
# ward_dis = prepare_ward_s_district(disown[2])
#
# street_enc, vill_enc, ward_enc, district_enc, province_enc = enc
# street_dec, vill_dec, ward_dec, district_dec, province_dec = dec
# street_id2w, vill_id2w, ward_id2w, district_id2w, province_id2w = id2w
# dis_street, dis_vill, dis_ward, dis_province = disown
address_inp = preparation(folder='services/address/tmt_address')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')


@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/create')
def create():
    return render_template('create.html')

@app.route('/summary')
def summary():
    return render_template('summary.html')

@app.route('/add')
def add():
    return render_template('add.html')

@app.route('/retrain')
def retrain():
    return render_template('retrain.html')

@app.route('/api/sample', methods=['POST'])
def create_sample_test_project():
    try:
        print('Create sample test project')
        refresh_client()
        new_project_id = handle_sample_request(flask_request, Client.doccano_client)
        print(f'Project ID: {new_project_id}')
        return jsonify({
            'status': 'OK',
            'link': f'http://103.113.81.36:8000/projects/{new_project_id}',
        }), 200
    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['GET'])
def evaluate_test_project():
    try:
        print('Evaluate test project')
        refresh_client()
        file_name = handle_evaluate_request(flask_request, Client.doccano_client)
        file_path = os.path.join(DATA_DOWNLOAD_PATH, file_name)
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/download', methods=['GET'])
def download_test_project():
    try:
        print('Download test project')
        refresh_client()
        file_name = handle_download_request(flask_request, Client.doccano_client)
        file_path = os.path.join(DATA_DOWNLOAD_PATH, file_name)
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary', methods=['GET'])
def download_summary_project():
    try:
        print('Download summary project')
        refresh_client()
        summary_name = handle_summary_request(flask_request, Client.doccano_client)
        summary_path = os.path.join(DATA_DOWNLOAD_PATH, summary_name)
        return send_file(summary_path, as_attachment=True)
    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/create', methods=['POST'])
def create_create_project():
    try:
        print('Create project')
        refresh_client()
        new_project_id = handle_create_request(flask_request, Client.doccano_client,address_inp)
        print(f'Project ID: {new_project_id}')
        return jsonify({
            'status': 'OK',
            'link': f'http://103.113.81.36:8000/projects/{new_project_id}',
        }), 200
    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/add', methods=['POST'])
def create_add_project():
    try:
        print('Add to project')
        refresh_client()
        # address_inp = (enc, dec, id2w, disown, ward_dis)
        project_id = handle_add_request(flask_request, Client.doccano_client, address_inp)
        print(f'Project ID: {project_id}')
        return jsonify({
            'status': 'OK',
            'link': f'http://103.113.81.36:8000/projects/{project_id}',
        }), 200
    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['GET'])
def retrain_test_project():
    try:
        print('Retrain project')
        refresh_client()
        handle_retrain_request(flask_request, Client.doccano_client)
        return jsonify({
            'status': 'OK'
        }), 200
    except Exception as e:
        logging.exception(e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=debug)
