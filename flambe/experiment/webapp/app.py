"""Report site using Flambe

The report site is the main control centre for the remote experiment.
It's responsible of showing the current status of the experiment and to
provide the resources that the experiment
is producing (metrics, trained models, etc.)

The website consumes the files that flambe outputs, it doesn't require
extra running resource (like for example, Redis DB).

The website is implemented using Flask.

*Important: no typing is provided for now.*

"""

import os
from flask import Flask, render_template, send_file, jsonify

import shutil

import pickle
import subprocess
import tempfile
from time import sleep

app = Flask(__name__)


def load_state():
    """Get status about the blocks (runned, running, remaining)

    """
    progress_file = app.config.get('progress_file')

    if not os.path.exists(progress_file):
        return None

    with open(progress_file, "rb") as f:
        state = pickle.load(f)

    return state


def analyze_download_params(block, variant):
    output_dir = app.config.get('output_dir')
    output_name = "experiment"

    if block is not None:
        output_dir = os.path.join(output_dir, block)
        output_name = f"{output_name}-{block}"

        if variant is not None:
            vs = [x for x in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, x))]
            for v in vs:
                if v.startswith(variant):
                    output_dir = os.path.join(output_dir, v)
                    output_name = f"{output_name}-{variant}"
                    break

    return output_dir, output_name


@app.route('/download', methods=['GET'])
@app.route('/download/<block>', methods=['GET'])
@app.route('/download/<block>/<variant>', methods=['GET'])
def download(block=None, variant=None):
    """
    Downloads models + logs stored in the filesystem of the Orchestrator
    """
    output_dir, output_name = analyze_download_params(block, variant)

    with tempfile.TemporaryDirectory() as d:
        shutil.make_archive(os.path.join(d, "output"), "tar", output_dir)
        return send_file(os.path.join(d, "output.tar"),
                         as_attachment=True,
                         attachment_filename=f"{output_name}.tar")


@app.route('/download_logs', methods=['GET'])
@app.route('/download_logs/<block>', methods=['GET'])
@app.route('/download_logs/<block>/<variant>', methods=['GET'])
def download_logs(block=None, variant=None):
    """
    Downloads all artifacts but the models stored in the filesystem
    of the Orchestrator machine
    """
    output_dir, output_name = analyze_download_params(block, variant)

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        subprocess.check_call(['rsync', '-av', '--exclude', '*.pt', output_dir, d2])
        shutil.make_archive(os.path.join(d1, "output_no_models"), "tar", d2)
        return send_file(os.path.join(d1, "output_no_models.tar"),
                         as_attachment=True,
                         attachment_filename=f"{output_name}-logs.tar")


@app.route('/stream')
def stream():
    if app.config['output_log'] is None:
        return None

    def generate():
        p = 0
        while True:
            with open(app.config['output_log'], 'r') as f:
                f.seek(p)
                content = f.read()
                p = f.tell()
                # Attention: if this content is used as direct HTML
                # it needs to be escaped. For example, using
                # html.escape(content)
                # Not escaping could have undesired effects, like
                # commenting out the content (if case an '<' is present)
                yield content
                sleep(1)

    return app.response_class(generate(), mimetype='text/event-stream')


@app.route('/state')
def state():
    state = load_state()
    return jsonify(state.toJSON())


@app.route('/console_log')
def console_log():
    """
    View console with the logs
    """
    return render_template('console.html')


@app.route('/')
def index():
    """
    Main endpoint. Works with the `index.html` template.
    """
    state = load_state()
    tensorboard_url = app.config.get('tensorboard_url')
    output_log = app.config.get('output_log')
    output_dir = app.config.get('output_dir')

    return render_template('index.html', state=state, tensorboard_url=tensorboard_url,
                           output_log=output_log, output_dir=output_dir)
