#!/usr/bin/env python3
"""
Minimal web frontend for GroundKG pipeline
"""
import os
import json
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Store running commands
running_commands = {}
command_outputs = {}

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def run_make_command(target, background=False):
    """Run a make command and return output"""
    cmd = ['make', '-f', 'Makefile.gk', target]
    try:
        if background:
            # Run in background, return immediately
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=PROJECT_ROOT,
                bufsize=1
            )
            running_commands[target] = process
            return {'status': 'started', 'pid': process.pid}
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=600  # 10 minute timeout
            )
            return {
                'status': 'completed' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'error': 'Command timed out'}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def get_file_info(path):
    """Get file information"""
    full_path = PROJECT_ROOT / path
    if not full_path.exists():
        return None
    stat = full_path.stat()
    return {
        'exists': True,
        'size': stat.st_size,
        'modified': stat.st_mtime,
        'lines': sum(1 for _ in open(full_path, 'rb')) if full_path.is_file() else 0
    }


def get_quality_report():
    """Get quality report data"""
    try:
        result = subprocess.run(
            ['python', 'tools/quality_report.py',
             'out/pack.scored.jsonl',
             'out/edges.dedup.jsonl',
             'training/re_train.jsonl',
             'models/thresholds.json'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30
        )
        return {
            'success': result.returncode == 0,
            'output': result.stdout,
            'error': result.stderr if result.returncode != 0 else None
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/run/<target>', methods=['POST'])
def run_command(target):
    """Run a make target"""
    background = request.json.get('background', False) if request.is_json else False
    result = run_make_command(target, background=background)
    return jsonify(result)


@app.route('/api/status/<target>')
def command_status(target):
    """Get status of a running command"""
    if target in running_commands:
        process = running_commands[target]
        if process.poll() is None:
            # Still running
            return jsonify({'status': 'running', 'pid': process.pid})
        else:
            # Finished
            stdout, stderr = process.communicate()
            del running_commands[target]
            return jsonify({
                'status': 'completed' if process.returncode == 0 else 'failed',
                'returncode': process.returncode,
                'stdout': stdout,
                'stderr': stderr
            })
    return jsonify({'status': 'not_found'})


@app.route('/api/files')
def list_files():
    """List important output files"""
    files = {
        'ner': 'out/pack.ner.jsonl',
        'candidates': 'out/pack.candidates.jsonl',
        'scored': 'out/pack.scored.jsonl',
        'edges': 'out/edges.dedup.jsonl',
        'graph': 'out/graph.ttl',
        'seed': 'training/seed.jsonl',
        'train': 'training/re_train.jsonl',
        'dev': 'training/re_dev.jsonl',
        'model': 'models/promoter_v1.onnx',
        'thresholds': 'models/thresholds.json',
        'classes': 'models/classes.json',
    }
    result = {}
    for key, path in files.items():
        info = get_file_info(path)
        result[key] = {
            'path': path,
            'info': info
        }
    return jsonify(result)


@app.route('/api/file/<file_key>')
def get_file(file_key):
    """Get file contents (first N lines)"""
    files = {
        'ner': 'out/pack.ner.jsonl',
        'candidates': 'out/pack.candidates.jsonl',
        'scored': 'out/pack.scored.jsonl',
        'edges': 'out/edges.dedup.jsonl',
        'graph': 'out/graph.ttl',
        'seed': 'training/seed.jsonl',
        'train': 'training/re_train.jsonl',
        'dev': 'training/re_dev.jsonl',
        'thresholds': 'models/thresholds.json',
        'classes': 'models/classes.json',
    }
    
    if file_key not in files:
        return jsonify({'error': 'File not found'}), 404
    
    path = PROJECT_ROOT / files[file_key]
    if not path.exists():
        return jsonify({'error': 'File does not exist'}), 404
    
    lines = request.args.get('lines', 50, type=int)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = ''.join(f.readlines()[:lines])
        return jsonify({
            'path': files[file_key],
            'content': content,
            'total_lines': sum(1 for _ in open(path, 'rb'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quality')
def quality():
    """Get quality report"""
    return jsonify(get_quality_report())


@app.route('/api/pipeline/status')
def pipeline_status():
    """Get overall pipeline status"""
    files = {
        'model': 'models/promoter_v1.onnx',
        'seed': 'training/seed.jsonl',
        'ner': 'out/pack.ner.jsonl',
        'candidates': 'out/pack.candidates.jsonl',
        'scored': 'out/pack.scored.jsonl',
        'edges': 'out/edges.dedup.jsonl',
    }
    
    status = {}
    for key, path in files.items():
        info = get_file_info(path)
        status[key] = info is not None
    
    # Determine pipeline stage
    stage = 'not_started'
    if status['model']:
        stage = 'ready'
    elif status['seed']:
        stage = 'seeded'
    if status['edges']:
        stage = 'complete'
    elif status['scored']:
        stage = 'scored'
    elif status['candidates']:
        stage = 'candidates'
    elif status['ner']:
        stage = 'ner'
    
    return jsonify({
        'stage': stage,
        'files': status
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5100)

