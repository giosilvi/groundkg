# GroundKG Web Dashboard

A minimal web frontend for running and monitoring the GroundKG pipeline.

## Features

- **Run Pipeline Commands**: Execute any Makefile.gk target from the web interface
- **View Output Files**: Browse generated files (NER, candidates, edges, etc.)
- **Quality Metrics**: View quality reports and statistics
- **Pipeline Status**: Monitor current pipeline stage and file status

## Setup

1. Install Flask (if not already installed):
```bash
pip install flask
```

2. Run the web server:
```bash
cd web
python app.py
```

3. Open your browser:
```
http://localhost:5000
```

## Usage

### Running Commands

Click any button in the "Pipeline Commands" section to run that Makefile target. The output will appear in the output panel below.

### Viewing Files

Click "Refresh File List" to see all output files with their sizes and modification dates. Click "View" next to any file to see its contents (first 100 lines).

### Quality Report

Click "Refresh Quality Report" to see the latest quality metrics including:
- Scored predictions per class
- Edges emitted
- Training label distribution
- Thresholds

## API Endpoints

- `GET /` - Main dashboard page
- `POST /api/run/<target>` - Run a make target
- `GET /api/pipeline/status` - Get pipeline status
- `GET /api/files` - List output files
- `GET /api/file/<key>` - Get file contents
- `GET /api/quality` - Get quality report

## Notes

- Commands run synchronously (you'll see output in real-time)
- The dashboard auto-refreshes status every 10 seconds
- File viewing is limited to first 100 lines for performance

