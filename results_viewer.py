"""
Evaluation results viewer - converts JSON results to navigable HTML.
"""

import json
import html
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from fnmatch import fnmatch
from datetime import datetime

from prefill_evals.parser import load_scenario_from_dir, render_transcript


def load_and_filter_results(
    results_path: Path,
    scenario_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    include_errors: bool = False,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Load results from JSON and apply filters."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract results list (handle both old and new formats)
    if isinstance(data, list):
        results = data
        metadata = None
    else:
        results = data.get('results', [])
        metadata = data.get('metadata', {})
    
    filtered = []
    for result in results:
        # Skip errors unless requested
        if 'error' in result and not include_errors:
            continue
        
        # Apply scenario filter
        if scenario_filter and not fnmatch(result.get('scenario_path', ''), scenario_filter):
            continue
        
        # Apply model filter
        model_id = result.get('model_id', '')
        if model_filter and not fnmatch(model_id, model_filter):
            continue
        
        # Apply score filters
        if min_score is not None or max_score is not None:
            # Get average score across all grades
            scores = []
            for grade_list in result.get('grades', []):
                for grade in grade_list:
                    if grade and 'score' in grade:
                        scores.append(grade['score'])
            
            if scores:
                avg_score = sum(scores) / len(scores)
                if min_score is not None and avg_score < min_score:
                    continue
                if max_score is not None and avg_score > max_score:
                    continue
        
        filtered.append(result)
    
    # Apply limit per scenario if specified
    if limit:
        scenario_counts = defaultdict(int)
        limited = []
        for result in filtered:
            scenario = result.get('scenario_path', '')
            if scenario_counts[scenario] < limit:
                limited.append(result)
                scenario_counts[scenario] += 1
        filtered = limited
    
    return {'results': filtered, 'metadata': metadata}


def group_by_scenario(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by scenario path."""
    grouped = defaultdict(list)
    for result in results:
        scenario = result.get('scenario_path', 'unknown')
        grouped[scenario].append(result)
    return dict(grouped)


def load_scenario_transcript(scenario_path: str) -> Optional[str]:
    """Load the transcript for a scenario."""
    try:
        path = Path(scenario_path)
        if not path.is_absolute():
            # Try common base paths
            for base in [Path.cwd(), Path.cwd() / 'evals']:
                full_path = base / path
                if full_path.exists():
                    path = full_path
                    break
        
        if path.exists() and path.is_dir():
            transcript_file = path / 'transcript.txt'
            if transcript_file.exists():
                return transcript_file.read_text()
    except Exception as e:
        print(f"Error loading transcript for {scenario_path}: {e}")
    return None


def format_grade(grade: Optional[Dict]) -> str:
    """Format a grade result for display."""
    if not grade:
        return '<span class="no-grade">No grade</span>'
    
    score = grade.get('score', 'N/A')
    data = grade.get('data', {})
    
    parts = [f'<span class="score">Score: {score}</span>']
    
    # Add additional data if present
    if data:
        if 'feedback' in data:
            parts.append(f'<div class="feedback">Feedback: {html.escape(data["feedback"])}</div>')
        if 'reasoning' in data:
            parts.append(f'<div class="reasoning">Reasoning: {html.escape(data["reasoning"])}</div>')
        if 'matched' in data:
            parts.append(f'<div class="matched">Matched: {data["matched"]}</div>')
    
    return '\n'.join(parts)


def format_error(error: Dict) -> str:
    """Format an error result for display."""
    error_type = error.get('type', 'Unknown')
    message = error.get('message', 'No message')
    
    html_str = f'<div class="error">'
    html_str += f'<strong>Error Type:</strong> {html.escape(error_type)}<br>'
    html_str += f'<strong>Message:</strong> {html.escape(message)}'
    
    if 'traceback' in error:
        html_str += f'<details><summary>Stack trace</summary><pre>{html.escape(error["traceback"])}</pre></details>'
    
    html_str += '</div>'
    return html_str


def render_html(grouped_results: Dict[str, List[Dict]], metadata: Optional[Dict] = None) -> str:
    """Render the results as HTML."""
    html_parts = ['''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Results Viewer</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        
        .container {
            display: flex;
            min-height: 100vh;
        }
        
        .sidebar {
            width: 250px;
            background: white;
            border-right: 1px solid #ddd;
            padding: 20px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
        }
        
        .content {
            margin-left: 250px;
            flex: 1;
            padding: 20px;
        }
        
        .sidebar h3 {
            margin-top: 0;
            color: #333;
        }
        
        .sidebar ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        
        .sidebar li {
            margin: 5px 0;
        }
        
        .sidebar a {
            color: #0066cc;
            text-decoration: none;
            display: block;
            padding: 5px;
            border-radius: 3px;
            font-size: 14px;
            word-break: break-all;
        }
        
        .sidebar a:hover {
            background: #f0f0f0;
        }
        
        .scenario-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .scenario-section h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        
        .transcript {
            background: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .transcript h3 {
            margin-top: 0;
            color: #555;
        }
        
        .transcript pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .responses-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .responses-table th {
            background: #f0f0f0;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #ddd;
            font-weight: 600;
        }
        
        .responses-table td {
            padding: 10px;
            border-bottom: 1px solid #eee;
            vertical-align: top;
        }
        
        .responses-table tr:hover {
            background: #f9f9f9;
        }
        
        .model-name {
            font-weight: 500;
            color: #0066cc;
            word-break: break-all;
        }
        
        .response-content {
            max-height: 300px;
            overflow-y: auto;
            background: #f8f8f8;
            padding: 10px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .grade-cell {
            min-width: 150px;
        }
        
        .score {
            font-weight: 600;
            color: #28a745;
        }
        
        .no-grade {
            color: #999;
            font-style: italic;
        }
        
        .error {
            background: #fee;
            border: 1px solid #fcc;
            padding: 10px;
            border-radius: 3px;
            color: #c00;
        }
        
        .feedback, .reasoning {
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }
        
        details {
            margin-top: 10px;
        }
        
        summary {
            cursor: pointer;
            color: #0066cc;
        }
        
        .metadata {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metadata h3 {
            margin-top: 0;
            color: #333;
        }
        
        .metadata-item {
            margin: 5px 0;
            color: #666;
        }
        
        @media print {
            .sidebar { display: none; }
            .content { margin-left: 0; }
            .scenario-section { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h3>Scenarios</h3>
            <ul>
    ''']
    
    # Add navigation links
    for i, scenario_path in enumerate(sorted(grouped_results.keys())):
        scenario_name = Path(scenario_path).name
        html_parts.append(f'<li><a href="#scenario-{i}">{html.escape(scenario_name)}</a></li>')
    
    html_parts.append('''
            </ul>
        </div>
        <div class="content">
    ''')
    
    # Add metadata if available
    if metadata:
        html_parts.append('<div class="metadata">')
        html_parts.append('<h3>Evaluation Metadata</h3>')
        
        if 'created_at' in metadata:
            html_parts.append(f'<div class="metadata-item">Created: {metadata["created_at"]}</div>')
        
        if 'config' in metadata:
            config = metadata['config']
            if 'models' in config:
                model_count = len(config['models'])
                html_parts.append(f'<div class="metadata-item">Models evaluated: {model_count}</div>')
            if 'num_scenarios' in config:
                html_parts.append(f'<div class="metadata-item">Total scenarios: {config["num_scenarios"]}</div>')
            if 'runs_per_model' in config:
                html_parts.append(f'<div class="metadata-item">Runs per model: {config["runs_per_model"]}</div>')
        
        html_parts.append('</div>')
    
    # Add scenario sections
    for i, (scenario_path, results) in enumerate(sorted(grouped_results.items())):
        scenario_name = Path(scenario_path).name
        
        html_parts.append(f'<section class="scenario-section" id="scenario-{i}">')
        html_parts.append(f'<h2>Scenario: {html.escape(scenario_name)}</h2>')
        html_parts.append(f'<div class="path">Path: {html.escape(scenario_path)}</div>')
        
        # Add transcript if available
        transcript = load_scenario_transcript(scenario_path)
        if transcript:
            html_parts.append('<div class="transcript">')
            html_parts.append('<h3>Transcript</h3>')
            html_parts.append(f'<pre>{html.escape(transcript)}</pre>')
            html_parts.append('</div>')
        
        # Add results table
        html_parts.append('<table class="responses-table">')
        html_parts.append('<thead><tr><th>Model</th><th>Response</th><th>Grade</th></tr></thead>')
        html_parts.append('<tbody>')
        
        for result in results:
            model_id = result.get('model_id', 'unknown')
            provider = result.get('provider', '')
            
            html_parts.append('<tr>')
            html_parts.append(f'<td class="model-name">{html.escape(provider)}/{html.escape(model_id)}</td>')
            
            # Handle errors
            if 'error' in result:
                html_parts.append('<td colspan="2">')
                html_parts.append(format_error(result['error']))
                html_parts.append('</td>')
            else:
                # Add responses
                responses = result.get('responses', [])
                if responses:
                    response_html = []
                    for j, response in enumerate(responses):
                        if len(responses) > 1:
                            response_html.append(f'<strong>Run {j+1}:</strong>')
                        response_html.append(f'<div class="response-content">{html.escape(response)}</div>')
                    html_parts.append(f'<td>{" ".join(response_html)}</td>')
                else:
                    html_parts.append('<td><em>No response</em></td>')
                
                # Add grades
                grades = result.get('grades', [])
                if grades:
                    grade_html = []
                    for j, grade_list in enumerate(grades):
                        if len(grades) > 1:
                            grade_html.append(f'<strong>Run {j+1}:</strong>')
                        for grade in grade_list:
                            grade_html.append(format_grade(grade))
                    html_parts.append(f'<td class="grade-cell">{" ".join(grade_html)}</td>')
                else:
                    html_parts.append('<td class="grade-cell"><em>No grades</em></td>')
            
            html_parts.append('</tr>')
        
        html_parts.append('</tbody></table>')
        html_parts.append('</section>')
    
    html_parts.append('''
        </div>
    </div>
</body>
</html>
    ''')
    
    return '\n'.join(html_parts)


def generate_results_html(
    results_path: Path,
    output_path: Path,
    scenario_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    include_errors: bool = False,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    limit: Optional[int] = None
) -> None:
    """Generate HTML viewer for evaluation results."""
    print(f"Loading results from {results_path}...")
    
    # Load and filter results
    data = load_and_filter_results(
        results_path,
        scenario_filter=scenario_filter,
        model_filter=model_filter,
        include_errors=include_errors,
        min_score=min_score,
        max_score=max_score,
        limit=limit
    )
    
    results = data['results']
    metadata = data.get('metadata')
    
    print(f"Filtered to {len(results)} results")
    
    # Group by scenario
    grouped = group_by_scenario(results)
    print(f"Found {len(grouped)} unique scenarios")
    
    # Generate HTML
    html = render_html(grouped, metadata)
    
    # Write to file
    output_path.write_text(html)
    print(f"Results viewer saved to: {output_path}")


def parse_transcript_messages(transcript: str) -> List[Tuple[str, str]]:
    """Parse transcript into list of (tag_type, content) tuples."""
    messages = []
    import re
    
    # Regular expression to match any XML-like tag and its content
    pattern = r'<(\w+(?::\w+)?|tool_result)>(.*?)</\1>'
    matches = re.finditer(pattern, transcript, re.DOTALL)
    
    for match in matches:
        tag_type = match.group(1)
        content = match.group(2).strip()
        
        # Normalize tag types
        if tag_type.startswith('tool_call'):
            display_type = 'tool-call'
        elif tag_type == 'tool_result':
            display_type = 'tool-result'
        else:
            display_type = tag_type
            
        messages.append((display_type, content))
    
    return messages


def render_scenario_html(scenario_path: Path) -> str:
    """Render a single scenario as HTML."""
    try:
        # Load the scenario
        scenario = load_scenario_from_dir(scenario_path)
        
        # Get the transcript
        transcript = render_transcript(scenario)
        
        # Generate HTML
        html_parts = ['''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Scenario: ''' + html.escape(scenario_path.name) + '''</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        
        .container {
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h1 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 15px;
        }
        
        .path {
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
            font-family: monospace;
        }
        
        .transcript {
            margin: 20px 0;
        }
        
        .transcript h2 {
            margin-top: 0;
            color: #555;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        
        .message-box {
            background: #f8f8f8;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        
        .message-box.user {
            background: #e3f2fd;
            border-color: #90caf9;
        }
        
        .message-box.agent {
            background: #f1f8e9;
            border-color: #aed581;
        }
        
        .message-box.system {
            background: #fce4ec;
            border-color: #f48fb1;
        }
        
        .message-box.tool-call {
            background: #fff3e0;
            border-color: #ffb74d;
        }
        
        .message-box.tool-result {
            background: #f3e5f5;
            border-color: #ce93d8;
        }
        
        .message-label {
            font-weight: 600;
            color: #555;
            margin-bottom: 8px;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }
        
        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
        }
        
        
        .metadata {
            background: #f0f0f0;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        
        .metadata h3 {
            margin-top: 0;
            color: #555;
        }
        
        .metadata-item {
            margin: 5px 0;
            font-size: 14px;
        }
        
        
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Scenario: ''' + html.escape(scenario_path.name) + '''</h1>
        <div class="path">''' + html.escape(str(scenario_path)) + '''</div>
        ''']
        
        # Add metadata if available
        if scenario.system or scenario.tools or scenario.extra_items:
            html_parts.append('<div class="metadata">')
            html_parts.append('<h3>Scenario Metadata</h3>')
            
            if scenario.system:
                html_parts.append('<div class="metadata-item"><strong>System prompt:</strong> Yes</div>')
            
            if scenario.tools:
                html_parts.append(f'<div class="metadata-item"><strong>Tools:</strong> {len(scenario.tools)} defined</div>')
            
            if scenario.extra_items:
                html_parts.append(f'<div class="metadata-item"><strong>Extra items:</strong> {", ".join(scenario.extra_items.keys())}</div>')
            
            html_parts.append('</div>')
        
        # Add transcript
        html_parts.append('<div class="transcript">')
        html_parts.append('<h2>Transcript</h2>')
        
        # Parse messages and render each in its own box
        messages = parse_transcript_messages(transcript)
        
        for msg_type, content in messages:
            # Create appropriate label for each message type
            label_map = {
                'user': 'User',
                'agent': 'Agent',
                'system': 'System',
                'tool-call': 'Tool Call',
                'tool-result': 'Tool Result'
            }
            label = label_map.get(msg_type, msg_type.title())
            
            html_parts.append(f'<div class="message-box {msg_type}">')
            html_parts.append(f'<div class="message-label">{label}</div>')
            html_parts.append(f'<div class="message-content">{html.escape(content)}</div>')
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        html_parts.append('''
    </div>
</body>
</html>
        ''')
        
        return '\n'.join(html_parts)
        
    except Exception as e:
        # Return error page
        return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Error Loading Scenario</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }}
        .error {{ background: #fee; border: 1px solid #fcc; padding: 20px; border-radius: 5px; color: #c00; }}
    </style>
</head>
<body>
    <h1>Error Loading Scenario</h1>
    <div class="error">
        <strong>Path:</strong> {html.escape(str(scenario_path))}<br>
        <strong>Error:</strong> {html.escape(str(e))}
    </div>
</body>
</html>'''


def generate_scenario_html(scenario_path: Path, output_path: Path) -> None:
    """Generate HTML viewer for a single scenario."""
    print(f"Loading scenario from {scenario_path}...")
    
    if not scenario_path.exists():
        print(f"Error: Scenario path does not exist: {scenario_path}")
        return
    
    if not scenario_path.is_dir():
        print(f"Error: Scenario path is not a directory: {scenario_path}")
        return
    
    # Generate HTML
    html = render_scenario_html(scenario_path)
    
    # Write to file
    output_path.write_text(html)
    print(f"Scenario viewer saved to: {output_path}")