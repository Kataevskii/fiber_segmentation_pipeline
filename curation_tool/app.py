import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from flask import Flask, request, jsonify, render_template_string
from curation_tool.engine import RealDataCurationEngine

app = Flask(__name__)

_default_raw = 'data/fibers_to_segment/COLLAGENCROP_003_0000.tif' if os.path.exists('data/fibers_to_segment/COLLAGENCROP_003_0000.tif') else (
    sorted([os.path.join('data/fibers_to_segment', f).replace('\\', '/') for f in os.listdir('data/fibers_to_segment') if f.lower().endswith(('.tif', '.tiff', '.npy'))])[0]
    if os.path.isdir('data/fibers_to_segment') and any(f.lower().endswith(('.tif', '.tiff', '.npy')) for f in os.listdir('data/fibers_to_segment'))
    else None
)
engine = RealDataCurationEngine(
    raw_volume_path=_default_raw,
    curated_output_dir='data/curated/patches',
    cube_size=96
)

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fiber Curator - 3D Box Annotator & Geodesic Resolver</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <style>
        :root {
            --bg-base: #090b10;
            --bg-surface: #10131d;
            --bg-surface-raised: #181c2b;
            --bg-card: #141724;
            --border-subtle: #222738;
            --border-accent: #353d56;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #586069;
            --accent-cyan: #00e5ff;
            --accent-orange: #ff6d00;
            --accent-lime: #00e676;
            --accent-magenta: #f50057;
            --accent-yellow: #ffd600;
            --accent-purple: #b388ff;
            --btn-primary: #00e5ff;
            --btn-primary-text: #051016;
            --btn-success: #00e676;
            --btn-success-text: #051a0e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            color: var(--text-primary);
            height: 100vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        header {
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-subtle);
            padding: 6px 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            z-index: 10;
            flex-shrink: 0;
            min-height: 46px;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-shrink: 0;
        }

        .app-title {
            font-size: 13px;
            font-weight: 800;
            letter-spacing: -0.2px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
            white-space: nowrap;
        }

        .app-title span {
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 229, 255, 0.12);
            padding: 1px 5px;
            border-radius: 4px;
            font-size: 10px;
        }

        .btn-header-upload {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.20) 0%, rgba(0, 230, 118, 0.15) 100%);
            border: 1px solid var(--accent-cyan);
            color: #ffffff;
            padding: 4px 10px;
            border-radius: 5px;
            font-size: 11.5px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-header-upload:hover {
            background: var(--accent-cyan);
            color: #051016;
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.45);
        }

        .btn-header-presets {
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 4px 8px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 4px;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-header-presets:hover {
            border-color: var(--border-medium);
            color: var(--text-primary);
        }

        .source-pill {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10.5px;
            color: var(--accent-lime);
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            cursor: pointer;
            transition: border-color 0.15s ease;
        }

        .source-pill:hover {
            border-color: var(--accent-cyan);
        }

        /* Center Group: Origin Capsule */
        .header-center {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .coord-capsule {
            display: flex;
            align-items: center;
            gap: 3px;
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            padding: 2px 6px;
            border-radius: 6px;
        }

        .coord-title {
            font-size: 10.5px;
            font-weight: 700;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            margin-right: 2px;
        }

        .c-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 9.5px;
            font-weight: 800;
            color: var(--accent-cyan);
            margin-left: 3px;
        }

        .c-input {
            width: 38px;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 3px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            text-align: center;
            padding: 2px 0;
            outline: none;
            -moz-appearance: textfield;
        }

        .c-input::-webkit-outer-spin-button,
        .c-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        .c-input:focus {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .btn-go-coords {
            background: rgba(0, 229, 255, 0.15);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 10.5px;
            font-weight: 800;
            cursor: pointer;
            margin-left: 4px;
            transition: all 0.15s ease;
        }

        .btn-go-coords:hover {
            background: var(--accent-cyan);
            color: #051016;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.3);
        }

        .btn-header-icon {
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 4px 8px;
            border-radius: 5px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-header-icon:hover {
            border-color: var(--accent-cyan);
            color: var(--text-primary);
        }

        .btn-header-icon.danger:hover {
            border-color: var(--accent-magenta);
            color: var(--accent-magenta);
            background: rgba(245, 0, 87, 0.1);
        }

        /* Right Group */
        .header-right {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-shrink: 0;
        }

        .dropdown-select-compact {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            padding: 4px 6px;
            border-radius: 5px;
            font-size: 11px;
            font-family: inherit;
            cursor: pointer;
            outline: none;
            max-width: 140px;
        }

        .dropdown-select-compact:hover {
            border-color: var(--accent-cyan);
        }

        .btn-action-resolve {
            background: var(--accent-cyan);
            color: #051016;
            border: 1px solid var(--accent-cyan);
            font-size: 11.5px;
            font-weight: 700;
            padding: 4px 11px;
            border-radius: 5px;
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.3);
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-action-resolve:hover {
            background: #40e9ff;
            box-shadow: 0 0 14px rgba(0, 229, 255, 0.6);
        }

        .btn-action-save {
            background: var(--accent-lime);
            color: #051016;
            border: 1px solid var(--accent-lime);
            font-size: 11.5px;
            font-weight: 700;
            padding: 4px 11px;
            border-radius: 5px;
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0, 230, 118, 0.3);
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-action-save:hover {
            background: #40ff99;
            box-shadow: 0 0 14px rgba(0, 230, 118, 0.6);
        }

        .curated-badge-compact {
            background: rgba(0, 230, 118, 0.10);
            border: 1px solid rgba(0, 230, 118, 0.3);
            color: var(--accent-lime);
            padding: 3px 7px;
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            white-space: nowrap;
        }



        /* 3D Point Density Slider */
        .slider-toggle-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .slider-label {
            font-weight: 700;
            color: var(--accent-cyan);
            white-space: nowrap;
        }

        .density-range-input {
            width: 75px;
            height: 4px;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        .slider-val-tag {
            font-size: 10px;
            font-weight: 800;
            color: var(--text-primary);
            min-width: 28px;
            text-align: right;
        }

        button {
            font-family: 'Inter', sans-serif;
            font-size: 12px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid transparent;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: all 0.15s ease;
        }

        .btn-secondary {
            background: var(--bg-surface-raised);
            border-color: var(--border-subtle);
            color: var(--text-primary);
        }

        .btn-secondary:hover {
            background: var(--border-accent);
            border-color: var(--text-secondary);
        }

        .btn-resolve {
            background: var(--accent-cyan);
            color: var(--btn-primary-text);
        }

        .btn-resolve:hover {
            filter: brightness(1.15);
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
        }

        .btn-save {
            background: var(--accent-lime);
            color: var(--btn-success-text);
        }

        .btn-save:hover {
            filter: brightness(1.15);
            box-shadow: 0 0 10px rgba(0, 230, 118, 0.4);
        }

        /* Workspace Layout - 60% Left Stage / 40% Right Controls Proportional Split */
        .workspace {
            display: grid;
            grid-template-columns: minmax(480px, 6fr) minmax(360px, 4fr);
            height: calc(100vh - 54px);
            gap: 14px;
            padding: 10px 14px;
            overflow: hidden;
            min-height: 0;
            min-width: 0;
        }

        /* Stage Panel & Bounding Box */
        .three-stage-panel {
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            position: relative;
            height: 100%;
            min-width: 0;
            overflow: hidden;
        }

        .stage-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 5px 8px;
            flex-shrink: 0;
            min-width: 0;
        }

        .toggles-cluster {
            display: flex;
            gap: 6px;
            align-items: center;
            flex-shrink: 0;
        }

        .toggle-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            font-size: 11px;
            color: var(--text-secondary);
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            padding: 3px 7px;
            border-radius: 5px;
            cursor: pointer;
            user-select: none;
            transition: all 0.15s ease;
        }

        .toggle-pill:hover {
            border-color: var(--border-medium);
            color: var(--text-primary);
        }

        .toggle-pill input {
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        /* Unified Fiber Controller Capsule */
        .fiber-control-capsule {
            display: flex;
            align-items: center;
            gap: 6px;
            background: var(--bg-card);
            border: 1px solid var(--border-medium);
            padding: 2px 6px;
            border-radius: 6px;
            flex-shrink: 0;
        }

        .fiber-nav-btn {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            color: var(--text-primary);
            width: 22px;
            height: 22px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 10px;
            font-weight: 700;
            transition: all 0.15s;
        }

        .fiber-nav-btn:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .active-fiber-badge {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 22px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 800;
            font-size: 12px;
            color: #000;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.35);
            padding: 0 4px;
            gap: 2px;
        }

        .active-fiber-input {
            width: 28px;
            background: transparent;
            border: none;
            outline: none;
            font-family: inherit;
            font-weight: 900;
            font-size: 12px;
            color: #000;
            text-align: center;
            -moz-appearance: textfield;
        }

        .active-fiber-input::-webkit-outer-spin-button,
        .active-fiber-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        .fiber-seed-dots {
            display: inline-flex;
            gap: 2px;
            font-size: 11px;
            letter-spacing: -1px;
            user-select: none;
        }

        .btn-new-fiber {
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s ease;
        }

        .btn-new-fiber:hover {
            background: var(--accent-cyan);
            color: #000;
        }

        /* Horizontally Scrollable Assigned Chips Strip */
        .fiber-chips-strip {
            display: flex;
            gap: 4px;
            overflow-x: auto;
            white-space: nowrap;
            flex-wrap: nowrap;
            min-width: 0;
            flex: 1;
            align-items: center;
            scrollbar-width: thin;
            padding: 1px 0;
        }

        .fiber-chips-strip::-webkit-scrollbar {
            height: 3px;
        }

        .fiber-chips-strip::-webkit-scrollbar-thumb {
            background: var(--border-medium);
            border-radius: 2px;
        }

        .strip-pill {
            min-width: 20px;
            height: 20px;
            padding: 0 4px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 800;
            color: #000;
            cursor: pointer;
            border: 1px solid transparent;
            flex-shrink: 0;
            transition: all 0.15s ease;
        }

        .strip-pill.active {
            border-color: #fff;
            transform: scale(1.14);
            box-shadow: 0 0 6px rgba(255, 255, 255, 0.7);
        }

        .dropdown-select {
            background: var(--bg-surface-raised);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-family: inherit;
            cursor: pointer;
            outline: none;
            transition: border-color 0.15s;
        }

        .dropdown-select:hover {
            border-color: var(--accent-cyan);
        }

        /* 3D Viewport */
        #three-container {
            flex: 1;
            min-height: 380px;
            background: radial-gradient(circle at center, #141826 0%, #090b10 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            position: relative;
            cursor: grab;
            overflow: hidden;
        }

        #three-container:active {
            cursor: grabbing;
        }

        .three-overlay-hint {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(16, 19, 29, 0.88);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border-subtle);
            padding: 5px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-secondary);
            pointer-events: none;
            z-index: 5;
        }

        .three-overlay-hint strong {
            color: var(--accent-cyan);
        }

        /* Placed Fiber Seeds Card in Right Panel */
        .seeds-dock {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex-shrink: 0;
        }

        .dock-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .seeds-chip-container {
            display: flex;
            flex-direction: column;
            gap: 5px;
            max-height: 220px;
            overflow-y: auto;
            padding-right: 4px;
        }

        .fiber-chip {
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            border-radius: 5px;
            padding: 4px 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            transition: all 0.15s ease;
        }

        .fiber-chip:hover {
            border-color: var(--border-medium);
        }

        .chip-left-group {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .chip-color-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .chip-status {
            font-size: 9px;
            font-weight: 700;
            padding: 1px 4px;
            border-radius: 3px;
        }

        .chip-status.connected {
            background: rgba(0, 230, 118, 0.2);
            color: var(--accent-lime);
        }

        .chip-status.single {
            background: rgba(255, 214, 0, 0.2);
            color: var(--accent-yellow);
        }

        .chip-del-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 0 2px;
            font-size: 11px;
        }

        .chip-del-btn:hover {
            color: var(--accent-magenta);
        }


        .chip-wp-tag {
            background: rgba(0, 229, 255, 0.15);
            border: 1px dashed var(--accent-cyan);
            color: var(--accent-cyan);
            border-radius: 4px;
            padding: 1px 5px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }

        /* Right Panel: Controls & Crop ROI */
        .preview-panel {
            background: var(--bg-surface);
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
            border-radius: 8px;
            border: 1px solid var(--border-subtle);
        }

        .panel-title {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
        }

        .info-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            padding: 1px 0;
        }

        .info-label {
            color: var(--text-secondary);
        }

        .info-value {
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Visual Sub-Box Crop Tool Card */
        .crop-card {
            background: var(--bg-card);
            border: 1px solid var(--border-accent);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .crop-row {
            display: grid;
            grid-template-columns: 20px 38px 1fr 1fr 38px 38px;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
        }

        .crop-axis-label {
            font-weight: 700;
            font-size: 11px;
            color: var(--accent-cyan);
            text-align: center;
        }

        .crop-range-slider {
            width: 100%;
            height: 5px;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        .crop-num-input {
            width: 38px;
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            text-align: center;
            padding: 2px 0;
            outline: none;
            -moz-appearance: textfield;
        }

        .crop-num-input::-webkit-outer-spin-button,
        .crop-num-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        .crop-num-input:focus {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .crop-presets-row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            align-items: center;
        }

        .btn-crop-action {
            flex: 1 1 auto;
            text-align: center;
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 10.5px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-crop-action:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.08);
        }

        .btn-crop-action.highlight {
            background: rgba(0, 229, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            font-weight: 700;
        }

        .shortcuts-hint {
            display: flex;
            gap: 10px;
            font-size: 10px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            justify-content: center;
        }

        .shortcuts-hint kbd {
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            padding: 1px 4px;
            border-radius: 3px;
            color: var(--text-secondary);
        }

        /* Modal Styles */
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(4, 6, 12, 0.85);
            backdrop-filter: blur(10px);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-accent);
            border-radius: 12px;
            width: 660px;
            max-width: 92vw;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.7), 0 0 30px rgba(0, 229, 255, 0.15);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            animation: modalFadeIn 0.2s ease-out;
        }

        @keyframes modalFadeIn {
            from { opacity: 0; transform: scale(0.96); }
            to { opacity: 1; transform: scale(1); }
        }

        .modal-header {
            padding: 14px 20px;
            background: var(--bg-surface-raised);
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-title {
            font-size: 14px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .modal-close-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 16px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            transition: all 0.15s ease;
        }

        .modal-close-btn:hover {
            color: var(--accent-magenta);
            background: rgba(245, 0, 87, 0.1);
        }

        .modal-body {
            padding: 18px 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            max-height: 72vh;
            overflow-y: auto;
        }

        .modal-section {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .modal-label {
            font-size: 11.5px;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .modal-hint {
            font-size: 10.5px;
            color: var(--text-secondary);
            line-height: 1.4;
        }

        .modal-text-input {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11.5px;
            outline: none;
            width: 100%;
            transition: border-color 0.15s ease;
        }

        .modal-text-input:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.25);
        }

        .preset-pills-grid {
            display: flex;
            flex-direction: column;
            gap: 5px;
            max-height: 190px;
            overflow-y: auto;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 8px;
        }

        .preset-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            border-radius: 5px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.12s ease;
        }

        .preset-item:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.08);
            transform: translateX(2px);
        }

        .preset-item-name {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .preset-tag {
            font-size: 9.5px;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 700;
            font-family: 'Inter', sans-serif;
            flex-shrink: 0;
            margin-left: 8px;
        }

        .preset-tag.segmented {
            background: rgba(0, 230, 118, 0.15);
            color: var(--accent-lime);
            border: 1px solid rgba(0, 230, 118, 0.3);
        }

        .preset-tag.raw {
            background: rgba(0, 229, 255, 0.15);
            color: var(--accent-cyan);
            border: 1px solid rgba(0, 229, 255, 0.3);
        }

        .modal-coord-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .modal-coord-inputs {
            display: flex;
            align-items: center;
            gap: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .modal-footer {
            padding: 12px 20px;
            border-top: 1px solid var(--border-subtle);
            background: var(--bg-surface-raised);
            display: flex;
            justify-content: flex-end;
            gap: 10px;
        }

        .btn-open-file {
            background: rgba(0, 229, 255, 0.15);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 11.5px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
            white-space: nowrap;
        }

        .btn-open-file:hover {
            background: var(--accent-cyan);
            color: #051016;
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.4);
        }

        .btn-open-file.highlight-btn {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.25) 0%, rgba(0, 230, 118, 0.20) 100%);
            border-color: var(--accent-cyan);
            color: #ffffff;
            font-weight: 800;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.25);
        }

        .btn-open-file.highlight-btn:hover {
            background: var(--accent-cyan);
            color: #051016;
            box-shadow: 0 0 16px rgba(0, 229, 255, 0.6);
        }

        /* PC Picker Card */
        .pc-picker-card {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.10) 0%, rgba(0, 230, 118, 0.08) 100%);
            border: 1.5px dashed var(--accent-cyan);
            border-radius: 8px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            transition: all 0.2s ease;
        }

        .pc-picker-card:hover {
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.16) 0%, rgba(0, 230, 118, 0.12) 100%);
            border-color: #40e9ff;
        }

        .btn-pc-browse {
            background: var(--accent-cyan);
            color: #051016;
            border: 1px solid var(--accent-cyan);
            font-size: 12px;
            font-weight: 700;
            padding: 7px 14px;
            border-radius: 6px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.15s ease;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
        }

        .btn-pc-browse:hover {
            background: #40e9ff;
            box-shadow: 0 0 16px rgba(0, 229, 255, 0.6);
            transform: translateY(-1px);
        }

        .btn-pc-browse.secondary {
            background: var(--bg-surface-raised);
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            box-shadow: none;
        }

        .btn-pc-browse.secondary:hover {
            border-color: var(--accent-lime);
            color: var(--accent-lime);
            background: rgba(0, 230, 118, 0.12);
        }
    </style>
</head>
<body>

    <!-- Open File Modal -->
    <div id="file-modal" class="modal-overlay" style="display: none;">
        <div class="modal-card">
            <div class="modal-header">
                <div class="modal-title">📂 Open 3D Volume or Segmented Instance TIFF</div>
                <button class="modal-close-btn" onclick="closeFileModal()">✕</button>
            </div>
            <div class="modal-body">
                <!-- Upload Volume Card -->
                <div class="pc-picker-card">
                    <div style="font-size: 13px; font-weight: 700; color: var(--text-primary); display: flex; align-items: center; gap: 6px;">
                        <span>📁 Upload Volume File</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-secondary); line-height: 1.4;">
                        Select or drag-and-drop any raw volume or segmented instance TIFF (.tif, .tiff, .npy) from your PC.
                    </div>
                    <div style="margin-top: 4px;">
                        <label class="btn-pc-browse" style="cursor: pointer;">
                            📁 Choose File from PC...
                            <input type="file" id="modal-browser-file-input" accept=".tif,.tiff,.npy" style="display: none;" onchange="handleBrowserFileSelect(this)">
                        </label>
                    </div>
                </div>

                <div class="modal-section">
                    <label class="modal-label">🎯 Or Specify File Path (.tif, .tiff, .npy):</label>
                    <input type="text" id="modal-file-path" class="modal-text-input" placeholder="e.g. outputs/topology_resolved_morpho/instance_volume.tif" value="outputs/topology_resolved_morpho/instance_volume.tif">
                    <div class="modal-hint">Opening a segmented instance volume automatically skeletonizes labels and extracts interactive fiber centerlines & wiring.</div>
                </div>

                <div class="modal-section">
                    <label class="modal-label">⭐ Quick Presets / Detected Volumes:</label>
                    <div class="preset-pills-grid" id="modal-preset-pills">
                        <div style="color: var(--text-muted); font-size: 11px; padding: 4px;">Loading available workspace files...</div>
                    </div>
                </div>

                <div class="modal-section" id="modal-raw-pairing-section">
                    <label class="modal-label">🔬 Optional Raw Volume (for intensity/density):</label>
                    <input type="text" id="modal-raw-path" class="modal-text-input" placeholder="e.g. data/fibers_to_segment/COLLAGENCROP_003_0000.tif (optional)">
                </div>

                <div class="modal-section">
                    <label class="modal-label">📍 Initial Patch Location:</label>
                    <div class="modal-coord-row">
                        <label class="toggle-pill">
                            <input type="radio" name="modal-extract-mode" id="modal-mode-random" value="random" checked onchange="toggleModalCoordMode()">
                            <span>🎲 Random High-Density Patch</span>
                        </label>
                        <label class="toggle-pill">
                            <input type="radio" name="modal-extract-mode" id="modal-mode-manual" value="manual" onchange="toggleModalCoordMode()">
                            <span>📍 Manual Origin (Z, Y, X)</span>
                        </label>
                    </div>
                    <div class="modal-coord-inputs" id="modal-manual-coords" style="display: none; margin-top: 6px;">
                        <span>Z:</span> <input type="number" id="modal-z" class="crop-num-input" value="100" style="width: 55px;">
                        <span>Y:</span> <input type="number" id="modal-y" class="crop-num-input" value="100" style="width: 55px;">
                        <span>X:</span> <input type="number" id="modal-x" class="crop-num-input" value="100" style="width: 55px;">
                    </div>
                </div>
            </div>
            <div class="modal-footer">
                <button class="btn-secondary" onclick="closeFileModal()">Cancel</button>
                <button class="btn-resolve" onclick="submitOpenFile()" id="btn-modal-open">🚀 Open & Inspect Patch</button>
            </div>
        </div>
    </div>

    <!-- Header (Clean, Decluttered Layout) -->
    <header>
        <!-- Left: Brand + Direct Upload + Presets + Source Badge -->
        <div class="header-left">
            <div class="app-title">FIBER <span>CURATOR</span></div>
            
            <label class="btn-header-upload" title="Upload any .tif, .tiff, or .npy volume from your PC">
                📁 Upload Volume
                <input type="file" id="browser-file-input" accept=".tif,.tiff,.npy" style="display:none;" onchange="handleBrowserFileSelect(this)">
            </label>

            <button class="btn-header-presets" onclick="openFileModal()" title="Browse workspace presets or type custom path">
                📂 Presets
            </button>

            <div class="source-pill" id="source-pill" onclick="openFileModal()" title="Active Volume (Click to change)">
                <span id="source-name">COLLAGENCROP_003_0000.tif</span>
            </div>
        </div>

        <!-- Center: Coordinate Origin Input + Random Button -->
        <div class="header-center">
            <div class="coord-capsule" title="Type (Z, Y, X) origin coordinates and press Enter or click Go">
                <span class="coord-title">Origin:</span>
                <span class="c-tag">Z</span><input type="number" id="load-coord-z" class="c-input global-coord-input" min="0" max="404" value="0" onkeydown="handleCoordKey(event)">
                <span class="c-tag">Y</span><input type="number" id="load-coord-y" class="c-input global-coord-input" min="0" max="404" value="0" onkeydown="handleCoordKey(event)">
                <span class="c-tag">X</span><input type="number" id="load-coord-x" class="c-input global-coord-input" min="0" max="404" value="0" onkeydown="handleCoordKey(event)">
                <button class="btn-go-coords" onclick="loadManualCoordinates()" title="Jump to coordinates [Enter]">Go ↵</button>
            </div>

            <button class="btn-header-icon" onclick="fetchNewRandomPatch()" title="Load Random Cube [R]">
                🎲 Random [R]
            </button>
        </div>

        <!-- Right: Saved Cubes + Primary Action Buttons -->
        <div class="header-right">
            <select id="saved-patches-select" class="dropdown-select-compact" onchange="loadSavedPatch(this.value)" title="Load an already-saved patch">
                <option value="">📂 Saved (0)...</option>
            </select>

            <button class="btn-action-resolve" onclick="resolveConnections()" title="Re-solve continuous geodesic paths [Space]">
                ⚡ Resolve [Space]
            </button>

            <button class="btn-action-save" onclick="saveAndNext()" title="Save Ground Truth patch & load next [Enter]">
                ✅ Save & Next [Enter]
            </button>

            <div class="curated-badge-compact" id="total-badge" title="Total Curated Patches">
                💾 <strong id="curated-count">0</strong>
            </div>

            <button class="btn-header-icon danger" onclick="clearAllSeeds()" title="Clear all seeds in current patch">
                🗑️
            </button>
        </div>
    </header>

    <!-- Workspace Layout -->
    <div class="workspace">
        
        <div class="three-stage-panel">
            <div class="stage-header">
                <!-- Viewport Density Slider -->
                <div class="toggles-cluster">
                    <label class="toggle-pill" title="Show / Hide 3D Coordinate Arrows next to the cube">
                        <input type="checkbox" id="toggle-axes-arrows" checked onchange="toggleAxesVisibility(this.checked)">
                        <span>🧭 Axes</span>
                    </label>

                    <div class="slider-toggle-pill" title="3D Fiber Point Cloud Density (0% to 100%)">
                        <span class="slider-label">✨ Points:</span>
                        <input type="range" id="slider-point-density" class="density-range-input" min="0" max="100" value="40" step="1" oninput="onPointDensityInput(this.value)">
                        <span class="slider-val-tag" id="val-point-density">40%</span>
                    </div>
                </div>

                <!-- Ergonomic Active Fiber Capsule -->
                <div class="fiber-control-capsule">
                    <button class="fiber-nav-btn" onclick="setActiveId(activeFiberId - 1)" title="Previous Fiber ID ([ or -])">◀</button>
                    <div class="active-fiber-badge" id="active-id-badge" style="background-color: #00e5ff;">
                        <span>#</span>
                        <input type="number" id="active-id-input" class="active-fiber-input" min="1" max="99" value="1" 
                                onchange="setActiveId(this.value)" oninput="setActiveId(this.value)" title="Type or scroll Fiber ID">
                    </div>
                    <button class="fiber-nav-btn" onclick="setActiveId(activeFiberId + 1)" title="Next Fiber ID (] or +)">▶</button>
                    
                    <div class="fiber-seed-dots" id="active-fiber-dots" title="Seeds placed for active fiber">○○</div>

                    <button class="btn-new-fiber" onclick="selectNextUnusedId()" title="Jump to next unused Fiber ID (+)">
                        ✨ + Next
                    </button>
                </div>

                <!-- Quick Assigned Chips Strip -->
                <div class="fiber-chips-strip" id="id-quick-strip" title="Click assigned fiber chip to switch"></div>
            </div>

            <!-- 3D Interactive Box Viewport -->
            <div id="three-container">
                <div class="three-overlay-hint">
                    🖱️ <strong>Click 3D Face</strong>: Drop/Pair Seed | <strong>Drag</strong>: Rotate | <strong>Wheel</strong>: Zoom
                </div>
            </div>

            <div class="shortcuts-hint">
                <span><kbd>Click 3D Face</kbd> Drop Seed</span>
                <span><kbd>Space</kbd> Resolve</span>
                <span><kbd>Enter</kbd> Save & Next</span>
                <span><kbd>R</kbd> Random Cube</span>
                <span><kbd>F</kbd> Focus Active Fiber Crop</span>
                <span><kbd>C</kbd> Reset Crop</span>
                <span><kbd>[ / ]</kbd> Prev/Next ID</span>
            </div>
        </div>

        <!-- Right: Controls & Crop ROI -->
        <div class="preview-panel">
            <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center;">
                <div class="panel-title">Controls & Sub-Box ROI</div>
                <div class="badge highlight" id="perf-badge">Zero-Model Geodesic Engine</div>
            </div>

            <!-- Visual Sub-Box Crop Tool Card -->
            <div class="crop-card" id="crop-card">
                <div class="panel-title">
                    <span>✂️ Visual Sub-Box ROI</span>
                </div>

                <!-- 3-Axis Sliders -->
                <div style="display: flex; flex-direction: column; gap: 4px;">
                    <!-- Z Axis (Depth) -->
                    <div class="crop-row">
                        <span class="crop-axis-label">Z:</span>
                        <input type="number" id="crop-z-min" class="crop-num-input" min="0" max="95" step="1" value="0" oninput="onCropRangeInput('z', 'min', this.value)" onwheel="handleCropWheel(event, 'z', 'min')">
                        <input type="range" id="crop-z-min-slider" class="crop-range-slider" min="0" max="95" step="1" value="0" oninput="onCropSliderInput('z', 'min', this.value)" onwheel="handleCropWheel(event, 'z', 'min')">
                        <input type="range" id="crop-z-max-slider" class="crop-range-slider" min="0" max="95" step="1" value="95" oninput="onCropSliderInput('z', 'max', this.value)" onwheel="handleCropWheel(event, 'z', 'max')">
                        <input type="number" id="crop-z-max" class="crop-num-input" min="0" max="95" step="1" value="95" oninput="onCropRangeInput('z', 'max', this.value)" onwheel="handleCropWheel(event, 'z', 'max')">
                        <span id="crop-z-span" class="badge" style="padding: 1px 4px; font-size: 9px;">Δ96</span>
                    </div>

                    <!-- Y Axis (Height) -->
                    <div class="crop-row">
                        <span class="crop-axis-label">Y:</span>
                        <input type="number" id="crop-y-min" class="crop-num-input" min="0" max="95" step="1" value="0" oninput="onCropRangeInput('y', 'min', this.value)" onwheel="handleCropWheel(event, 'y', 'min')">
                        <input type="range" id="crop-y-min-slider" class="crop-range-slider" min="0" max="95" step="1" value="0" oninput="onCropSliderInput('y', 'min', this.value)" onwheel="handleCropWheel(event, 'y', 'min')">
                        <input type="range" id="crop-y-max-slider" class="crop-range-slider" min="0" max="95" step="1" value="95" oninput="onCropSliderInput('y', 'max', this.value)" onwheel="handleCropWheel(event, 'y', 'max')">
                        <input type="number" id="crop-y-max" class="crop-num-input" min="0" max="95" step="1" value="95" oninput="onCropRangeInput('y', 'max', this.value)" onwheel="handleCropWheel(event, 'y', 'max')">
                        <span id="crop-y-span" class="badge" style="padding: 1px 4px; font-size: 9px;">Δ96</span>
                    </div>

                    <!-- X Axis (Width) -->
                    <div class="crop-row">
                        <span class="crop-axis-label">X:</span>
                        <input type="number" id="crop-x-min" class="crop-num-input" min="0" max="95" step="1" value="0" oninput="onCropRangeInput('x', 'min', this.value)" onwheel="handleCropWheel(event, 'x', 'min')">
                        <input type="range" id="crop-x-min-slider" class="crop-range-slider" min="0" max="95" step="1" value="0" oninput="onCropSliderInput('x', 'min', this.value)" onwheel="handleCropWheel(event, 'x', 'min')">
                        <input type="range" id="crop-x-max-slider" class="crop-range-slider" min="0" max="95" step="1" value="95" oninput="onCropSliderInput('x', 'max', this.value)" onwheel="handleCropWheel(event, 'x', 'max')">
                        <input type="number" id="crop-x-max" class="crop-num-input" min="0" max="95" step="1" value="95" oninput="onCropRangeInput('x', 'max', this.value)" onwheel="handleCropWheel(event, 'x', 'max')">
                        <span id="crop-x-span" class="badge" style="padding: 1px 4px; font-size: 9px;">Δ96</span>
                    </div>
                </div>

                <!-- Quick Presets -->
                <div class="crop-presets-row">
                    <button class="btn-crop-action highlight" onclick="focusActiveFiberCrop()" title="Auto-crop bounding box around active fiber [Hotkey: F]">🎯 Focus Active [F]</button>
                    <button class="btn-crop-action" onclick="setCropPreset(16, 79, 16, 79, 16, 79)" title="Center 64³ crop">📦 64³</button>
                    <button class="btn-crop-action" onclick="setCropPreset(24, 71, 24, 71, 24, 71)" title="Center 48³ crop">📦 48³</button>
                    <button class="btn-crop-action" onclick="setCropPreset(32, 63, 32, 63, 32, 63)" title="Center 32³ crop">📦 32³</button>
                    <button class="btn-crop-action" onclick="resetCrop()" title="Restore full view">🔄 Reset</button>
                    <button class="btn-crop-action" onclick="centerCameraOnCrop()" title="Center 3D orbit controls on crop box">🔍 Center View</button>
                </div>
            </div>

            <!-- Active Subvolume Info Card -->
            <div class="info-card" id="patch-info-card">
                <div class="panel-title" style="margin-bottom: 2px;">📦 Subvolume Overview</div>
                <div class="info-row">
                    <span class="info-label">Patch Origin (Z, Y, X):</span>
                    <span class="info-value" id="info-origin-val">(0, 0, 0)</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Fiber Density:</span>
                    <span class="info-value" id="info-density-val">0.0%</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Total Curated Dataset:</span>
                    <span class="info-value" id="info-curated-val">0 cubes</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Status:</span>
                    <span class="info-value" id="info-status-val" style="color: var(--accent-cyan);">Ready for Annotation</span>
                </div>
            </div>

            <!-- Placed Fiber Seeds & Pairing Status Card -->
            <div class="seeds-dock" id="seeds-dock-card">
                <div class="dock-header">
                    <div class="panel-title" style="display: flex; align-items: center; gap: 6px;">
                        <span>📌 Placed Fiber Seeds</span>
                    </div>
                    <span id="seeds-count-tag" class="badge" style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace;">0 Seeds</span>
                </div>
                <div class="seeds-chip-container" id="seeds-chip-container">
                    <div style="color: var(--text-muted); font-size: 11px;">No seeds placed. Click on 3D Box faces to place fiber seeds!</div>
                </div>
            </div>
        </div>

    </div>

    <!-- Client Script -->
    <script>
        const COLOR_PALETTE = [
            '#00e5ff', '#ff6d00', '#00e676', '#f50057', '#ffd600', '#b388ff', '#2979ff', '#ff1744',
            '#00b0ff', '#ff9100', '#76ff03', '#d500f9', '#ffea00', '#651fff', '#1de9b6', '#ff5252',
            '#40c4ff', '#ffab00', '#69f0ae', '#e040fb', '#eeff41', '#7c4dff', '#64ffda', '#ff3d00',
            '#00bfa5', '#ff6e40', '#a7ffeb', '#ea80fc', '#f4ff81', '#b388ff', '#84ffff', '#ff8a80'
        ];

        let currentPatchData = null;
        let activeFiberId = 1;
        let seedsList = [];
        let isTransparentMode = true;
        let latestResolvedCurves = {};
        let rawPointCloudFull = [];
        let currentPointDensity = 40;

        // Visual Crop State (Always active)
        let cropState = {
            z_min: 0,
            z_max: 95,
            y_min: 0,
            y_max: 95,
            x_min: 0,
            x_max: 95
        };
        let cropDebounceTimer = null;

        // Three.js Global Variables
        let scene, camera, renderer, controls, raycaster, mouse;
        let faceMeshes = [];
        let offscreenCanvases = {};
        let faceTextures = {};
        let seedPinsGroup, curves3DGroup, pointCloudMesh, axes3DGroup;
        let globalWireMesh = null;
        let cropWireMesh = null;
        let isAxesVisible = true;

        function initIdSelector() {
            const badge = document.getElementById('active-id-badge');
            const input = document.getElementById('active-id-input');
            const dots = document.getElementById('active-fiber-dots');
            const strip = document.getElementById('id-quick-strip');

            const color = COLOR_PALETTE[(activeFiberId - 1) % COLOR_PALETTE.length];
            if (badge) badge.style.backgroundColor = color;
            if (input) input.value = activeFiberId;

            // Update seed status dots for active fiber
            const activeSeeds = seedsList.filter(s => s.fiber_id === activeFiberId);
            if (dots) {
                if (activeSeeds.length === 0) {
                    dots.innerHTML = `<span style="color: var(--text-muted); opacity: 0.6;">○○</span>`;
                    dots.title = `Fiber #${activeFiberId}: 0/2 seeds placed (Ready to drop Start)`;
                } else if (activeSeeds.length === 1) {
                    dots.innerHTML = `<span style="color: #ffd600;">●</span><span style="color: var(--text-muted); opacity: 0.6;">○</span>`;
                    dots.title = `Fiber #${activeFiberId}: 1/2 seeds placed (Click 2nd face to pair or leave as singleton)`;
                } else {
                    dots.innerHTML = `<span style="color: #00e676;">●●</span>`;
                    dots.title = `Fiber #${activeFiberId}: 2/2 seeds connected!`;
                }
            }

            if (strip) {
                strip.innerHTML = '';
                const usedSet = new Set(seedsList.map(s => s.fiber_id));
                usedSet.add(activeFiberId);
                const sortedIds = Array.from(usedSet).sort((a, b) => a - b);

                sortedIds.forEach(id => {
                    const pill = document.createElement('div');
                    pill.className = `strip-pill ${id === activeFiberId ? 'active' : ''}`;
                    pill.style.backgroundColor = COLOR_PALETTE[(id - 1) % COLOR_PALETTE.length];
                    const count = seedsList.filter(s => s.fiber_id === id).length;
                    pill.innerText = id;
                    pill.title = `Fiber #${id} (${count}/2 seeds). Click to switch.`;
                    pill.onclick = () => setActiveId(id);
                    strip.appendChild(pill);
                });
            }
        }

        function setActiveId(id) {
            const parsed = parseInt(id);
            activeFiberId = Math.max(1, isNaN(parsed) ? 1 : parsed);
            initIdSelector();
        }

        function selectNextUnusedId() {
            const used = new Set(seedsList.map(s => s.fiber_id));
            let nextId = 1;
            while (used.has(nextId)) {
                nextId++;
            }
            setActiveId(nextId);
        }

        function deleteWaypoint(fid, z, y, x) {
            const idx = seedsList.findIndex(s => s.fiber_id === fid && s.pos3d[0] === z && s.pos3d[1] === y && s.pos3d[2] === x);
            if (idx >= 0) {
                seedsList.splice(idx, 1);
                updateThreeSeeds();
                updateSeedsDock();
                redrawFiberCurveLocally(fid);
            }
        }

        function isCropFull() {
            return cropState.z_min === 0 && cropState.z_max === 95 &&
                   cropState.y_min === 0 && cropState.y_max === 95 &&
                   cropState.x_min === 0 && cropState.x_max === 95;
        }

        function getCropClippingPlanes() {
            if (isCropFull()) return [];
            return [
                new THREE.Plane(new THREE.Vector3( 1,  0,  0), -cropState.x_min),
                new THREE.Plane(new THREE.Vector3(-1,  0,  0),  cropState.x_max + 1),
                new THREE.Plane(new THREE.Vector3( 0,  1,  0), -cropState.y_min),
                new THREE.Plane(new THREE.Vector3( 0, -1,  0),  cropState.y_max + 1),
                new THREE.Plane(new THREE.Vector3( 0,  0,  1), -cropState.z_min),
                new THREE.Plane(new THREE.Vector3( 0,  0, -1),  cropState.z_max + 1)
            ];
        }

        function updateThreeCurvesClipping() {
            if (!curves3DGroup) return;
            const planes = getCropClippingPlanes();
            curves3DGroup.children.forEach(child => {
                if (child.material) {
                    child.material.clippingPlanes = planes;
                    child.material.clipShadows = true;
                    child.material.needsUpdate = true;
                }
            });
        }

        function redrawFiberCurveLocally(fid) {
            const fSeeds = seedsList.filter(s => s.fiber_id === fid);
            if (fSeeds.length < 2) return;

            const controlPoints = fSeeds.map(s => new THREE.Vector3(s.pos3d[2], s.pos3d[1], s.pos3d[0]));

            const oldMesh = curves3DGroup.children.find(c => c.userData && c.userData.fiberId === fid);
            if (oldMesh) {
                curves3DGroup.remove(oldMesh);
                if (oldMesh.geometry) oldMesh.geometry.dispose();
                if (oldMesh.material) oldMesh.material.dispose();
            }

            const curve = new THREE.CatmullRomCurve3(controlPoints);
            const tubeGeo = new THREE.TubeGeometry(curve, 32, 1.3, 8, false);
            const colorHex = parseInt(COLOR_PALETTE[(fid - 1) % COLOR_PALETTE.length].replace('#', '0x'));
            const tubeMat = new THREE.MeshStandardMaterial({
                color: colorHex,
                emissive: colorHex,
                emissiveIntensity: 0.75,
                roughness: 0.3,
                clippingPlanes: getCropClippingPlanes(),
                clipShadows: true
            });
            const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
            tubeMesh.userData = { fiberId: fid };
            curves3DGroup.add(tubeMesh);
        }

        // =========================================================================
        // Visual Crop ROI Logic (Always Active, Step of 1, No Border Highlight)
        // =========================================================================

        function handleCropWheel(event, axis, bound) {
            event.preventDefault();
            const input = document.getElementById(`crop-${axis}-${bound}`);
            if (!input) return;
            let val = parseInt(input.value) || 0;
            val += (event.deltaY < 0 ? 1 : -1);
            val = Math.max(0, Math.min(95, val));
            onCropRangeInput(axis, bound, val);
        }

        function onCropRangeInput(axis, bound, valStr) {
            let val = parseInt(valStr);
            if (isNaN(val)) val = (bound === 'min' ? 0 : 95);
            val = Math.max(0, Math.min(95, val));

            if (bound === 'min') {
                cropState[`${axis}_min`] = Math.min(val, cropState[`${axis}_max`]);
            } else {
                cropState[`${axis}_max`] = Math.max(val, cropState[`${axis}_min`]);
            }

            syncCropUI();
            applyCropView(true);
        }

        function onCropSliderInput(axis, bound, valStr) {
            let val = parseInt(valStr);
            if (isNaN(val)) return;

            if (bound === 'min') {
                if (val > cropState[`${axis}_max`]) {
                    cropState[`${axis}_max`] = val;
                }
                cropState[`${axis}_min`] = val;
            } else {
                if (val < cropState[`${axis}_min`]) {
                    cropState[`${axis}_min`] = val;
                }
                cropState[`${axis}_max`] = val;
            }

            syncCropUI();
            applyCropView(true);
        }

        function setCropPresetValues(z0, z1, y0, y1, x0, x1) {
            cropState.z_min = Math.max(0, Math.min(95, z0));
            cropState.z_max = Math.max(cropState.z_min, Math.min(95, z1));
            cropState.y_min = Math.max(0, Math.min(95, y0));
            cropState.y_max = Math.max(cropState.y_min, Math.min(95, y1));
            cropState.x_min = Math.max(0, Math.min(95, x0));
            cropState.x_max = Math.max(cropState.x_min, Math.min(95, x1));
        }

        function setCropPreset(z0, z1, y0, y1, x0, x1) {
            setCropPresetValues(z0, z1, y0, y1, x0, x1);
            syncCropUI();
            applyCropView(false);
            centerCameraOnCrop();
        }

        function focusActiveFiberCrop() {
            const fSeeds = seedsList.filter(s => s.fiber_id === activeFiberId);
            const curvePts = latestResolvedCurves[activeFiberId] || [];

            const allPts = [];
            fSeeds.forEach(s => allPts.push(s.pos3d));
            curvePts.forEach(p => allPts.push(p));

            if (allPts.length === 0) {
                setCropPreset(24, 71, 24, 71, 24, 71);
                return;
            }

            let minZ = 95, maxZ = 0, minY = 95, maxY = 0, minX = 95, maxX = 0;
            allPts.forEach(p => {
                minZ = Math.min(minZ, p[0]); maxZ = Math.max(maxZ, p[0]);
                minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
                minX = Math.min(minX, p[2]); maxX = Math.max(maxX, p[2]);
            });

            const pad = 10;
            setCropPresetValues(
                Math.max(0, minZ - pad), Math.min(95, maxZ + pad),
                Math.max(0, minY - pad), Math.min(95, maxY + pad),
                Math.max(0, minX - pad), Math.min(95, maxX + pad)
            );

            syncCropUI();
            applyCropView(false);
            centerCameraOnCrop();
        }

        function resetCrop() {
            setCropPresetValues(0, 95, 0, 95, 0, 95);
            syncCropUI();
            applyCropView(false);
            if (controls) controls.target.set(48, 48, 48);
        }

        function centerCameraOnCrop() {
            if (!controls) return;
            const cx = (cropState.x_min + cropState.x_max + 1) / 2;
            const cy = (cropState.y_min + cropState.y_max + 1) / 2;
            const cz = (cropState.z_min + cropState.z_max + 1) / 2;
            controls.target.set(cx, cy, cz);
        }

        function syncCropUI() {
            ['z', 'y', 'x'].forEach(axis => {
                const minVal = cropState[`${axis}_min`];
                const maxVal = cropState[`${axis}_max`];

                const minInput = document.getElementById(`crop-${axis}-min`);
                const maxInput = document.getElementById(`crop-${axis}-max`);
                const minSlider = document.getElementById(`crop-${axis}-min-slider`);
                const maxSlider = document.getElementById(`crop-${axis}-max-slider`);
                const spanTag = document.getElementById(`crop-${axis}-span`);

                if (minInput) minInput.value = minVal;
                if (maxInput) maxInput.value = maxVal;
                if (minSlider) minSlider.value = minVal;
                if (maxSlider) maxSlider.value = maxVal;
                if (spanTag) spanTag.innerText = `Δ${maxVal - minVal + 1}`;
            });
        }

        function applyCropView(debounce = true) {
            updateThreeCropBoxGeometry(cropState.z_min, cropState.z_max, cropState.y_min, cropState.y_max, cropState.x_min, cropState.x_max);
            updateThreePointCloud();
            updateThreeSeeds();

            if (cropDebounceTimer) clearTimeout(cropDebounceTimer);

            if (debounce) {
                cropDebounceTimer = setTimeout(fetchCroppedData, 50);
            } else {
                fetchCroppedData();
            }
        }

        async function fetchCroppedData() {
            if (!currentPatchData) return;

            if (isCropFull()) {
                updateThreeFaceTextures(currentPatchData.face_images, currentPatchData.face_images_rgba);
                return;
            }

            try {
                const res = await fetch('/api/patch/crop', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        z_min: cropState.z_min,
                        z_max: cropState.z_max,
                        y_min: cropState.y_min,
                        y_max: cropState.y_max,
                        x_min: cropState.x_min,
                        x_max: cropState.x_max
                    })
                });
                const data = await res.json();
                if (!res.ok) return;

                updateThreeFaceTextures(data.face_images, data.face_images_rgba);
            } catch (err) {
                console.error("Error fetching cropped view data:", err);
            }
        }

        // =========================================================================
        // Three.js 3D Viewport Implementation
        // =========================================================================

        function initThreeJS() {
            const container = document.getElementById('three-container');
            const width = container.clientWidth || 600;
            const height = container.clientHeight || 450;

            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(130, 110, 160);

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.localClippingEnabled = true;
            renderer.setSize(width, height);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.target.set(48, 48, 48);

            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();

            // Lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
            scene.add(ambientLight);
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
            dirLight.position.set(100, 150, 100);
            scene.add(dirLight);

            // 1. Global Plain Square Wireframe Box (Outer 96³ reference - NO diagonal triangle splitting lines)
            const globalBoxGeo = new THREE.BoxGeometry(96, 96, 96);
            const edgesGeo = new THREE.EdgesGeometry(globalBoxGeo);
            const globalWireMat = new THREE.LineBasicMaterial({
                color: 0x3d4b6e,
                transparent: true,
                opacity: 0.65
            });
            globalWireMesh = new THREE.LineSegments(edgesGeo, globalWireMat);
            globalWireMesh.position.set(48, 48, 48);
            scene.add(globalWireMesh);

            createThreeFacePlanes();

            seedPinsGroup = new THREE.Group();
            scene.add(seedPinsGroup);

            curves3DGroup = new THREE.Group();
            scene.add(curves3DGroup);

            build3DAxes();

            // Robust Click listener for 3D Face Seed Placement
            let downTime = 0;
            let downPos = { x: 0, y: 0 };
            renderer.domElement.addEventListener('pointerdown', (e) => {
                downTime = Date.now();
                downPos = { x: e.clientX, y: e.clientY };
            });
            renderer.domElement.addEventListener('pointerup', (e) => {
                const dist = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);
                const duration = Date.now() - downTime;
                if (dist < 10 && duration < 400) {
                    handleThreeFaceClick(e);
                }
            });

            window.addEventListener('resize', onWindowResize);
            if (window.ResizeObserver) {
                const ro = new ResizeObserver(() => onWindowResize());
                ro.observe(container);
            }

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        }

        // =========================================================================
        // 3D Scene Coordinate Arrows (Anchored directly next to the 3D Big Cube)
        // =========================================================================
        function createAxisSprite(labelText, colorHex) {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');

            const colorHexStr = '#' + colorHex.toString(16).padStart(6, '0');

            ctx.beginPath();
            ctx.arc(64, 64, 52, 0, Math.PI * 2);
            ctx.fillStyle = colorHexStr;
            ctx.fill();
            ctx.lineWidth = 10;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(64, 64, 44, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
            ctx.fill();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 54px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(labelText, 64, 66);

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({
                map: texture,
                transparent: true,
                depthTest: false
            });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(7.5, 7.5, 1);
            return sprite;
        }

        function createAxisArrowMesh(dir, length, colorHex, labelText) {
            const group = new THREE.Group();
            const shaftRadius = 1.3;
            const headRadius = 2.9;
            const headLength = 7.5;
            const shaftLength = Math.max(1, length - headLength);

            const shaftGeo = new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 16);
            const shaftMat = new THREE.MeshStandardMaterial({
                color: colorHex,
                emissive: colorHex,
                emissiveIntensity: 0.75,
                roughness: 0.25
            });
            const shaftMesh = new THREE.Mesh(shaftGeo, shaftMat);
            shaftMesh.position.y = shaftLength / 2;

            const headGeo = new THREE.ConeGeometry(headRadius, headLength, 16);
            const headMat = new THREE.MeshStandardMaterial({
                color: colorHex,
                emissive: colorHex,
                emissiveIntensity: 0.95,
                roughness: 0.2
            });
            const headMesh = new THREE.Mesh(headGeo, headMat);
            headMesh.position.y = shaftLength + headLength / 2;

            const arrowObj = new THREE.Group();
            arrowObj.add(shaftMesh);
            arrowObj.add(headMesh);

            const normDir = dir.clone().normalize();
            const defaultDir = new THREE.Vector3(0, 1, 0);
            const quat = new THREE.Quaternion().setFromUnitVectors(defaultDir, normDir);
            arrowObj.applyQuaternion(quat);

            group.add(arrowObj);

            const sprite = createAxisSprite(labelText, colorHex);
            const labelOffset = normDir.clone().multiplyScalar(length + 5.5);
            sprite.position.copy(labelOffset);
            group.add(sprite);

            return group;
        }

        function build3DAxes() {
            if (axes3DGroup) {
                scene.remove(axes3DGroup);
            }
            axes3DGroup = new THREE.Group();
            const baseOffset = new THREE.Vector3(-8, -8, -8);

            // Origin Sphere at (-8, -8, -8)
            const originGeo = new THREE.SphereGeometry(2.2, 16, 16);
            const originMat = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                emissive: 0x888888,
                emissiveIntensity: 0.6,
                roughness: 0.2
            });
            const originMesh = new THREE.Mesh(originGeo, originMat);
            originMesh.position.copy(baseOffset);
            axes3DGroup.add(originMesh);

            // +X Arrow (Width, Red)
            const arrowX = createAxisArrowMesh(new THREE.Vector3(1, 0, 0), 32.0, 0xff3b30, 'X');
            arrowX.position.copy(baseOffset);
            axes3DGroup.add(arrowX);

            // +Y Arrow (Height, Green)
            const arrowY = createAxisArrowMesh(new THREE.Vector3(0, 1, 0), 32.0, 0x30d158, 'Y');
            arrowY.position.copy(baseOffset);
            axes3DGroup.add(arrowY);

            // +Z Arrow (Depth, Cyan)
            const arrowZ = createAxisArrowMesh(new THREE.Vector3(0, 0, 1), 32.0, 0x00e5ff, 'Z');
            arrowZ.position.copy(baseOffset);
            axes3DGroup.add(arrowZ);

            axes3DGroup.visible = isAxesVisible;
            scene.add(axes3DGroup);
            updateThreeAxesPosition();
        }

        function toggleAxesVisibility(visible) {
            isAxesVisible = !!visible;
            if (axes3DGroup) {
                axes3DGroup.visible = isAxesVisible;
            }
        }

        function updateThreeAxesPosition() {
            if (!axes3DGroup) return;
            axes3DGroup.position.set(cropState.x_min, cropState.y_min, cropState.z_min);
        }

        function onWindowResize() {
            const container = document.getElementById('three-container');
            if (!container || !renderer || !camera) return;
            const width = container.clientWidth;
            const height = container.clientHeight;
            if (width === 0 || height === 0) return;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        }

        function createQuadGeometry(v0, v1, v2, v3) {
            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array([
                ...v0, ...v1, ...v2,
                ...v0, ...v2, ...v3
            ]);
            const uvs = new Float32Array([
                0, 0,  1, 0,  1, 1,
                0, 0,  1, 1,  0, 1
            ]);
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
            geo.computeVertexNormals();
            geo.computeBoundingSphere();
            geo.computeBoundingBox();
            return geo;
        }

        function getFaceQuadVertices(faceName, z0, z1, y0, y1, x0, x1) {
            const xMax = x1 + 1;
            const yMax = y1 + 1;
            const zMax = z1 + 1;

            if (faceName === 'z_min') {
                return { v0: [x0, y0, z0], v1: [xMax, y0, z0], v2: [xMax, yMax, z0], v3: [x0, yMax, z0] };
            } else if (faceName === 'z_max') {
                return { v0: [x0, y0, zMax], v1: [xMax, y0, zMax], v2: [xMax, yMax, zMax], v3: [x0, yMax, zMax] };
            } else if (faceName === 'y_min') {
                return { v0: [x0, y0, z0], v1: [xMax, y0, z0], v2: [xMax, y0, zMax], v3: [x0, y0, zMax] };
            } else if (faceName === 'y_max') {
                return { v0: [x0, yMax, z0], v1: [xMax, yMax, z0], v2: [xMax, yMax, zMax], v3: [x0, yMax, zMax] };
            } else if (faceName === 'x_min') {
                return { v0: [x0, y0, z0], v1: [x0, yMax, z0], v2: [x0, yMax, zMax], v3: [x0, y0, zMax] };
            } else { // x_max
                return { v0: [xMax, y0, z0], v1: [xMax, yMax, z0], v2: [xMax, yMax, zMax], v3: [xMax, y0, zMax] };
            }
        }

        function createThreeFacePlanes() {
            faceMeshes = [];
            const faceNames = ['z_min', 'z_max', 'y_min', 'y_max', 'x_min', 'x_max'];

            faceNames.forEach(name => {
                const cvs = document.createElement('canvas');
                cvs.width = 96;
                cvs.height = 96;
                offscreenCanvases[name] = cvs;

                const tex = new THREE.CanvasTexture(cvs);
                tex.magFilter = THREE.NearestFilter;
                tex.minFilter = THREE.NearestFilter;
                tex.flipY = false;
                faceTextures[name] = tex;

                const quad = getFaceQuadVertices(name, 0, 95, 0, 95, 0, 95);
                const geo = createQuadGeometry(quad.v0, quad.v1, quad.v2, quad.v3);
                const mat = new THREE.MeshBasicMaterial({
                    map: tex,
                    transparent: true,
                    opacity: 0.95,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const mesh = new THREE.Mesh(geo, mat);
                mesh.userData = { faceName: name };
                scene.add(mesh);
                faceMeshes.push(mesh);
            });
        }

        function updateThreeCropBoxGeometry(z0, z1, y0, y1, x0, x1) {
            faceMeshes.forEach(mesh => {
                const name = mesh.userData.faceName;
                const quad = getFaceQuadVertices(name, z0, z1, y0, y1, x0, x1);
                if (mesh.geometry) mesh.geometry.dispose();
                mesh.geometry = createQuadGeometry(quad.v0, quad.v1, quad.v2, quad.v3);
            });

            if (cropWireMesh) {
                scene.remove(cropWireMesh);
                if (cropWireMesh.geometry) cropWireMesh.geometry.dispose();
                if (cropWireMesh.material) cropWireMesh.material.dispose();
                cropWireMesh = null;
            }

            if (!isCropFull()) {
                const szX = x1 - x0 + 1;
                const szY = y1 - y0 + 1;
                const szZ = z1 - z0 + 1;
                const cX = x0 + szX / 2;
                const cY = y0 + szY / 2;
                const cZ = z0 + szZ / 2;
                const boxGeo = new THREE.BoxGeometry(szX, szY, szZ);
                const edgesGeo = new THREE.EdgesGeometry(boxGeo);
                const mat = new THREE.LineBasicMaterial({
                    color: 0x00e5ff,
                    transparent: true,
                    opacity: 0.8
                });
                cropWireMesh = new THREE.LineSegments(edgesGeo, mat);
                cropWireMesh.position.set(cX, cY, cZ);
                scene.add(cropWireMesh);
            }

            updateThreeCurvesClipping();
            updateThreeAxesPosition();
        }

        function updateThreeFaceTextures(faceImagesGrayscale, faceImagesRGBA) {
            const imagesToUse = faceImagesRGBA || faceImagesGrayscale;
            if (!imagesToUse) return;

            Object.entries(imagesToUse).forEach(([faceName, b64Url]) => {
                if (!b64Url) return;
                const img = new Image();
                img.onload = () => {
                    const cvs = offscreenCanvases[faceName];
                    if (cvs) {
                        cvs.width = img.naturalWidth || img.width;
                        cvs.height = img.naturalHeight || img.height;
                        const ctx = cvs.getContext('2d');
                        ctx.clearRect(0, 0, cvs.width, cvs.height);
                        ctx.drawImage(img, 0, 0);
                        if (faceTextures[faceName]) {
                            faceTextures[faceName].needsUpdate = true;
                        }
                    }
                };
                img.src = b64Url;
            });
        }

        function onPointDensityInput(val) {
            currentPointDensity = parseInt(val) || 0;
            const tag = document.getElementById('val-point-density');
            if (tag) tag.innerText = `${currentPointDensity}%`;
            updateThreePointCloud();
        }

        function handleThreeFaceClick(event) {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);

            // Raycast strictly against the 3D box faces (closest front-facing intersection)
            const intersects = raycaster.intersectObjects(faceMeshes, false);

            if (intersects.length > 0) {
                const hit = intersects[0];
                const faceName = hit.object.userData.faceName;
                const pt = hit.point;

                const z0 = cropState.z_min;
                const z1 = cropState.z_max;
                const y0 = cropState.y_min;
                const y1 = cropState.y_max;
                const x0 = cropState.x_min;
                const x1 = cropState.x_max;

                let z = Math.max(z0, Math.min(z1, Math.round(pt.z)));
                let y = Math.max(y0, Math.min(y1, Math.round(pt.y)));
                let x = Math.max(x0, Math.min(x1, Math.round(pt.x)));

                if (faceName === 'z_min') z = z0;
                else if (faceName === 'z_max') z = z1;
                else if (faceName === 'y_min') y = y0;
                else if (faceName === 'y_max') y = y1;
                else if (faceName === 'x_min') x = x0;
                else if (faceName === 'x_max') x = x1;

                let u = y, v = x;
                if (faceName === 'z_min' || faceName === 'z_max') { u = y; v = x; }
                else if (faceName === 'y_min' || faceName === 'y_max') { u = z; v = x; }
                else { u = z; v = y; }

                // Strict check: Must be on the exact same face AND within 4 voxels in 3D
                // This completely prevents deleting points on opposite faces or other sides of the cube
                const existingIdx = seedsList.findIndex(s => {
                    if (s.face !== faceName) return false;
                    const dz = s.pos3d[0] - z;
                    const dy = s.pos3d[1] - y;
                    const dx = s.pos3d[2] - x;
                    return Math.hypot(dz, dy, dx) < 4.0;
                });

                if (existingIdx >= 0) {
                    seedsList.splice(existingIdx, 1);
                } else {
                    seedsList.push({
                        face: faceName,
                        u: u,
                        v: v,
                        pos3d: [z, y, x],
                        fiber_id: activeFiberId
                    });

                    // QoL: Only auto-advance if we just completed the 2nd point for the HIGHEST current Fiber ID
                    const currentFiberBoundarySeeds = seedsList.filter(s => s.fiber_id === activeFiberId && !s.is_waypoint && s.face !== 'waypoint');
                    if (currentFiberBoundarySeeds.length === 2) {
                        const allFiberIds = seedsList.map(s => s.fiber_id);
                        const maxCurrentId = allFiberIds.length > 0 ? Math.max(...allFiberIds) : 1;
                        if (activeFiberId >= maxCurrentId) {
                            setActiveId(activeFiberId + 1);
                        }
                    }
                }

                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            }
        }

        function createSeedSprite(fiberId, colorHexStr) {
            const canvas = document.createElement('canvas');
            canvas.width = 128;
            canvas.height = 128;
            const ctx = canvas.getContext('2d');

            ctx.beginPath();
            ctx.arc(64, 64, 54, 0, Math.PI * 2);
            ctx.fillStyle = colorHexStr;
            ctx.fill();
            ctx.lineWidth = 10;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(64, 64, 46, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.40)';
            ctx.fill();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 44px monospace';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(fiberId), 64, 66);

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMat = new THREE.SpriteMaterial({
                map: texture,
                transparent: true,
                depthTest: false
            });
            const sprite = new THREE.Sprite(spriteMat);
            sprite.scale.set(6, 6, 1);
            return sprite;
        }

        function updateThreeSeeds() {
            while (seedPinsGroup.children.length > 0) {
                const child = seedPinsGroup.children[0];
                seedPinsGroup.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (child.material.map) child.material.map.dispose();
                    child.material.dispose();
                }
            }

            const z0 = cropState.z_min;
            const z1 = cropState.z_max;
            const y0 = cropState.y_min;
            const y1 = cropState.y_max;
            const x0 = cropState.x_min;
            const x1 = cropState.x_max;

            seedsList.forEach(s => {
                const z = s.pos3d[0];
                const y = s.pos3d[1];
                const x = s.pos3d[2];

                const isInside = (z >= z0 && z <= z1 && y >= y0 && y <= y1 && x >= x0 && x <= x1);
                if (!isInside) {
                    return;
                }

                const colorHexStr = COLOR_PALETTE[(s.fiber_id - 1) % COLOR_PALETTE.length];
                const colorHex = parseInt(colorHexStr.replace('#', '0x'));
                const isWp = s.is_waypoint || s.face === 'waypoint' || s.face === 'internal';

                const pinGeo = isWp ? new THREE.OctahedronGeometry(2.8, 0) : new THREE.SphereGeometry(2.0, 16, 16);
                const pinMat = new THREE.MeshStandardMaterial({
                    color: isWp ? 0xffffff : colorHex,
                    emissive: colorHex,
                    emissiveIntensity: isWp ? 1.4 : 0.85,
                    roughness: 0.2,
                    transparent: false,
                    opacity: 1.0
                });
                const pinMesh = new THREE.Mesh(pinGeo, pinMat);
                pinMesh.position.set(x, y, z);
                seedPinsGroup.add(pinMesh);

                const sprite = createSeedSprite(s.fiber_id, colorHexStr);
                sprite.position.set(x, y, z);
                seedPinsGroup.add(sprite);
            });
        }

        function renderThree3DCurves(curvesDict) {
            latestResolvedCurves = curvesDict || {};

            while (curves3DGroup.children.length > 0) {
                const child = curves3DGroup.children[0];
                curves3DGroup.remove(child);
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            }

            if (!curvesDict) return;

            Object.entries(curvesDict).forEach(([fidStr, pts]) => {
                const fid = parseInt(fidStr);
                const colorHex = parseInt(COLOR_PALETTE[(fid - 1) % COLOR_PALETTE.length].replace('#', '0x'));

                const vectors = pts.map(p => new THREE.Vector3(p[2], p[1], p[0]));
                if (vectors.length >= 2) {
                    const curve = new THREE.CatmullRomCurve3(vectors);
                    const tubeGeo = new THREE.TubeGeometry(curve, Math.max(20, pts.length), 1.3, 8, false);
                    const tubeMat = new THREE.MeshStandardMaterial({
                        color: colorHex,
                        emissive: colorHex,
                        emissiveIntensity: 0.75,
                        roughness: 0.3,
                        clippingPlanes: getCropClippingPlanes(),
                        clipShadows: true
                    });
                    const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
                    tubeMesh.userData = { fiberId: fid };
                    curves3DGroup.add(tubeMesh);
                }
            });
        }

        function updateThreePointCloud(points) {
            if (points) rawPointCloudFull = points;

            if (pointCloudMesh) {
                scene.remove(pointCloudMesh);
                if (pointCloudMesh.geometry) pointCloudMesh.geometry.dispose();
                if (pointCloudMesh.material) pointCloudMesh.material.dispose();
                pointCloudMesh = null;
            }

            if (!rawPointCloudFull || rawPointCloudFull.length === 0 || currentPointDensity <= 0) return;

            const z0 = cropState.z_min;
            const z1 = cropState.z_max;
            const y0 = cropState.y_min;
            const y1 = cropState.y_max;
            const x0 = cropState.x_min;
            const x1 = cropState.x_max;

            const inCrop = rawPointCloudFull.filter(p => 
                p[0] >= z0 && p[0] <= z1 &&
                p[1] >= y0 && p[1] <= y1 &&
                p[2] >= x0 && p[2] <= x1
            );

            if (inCrop.length === 0) return;

            // Continuous linear subsampling: scales point count strictly with currentPointDensity (1% to 100%)
            const fraction = Math.max(0.01, Math.min(1.0, currentPointDensity / 100.0));
            const targetCount = Math.max(1, Math.round(inCrop.length * fraction));

            const positions = new Float32Array(targetCount * 3);
            const stride = inCrop.length / targetCount;
            for (let i = 0; i < targetCount; i++) {
                const srcIdx = Math.min(inCrop.length - 1, Math.floor(i * stride));
                const pt = inCrop[srcIdx];
                positions[i * 3] = pt[2];     // X = x
                positions[i * 3 + 1] = pt[1]; // Y = y
                positions[i * 3 + 2] = pt[0]; // Z = z
            }

            const geo = new THREE.BufferGeometry();
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

            const mat = new THREE.PointsMaterial({
                color: 0x00e5ff,
                size: 2.2,
                transparent: true,
                opacity: Math.min(0.85, 0.35 + fraction * 0.35),
                blending: THREE.AdditiveBlending,
                depthWrite: false
            });
            pointCloudMesh = new THREE.Points(geo, mat);
            pointCloudMesh.visible = (currentPointDensity > 0);
            scene.add(pointCloudMesh);
        }

        let currentlyLoadedPatchIndex = null;

        async function refreshSavedPatchesList() {
            try {
                const res = await fetch('/api/patches/list?t=' + Date.now());
                const patches = await res.json();
                const sel = document.getElementById('saved-patches-select');
                if (!sel) return;

                sel.innerHTML = `<option value="">📂 Load Saved Cube (${patches.length})...</option>`;
                patches.forEach(p => {
                    const opt = document.createElement('option');
                    opt.value = p.index;
                    opt.innerText = `#${String(p.index).padStart(4, '0')} (${p.num_fibers} Fibers, ${(p.density * 100).toFixed(1)}%)`;
                    if (currentlyLoadedPatchIndex === p.index) opt.selected = true;
                    sel.appendChild(opt);
                });
            } catch (e) {
                console.error("Failed to load saved patches list:", e);
            }
        }

        function updateCoordinateInputs(origin, maxOrigin) {
            if (!origin) return;
            const zInput = document.getElementById('load-coord-z');
            const yInput = document.getElementById('load-coord-y');
            const xInput = document.getElementById('load-coord-x');
            if (zInput) zInput.value = origin[0];
            if (yInput) yInput.value = origin[1];
            if (xInput) xInput.value = origin[2];
            if (maxOrigin) {
                if (zInput) zInput.max = maxOrigin[0];
                if (yInput) yInput.max = maxOrigin[1];
                if (xInput) xInput.max = maxOrigin[2];
            }
        }

        function updateSidebarInfo(origin, density, curatedTotal, statusText) {
            const orgVal = document.getElementById('info-origin-val');
            const denVal = document.getElementById('info-density-val');
            const curVal = document.getElementById('info-curated-val');
            const stVal = document.getElementById('info-status-val');

            if (orgVal && origin) orgVal.innerText = `(Z:${origin[0]}, Y:${origin[1]}, X:${origin[2]})`;
            if (denVal && density !== undefined) denVal.innerText = `${(density * 100).toFixed(1)}%`;
            if (curVal && curatedTotal !== undefined) curVal.innerText = `${curatedTotal} cubes`;
            if (stVal && statusText) stVal.innerText = statusText;
        }

        function setPerfStatus(text) {
            const el = document.getElementById('perf-badge');
            if (el) el.innerText = text;
        }

        function handleCoordKey(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                loadManualCoordinates();
            }
        }

        async function loadManualCoordinates(zVal, yVal, xVal) {
            try {
                let z = zVal !== undefined ? parseInt(zVal) : parseInt(document.getElementById('load-coord-z').value || 0);
                let y = yVal !== undefined ? parseInt(yVal) : parseInt(document.getElementById('load-coord-y').value || 0);
                let x = xVal !== undefined ? parseInt(xVal) : parseInt(document.getElementById('load-coord-x').value || 0);

                if (isNaN(z)) z = 0;
                if (isNaN(y)) y = 0;
                if (isNaN(x)) x = 0;

                currentlyLoadedPatchIndex = null;
                const sel = document.getElementById('saved-patches-select');
                if (sel) sel.value = '';

                setPerfStatus(`⏳ Loading 96³ Subvolume at (${z}, ${y}, ${x})...`);
                const res = await fetch('/api/patch/coords', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ z, y, x })
                });
                const data = await res.json();
                if (!res.ok || data.error) {
                    setPerfStatus('❌ ' + (data.error || "Failed to load coordinates"));
                    alert(data.error || "Failed to load coordinates");
                    return;
                }

                currentPatchData = data;
                seedsList = data.loaded_seeds || [];

                updateCoordinateInputs(data.origin, data.max_origin);

                const srcName = data.source_info ? data.source_info.name : '';
                const isSeg = (data.is_segmented_source || (data.source_info && data.source_info.type === 'segmented_instance'));

                const srcBadge = document.getElementById('source-name');
                if (srcBadge && srcName) {
                    srcBadge.innerText = `${srcName} [${isSeg ? 'Segmented' : 'Raw'}]`;
                    srcBadge.style.color = isSeg ? 'var(--accent-lime)' : 'var(--accent-cyan)';
                }

                const origBadge = document.getElementById('origin-badge');
                if (origBadge && data.origin) origBadge.innerHTML = `Origin: <strong>(Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})</strong>`;
                const densBadge = document.getElementById('density-badge');
                if (densBadge && data.density !== undefined) densBadge.innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
                const curatedBadge = document.getElementById('curated-count');
                if (curatedBadge && data.curated_total !== undefined) curatedBadge.innerText = data.curated_total;

                if (isSeg && data.num_fibers > 0) {
                    setPerfStatus(`🟢 Loaded ${data.num_fibers} Fibers from Segmented TIFF! Inspect & Fix Wiring.`);
                    updateSidebarInfo(data.origin, data.density, data.curated_total, `Segmented: ${data.num_fibers} Fibers`);
                } else {
                    setPerfStatus(`📍 Loaded 96³ Subvolume at Origin (Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})`);
                    updateSidebarInfo(data.origin, data.density, data.curated_total, 'Ready for Annotation');
                }

                rawPointCloudFull = data.point_cloud || [];
                applyCropView(false);

                activeFiberId = 1;
                if (data.curves_3d && Object.keys(data.curves_3d).length > 0) {
                    renderThree3DCurves(data.curves_3d);
                } else {
                    renderThree3DCurves(null);
                }

                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            } catch (err) {
                console.error("Failed to load coordinates:", err);
                setPerfStatus('❌ Error loading coordinates: ' + (err.message || err));
            }
        }

        async function loadSavedPatch(patchId) {
            if (!patchId) return;
            try {
                setPerfStatus(`⏳ Loading Saved Sample #${String(patchId).padStart(4, '0')}...`);
                const res = await fetch('/api/patch/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ patch_id: parseInt(patchId) })
                });
                const data = await res.json();
                if (!res.ok || data.error) {
                    setPerfStatus('❌ ' + (data.error || "Failed to load patch"));
                    alert(data.error || "Failed to load patch");
                    return;
                }

                currentlyLoadedPatchIndex = parseInt(patchId);
                currentPatchData = data;
                seedsList = data.loaded_seeds || [];

                updateCoordinateInputs(data.origin, data.max_origin);

                const origBadge = document.getElementById('origin-badge');
                if (origBadge && data.origin) origBadge.innerHTML = `Sample: <strong>#${String(patchId).padStart(4, '0')}</strong> (Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})`;
                const densBadge = document.getElementById('density-badge');
                if (densBadge && data.density !== undefined) densBadge.innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
                const curatedBadge = document.getElementById('curated-count');
                if (curatedBadge && data.curated_total !== undefined) curatedBadge.innerText = data.curated_total;
                setPerfStatus(`⚡ Loaded Saved Sample #${String(patchId).padStart(4, '0')} with ${data.num_fibers} Fibers!`);

                updateSidebarInfo(data.origin, data.density, data.curated_total, `Loaded #${String(patchId).padStart(4, '0')} (${data.num_fibers} Fibers)`);

                rawPointCloudFull = data.point_cloud || [];
                applyCropView(false);

                renderThree3DCurves(data.curves_3d);
                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            } catch (err) {
                console.error("Failed to load saved patch:", err);
                setPerfStatus('❌ Error loading saved patch: ' + (err.message || err));
            }
        }

        async function fetchNewRandomPatch() {
            try {
                currentlyLoadedPatchIndex = null;
                const sel = document.getElementById('saved-patches-select');
                if (sel) sel.value = '';

                setPerfStatus('⏳ Loading 96³ Cube...');
                const res = await fetch('/api/patch/new?t=' + Date.now(), { cache: 'no-store' });
                const data = await res.json();
                if (!res.ok || data.error) {
                    console.error("API error fetching patch:", data.error);
                    setPerfStatus('❌ ' + (data.error || 'Error fetching patch'));
                    return;
                }

                currentPatchData = data;
                seedsList = data.loaded_seeds || [];

                updateCoordinateInputs(data.origin, data.max_origin);

                const srcName = data.source_info ? data.source_info.name : '';
                const isSeg = (data.is_segmented_source || (data.source_info && data.source_info.type === 'segmented_instance'));

                const srcBadge = document.getElementById('source-name');
                if (srcBadge && srcName) {
                    srcBadge.innerText = `${srcName} [${isSeg ? 'Segmented' : 'Raw'}]`;
                    srcBadge.style.color = isSeg ? 'var(--accent-lime)' : 'var(--accent-cyan)';
                }

                const origBadgeRand = document.getElementById('origin-badge');
                if (origBadgeRand && data.origin) origBadgeRand.innerHTML = `Origin: <strong>(Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})</strong>`;
                const densBadgeRand = document.getElementById('density-badge');
                if (densBadgeRand && data.density !== undefined) densBadgeRand.innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
                const curatedBadgeRand = document.getElementById('curated-count');
                if (curatedBadgeRand && data.curated_total !== undefined) curatedBadgeRand.innerText = data.curated_total;

                if (isSeg && data.num_fibers > 0) {
                    setPerfStatus(`🟢 Loaded ${data.num_fibers} Fibers from Segmented TIFF! Inspect & Fix Wiring.`);
                    updateSidebarInfo(data.origin, data.density, data.curated_total, `Segmented: ${data.num_fibers} Fibers`);
                } else {
                    setPerfStatus('Ready for Seed Placement');
                    updateSidebarInfo(data.origin, data.density, data.curated_total, 'Ready for Annotation');
                }

                rawPointCloudFull = data.point_cloud || [];
                applyCropView(false);

                activeFiberId = 1;
                if (data.curves_3d && Object.keys(data.curves_3d).length > 0) {
                    renderThree3DCurves(data.curves_3d);
                } else {
                    renderThree3DCurves(null);
                }

                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            } catch (err) {
                console.error("Failed to fetch patch:", err);
                setPerfStatus('❌ Error fetching patch: ' + (err.message || err));
            }
        }

        function clearAllSeeds() {
            seedsList = [];
            activeFiberId = 1;
            updateThreeSeeds();
            updateSeedsDock();
            renderThree3DCurves(null);
            initIdSelector();
        }

        function updateSeedsDock() {
            const container = document.getElementById('seeds-chip-container');
            const countTag = document.getElementById('seeds-count-tag');
            container.innerHTML = '';

            const grouped = {};
            seedsList.forEach(s => {
                if (!grouped[s.fiber_id]) grouped[s.fiber_id] = [];
                grouped[s.fiber_id].push(s);
            });

            const sortedIds = Object.keys(grouped).map(Number).sort((a, b) => a - b);
            countTag.innerText = `${seedsList.length} Seeds (${sortedIds.length} Fibers)`;

            if (sortedIds.length === 0) {
                container.innerHTML = `<div style="color: var(--text-muted); font-size: 11px;">No seeds placed. Click on 3D Box faces to place fiber seeds!</div>`;
                return;
            }

            sortedIds.forEach(id => {
                const list = grouped[id];
                const chip = document.createElement('div');
                chip.className = 'fiber-chip';
                
                const color = COLOR_PALETTE[(id - 1) % COLOR_PALETTE.length];
                const isMulti = list.length > 2;
                const isPair = list.length === 2;
                const statusClass = (isPair || isMulti) ? 'connected' : 'single';
                const statusText = isMulti ? `GUIDED (${list.length} pts)` : (isPair ? 'PAIR' : 'SINGLE');

                const coordsHtml = list.map((s, idx) => {
                    const isWp = s.is_waypoint || s.face === 'waypoint' || s.face === 'internal';
                    if (isWp) {
                        return `<span class="chip-wp-tag" title="Waypoint (Z=${s.pos3d[0]}, Y=${s.pos3d[1]}, X=${s.pos3d[2]})">
                            <span>W:[${s.pos3d[0]},${s.pos3d[1]},${s.pos3d[2]}]</span>
                            <button style="background:none; border:none; color:var(--accent-magenta); cursor:pointer; font-size:9px; padding:0;" onclick="deleteWaypoint(${id}, ${s.pos3d[0]}, ${s.pos3d[1]}, ${s.pos3d[2]})" title="Delete this waypoint">✕</button>
                        </span>`;
                    }
                    return `<span style="font-family:'JetBrains Mono',monospace;">[${s.pos3d[0]},${s.pos3d[1]},${s.pos3d[2]}]</span>`;
                }).join('<span style="color:var(--accent-cyan); font-size:10px; margin:0 2px;">➜</span>');

                chip.innerHTML = `
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="chip-color-dot" style="background: ${color}; cursor:pointer;" onclick="setActiveId(${id})" title="Click to make Fiber #${id} active"></span>
                        <strong style="cursor:pointer;" onclick="setActiveId(${id})" title="Click to make Fiber #${id} active">Fiber ${id}</strong>
                        <span class="chip-status ${statusClass}">${statusText}</span>
                    </div>
                    <div style="display:inline-flex; align-items:center; flex-wrap:wrap; gap:4px;">${coordsHtml}</div>
                    <button class="chip-del-btn" onclick="deleteFiberId(${id})" title="Delete entire Fiber #${id}">✕</button>
                `;
                container.appendChild(chip);
            });
        }

        function deleteFiberId(id) {
            seedsList = seedsList.filter(s => s.fiber_id !== id);
            updateThreeSeeds();
            updateSeedsDock();
            initIdSelector();
        }

        async function resolveConnections() {
            if (seedsList.length === 0) {
                alert("Please place at least one pair of seeds or a singleton seed first!");
                return;
            }

            try {
                document.getElementById('perf-badge').innerText = '⏳ Resolving Geodesic Paths...';
                const cropBounds = isCropFull() ? null : [cropState.z_min, cropState.z_max, cropState.y_min, cropState.y_max, cropState.x_min, cropState.x_max];

                const res = await fetch('/api/resolve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        seeds: seedsList,
                        crop_bounds: cropBounds
                    })
                });
                const data = await res.json();

                if (data.success) {
                    document.getElementById('perf-badge').innerText = `⚡ Resolved ${data.num_fibers} Fibers in ${data.resolve_time_ms} ms!`;
                    updateSidebarInfo(undefined, undefined, undefined, `Resolved ${data.num_fibers} Fibers (${data.resolve_time_ms} ms)`);
                    renderThree3DCurves(data.curves_3d);
                }
            } catch (err) {
                console.error("Resolution failed:", err);
                document.getElementById('perf-badge').innerText = '❌ Resolution error';
            }
        }

        async function saveAndNext() {
            try {
                const res = await fetch('/api/save', { 
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ overwrite: (currentlyLoadedPatchIndex !== null) })
                });
                const data = await res.json();
                if (data.success) {
                    document.getElementById('curated-count').innerText = data.curated_total;
                    await refreshSavedPatchesList();
                    fetchNewRandomPatch();
                } else {
                    alert(data.error || "Save failed. Make sure to Resolve Connections first!");
                }
            } catch (err) {
                console.error("Save failed:", err);
            }
        }

        // =====================================================================
        // Open File Modal & Custom Segmented Volume Loading Logic
        // =====================================================================

        async function openFileModal() {
            const modal = document.getElementById('file-modal');
            if (!modal) return;
            modal.style.display = 'flex';

            const grid = document.getElementById('modal-preset-pills');
            if (grid) {
                grid.innerHTML = '<div style="color: var(--text-muted); font-size: 11px; padding: 4px;">🔍 Scanning workspace volumes...</div>';
            }

            try {
                const res = await fetch('/api/files/available?t=' + Date.now());
                const data = await res.json();
                if (grid) {
                    grid.innerHTML = '';

                    // Segmented Instances (Top Priority)
                    if (data.segmented && data.segmented.length > 0) {
                        const sec = document.createElement('div');
                        sec.style = "font-size:10px; font-weight:800; color:var(--accent-lime); text-transform:uppercase; margin-bottom:4px;";
                        sec.innerText = "⭐ Segmented Instances (Auto-Extracts Centerlines & Wiring):";
                        grid.appendChild(sec);

                        data.segmented.forEach(f => {
                            const item = document.createElement('div');
                            item.className = 'preset-item';
                            item.innerHTML = `<span class="preset-item-name">${f.path}</span> <span class="preset-tag segmented">Segmented (${f.size_mb} MB)</span>`;
                            item.onclick = () => {
                                document.getElementById('modal-file-path').value = f.path;
                            };
                            grid.appendChild(item);
                        });
                    }

                    // Raw Volumes
                    if (data.raw && data.raw.length > 0) {
                        const sec = document.createElement('div');
                        sec.style = "font-size:10px; font-weight:800; color:var(--accent-cyan); text-transform:uppercase; margin:8px 0 4px 0;";
                        sec.innerText = "🔬 Raw Microscopy Volumes:";
                        grid.appendChild(sec);

                        data.raw.forEach(f => {
                            const item = document.createElement('div');
                            item.className = 'preset-item';
                            item.innerHTML = `<span class="preset-item-name">${f.path}</span> <span class="preset-tag raw">Raw (${f.size_mb} MB)</span>`;
                            item.onclick = () => {
                                document.getElementById('modal-file-path').value = f.path;
                            };
                            grid.appendChild(item);
                        });
                    }
                }
            } catch (e) {
                console.error("Failed to load available files:", e);
                if (grid) grid.innerHTML = '<div style="color: var(--accent-magenta); font-size: 11px;">Error scanning files</div>';
            }
        }

        function closeFileModal() {
            const modal = document.getElementById('file-modal');
            if (modal) modal.style.display = 'none';
        }

        function toggleModalCoordMode() {
            const isManual = document.getElementById('modal-mode-manual').checked;
            const manualDiv = document.getElementById('modal-manual-coords');
            if (manualDiv) manualDiv.style.display = isManual ? 'flex' : 'none';
        }

        // Apply Patch Data & Re-render Viewports
        function applyLoadedPatchData(data) {
            currentPatchData = data;
            currentlyLoadedPatchIndex = null;
            seedsList = data.loaded_seeds || [];

            updateCoordinateInputs(data.origin, data.max_origin);

            const srcName = data.source_info ? data.source_info.name : 'volume';
            const isSeg = (data.is_segmented_source || (data.source_info && data.source_info.type === 'segmented_instance'));

            const srcBadge = document.getElementById('source-name');
            if (srcBadge) {
                srcBadge.innerText = `${srcName} [${isSeg ? 'Segmented' : 'Raw'}]`;
                srcBadge.style.color = isSeg ? 'var(--accent-lime)' : 'var(--accent-cyan)';
            }

            const origBadge = document.getElementById('origin-badge');
            if (origBadge && data.origin) origBadge.innerHTML = `Origin: <strong>(Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})</strong>`;
            const densBadge = document.getElementById('density-badge');
            if (densBadge && data.density !== undefined) densBadge.innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
            const curatedBadge = document.getElementById('curated-count');
            if (curatedBadge && data.curated_total !== undefined) curatedBadge.innerText = data.curated_total;

            if (isSeg && data.num_fibers > 0) {
                setPerfStatus(`🟢 Loaded ${data.num_fibers} Fibers from Segmented TIFF! Inspect & Fix Wiring.`);
                updateSidebarInfo(data.origin, data.density, data.curated_total, `Segmented: ${data.num_fibers} Fibers`);
            } else {
                setPerfStatus(`📂 Opened ${srcName}`);
                updateSidebarInfo(data.origin, data.density, data.curated_total, 'Ready for Annotation');
            }

            rawPointCloudFull = data.point_cloud || [];
            applyCropView(false);

            activeFiberId = 1;
            if (data.curves_3d && Object.keys(data.curves_3d).length > 0) {
                renderThree3DCurves(data.curves_3d);
            } else {
                renderThree3DCurves(null);
            }

            updateThreeSeeds();
            updateSeedsDock();
            initIdSelector();
        }

        async function handleBrowserFileSelect(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const formData = new FormData();
            formData.append('file', file);

            document.getElementById('perf-badge').innerText = `⏳ Uploading and parsing ${file.name}...`;
            try {
                const res = await fetch('/api/volume/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) {
                    alert(data.error || "Failed to upload volume");
                    document.getElementById('perf-badge').innerText = '❌ Upload error';
                    return;
                }
                closeFileModal();
                applyLoadedPatchData(data);
            } catch (err) {
                console.error("Upload error:", err);
                alert("Upload error: " + err.message);
                document.getElementById('perf-badge').innerText = '❌ Upload error';
            }
        }

        // Drag & Drop Volume Loader
        window.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        window.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.match(/\.(tif|tiff|npy)$/i)) {
                    handleBrowserFileSelect({ files: [file] });
                } else {
                    alert("Please drop a 3D .tif, .tiff, or .npy file.");
                }
            }
        });

        async function submitOpenFile() {
            const filePath = document.getElementById('modal-file-path').value.trim();
            if (!filePath) {
                alert("Please specify a file path (.tif, .tiff, .npy)");
                return;
            }
            const rawPath = document.getElementById('modal-raw-path').value.trim();
            const isManual = document.getElementById('modal-mode-manual').checked;

            let z = null, y = null, x = null;
            if (isManual) {
                z = parseInt(document.getElementById('modal-z').value) || 0;
                y = parseInt(document.getElementById('modal-y').value) || 0;
                x = parseInt(document.getElementById('modal-x').value) || 0;
            }

            const btn = document.getElementById('btn-modal-open');
            btn.innerText = "⏳ Loading Volume & Centerlines...";
            btn.disabled = true;

            try {
                const payload = { file_path: filePath };
                if (rawPath) payload.raw_path = rawPath;
                if (z !== null) { payload.z = z; payload.y = y; payload.x = x; }

                const res = await fetch('/api/volume/open', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) {
                    alert(data.error || "Failed to open file");
                    btn.innerText = "🚀 Open & Inspect Patch";
                    btn.disabled = false;
                    return;
                }

                closeFileModal();
                applyLoadedPatchData(data);
            } catch (err) {
                console.error("Failed to open volume file:", err);
                alert("Error opening volume file: " + err.message);
            } finally {
                btn.innerText = "🚀 Open & Inspect Patch";
                btn.disabled = false;
            }
        }

        window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
                if (e.code === 'Enter' && e.target.classList.contains('global-coord-input')) {
                    e.preventDefault();
                    loadManualCoordinates();
                }
                return;
            }

            if (e.code === 'Space') {
                e.preventDefault();
                resolveConnections();
            } else if (e.code === 'Enter') {
                e.preventDefault();
                saveAndNext();
            } else if (e.key === 'r' || e.key === 'R') {
                fetchNewRandomPatch();
            } else if (e.key === 'f' || e.key === 'F') {
                focusActiveFiberCrop();
            } else if (e.key === 'c' || e.key === 'C') {
                resetCrop();
            } else if (e.key >= '1' && e.key <= '9') {
                setActiveId(parseInt(e.key));
            } else if (e.key === '0') {
                setActiveId(10);
            } else if (e.key === ']' || e.key === '+' || e.key === '=') {
                setActiveId(activeFiberId + 1);
            } else if (e.key === '[' || e.key === '-' || e.key === '_') {
                setActiveId(Math.max(1, activeFiberId - 1));
            }
        });

        window.onload = async () => {
            initIdSelector();
            initThreeJS();
            syncCropUI();
            await refreshSavedPatchesList();
            fetchNewRandomPatch();
        };
    </script>
</body>
</html>
"""

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/volume/upload', methods=['POST'])
def api_upload_volume():
    """Uploads a volume file via HTML5 browser file picker or drag-and-drop."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    upload_dir = os.path.join('data', 'fibers_to_segment', 'uploaded_volumes')
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, file.filename).replace('\\', '/')
    file.save(save_path)

    try:
        patch_info = engine.open_volume_file(file_path=save_path)
        return jsonify(patch_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/files/available', methods=['GET'])
def api_available_files():
    try:
        files = engine.list_available_files()
        return jsonify(files)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/volume/open', methods=['POST'])
def api_open_volume():
    data = request.get_json() or {}
    file_path = data.get('file_path')
    raw_path = data.get('raw_path')
    z = data.get('z')
    y = data.get('y')
    x = data.get('x')
    if not file_path:
        return jsonify({'error': 'file_path is required'}), 400
    try:
        z_val = int(z) if z is not None else None
        y_val = int(y) if y is not None else None
        x_val = int(x) if x is not None else None
        patch_info = engine.open_volume_file(file_path=file_path, raw_path=raw_path, z=z_val, y=y_val, x=x_val)
        return jsonify(patch_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/patch/new', methods=['GET'])
def api_new_patch():
    if 'z' in request.args and 'y' in request.args and 'x' in request.args:
        try:
            z = int(request.args.get('z', 0))
            y = int(request.args.get('y', 0))
            x = int(request.args.get('x', 0))
            patch_info = engine.extract_patch_at(z, y, x)
            return jsonify(patch_info)
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    patch_info = engine.extract_random_patch()
    return jsonify(patch_info)

@app.route('/api/patch/coords', methods=['POST', 'GET'])
def api_load_patch_coords():
    if request.method == 'POST':
        data = request.get_json() or {}
        z = int(data.get('z', 0))
        y = int(data.get('y', 0))
        x = int(data.get('x', 0))
    else:
        z = int(request.args.get('z', 0))
        y = int(request.args.get('y', 0))
        x = int(request.args.get('x', 0))
    try:
        patch_info = engine.extract_patch_at(z, y, x)
        return jsonify(patch_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/patch/crop', methods=['POST'])
def api_crop_patch():
    data = request.get_json() or {}
    z_min = int(data.get('z_min', 0))
    z_max = int(data.get('z_max', 95))
    y_min = int(data.get('y_min', 0))
    y_max = int(data.get('y_max', 95))
    x_min = int(data.get('x_min', 0))
    x_max = int(data.get('x_max', 95))
    step = int(data.get('step', 4))
    try:
        crop_data = engine.get_cropped_view_data(z_min, z_max, y_min, y_max, x_min, x_max, step=step)
        return jsonify(crop_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/patches/list', methods=['GET'])
def api_list_patches():
    return jsonify(engine.list_saved_patches())

@app.route('/api/patch/load', methods=['POST'])
def api_load_patch():
    data = request.get_json() or {}
    patch_idx = int(data.get('patch_id', 1))
    try:
        res = engine.load_curated_patch(patch_idx)
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/resolve', methods=['POST'])
def api_resolve():
    data = request.get_json() or {}
    seeds = data.get('seeds', [])
    crop_bounds = data.get('crop_bounds')
    res = engine.resolve_connections(seeds, crop_bounds=crop_bounds)
    return jsonify(res)

@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.get_json() or {}
    overwrite = data.get('overwrite', False)
    try:
        res = engine.save_current_curated_sample(overwrite=overwrite)
        res['success'] = True
        return jsonify(res)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify({
        'curated_total': engine.patch_counter,
        'curated_dir': engine.curated_output_dir,
        'volume_shape': [int(engine.D), int(engine.H), int(engine.W)],
        'max_origin': [int(max(0, engine.D - engine.cube_size)), int(max(0, engine.H - engine.cube_size)), int(max(0, engine.W - engine.cube_size))],
        'cube_size': engine.cube_size
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n" + "=" * 80)
    print(f" FIBER CURATOR RUNNING AT: http://127.0.0.1:{port}")
    print("=" * 80 + "\n")
    app.run(host='127.0.0.1', port=port, debug=False)
