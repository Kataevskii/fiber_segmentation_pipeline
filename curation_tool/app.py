import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from flask import Flask, request, jsonify, render_template_string
from curation_tool.engine import RealDataCurationEngine

app = Flask(__name__)

engine = RealDataCurationEngine(
    raw_volume_path='process_data/COLLAGENCROP_003_0000.tif',
    curated_output_dir='real_train_data/curated_patches',
    cube_size=96
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real Collagen 96³ 3D Box Annotator & Geodesic Resolver</title>
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
            padding: 10px 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
            flex-shrink: 0;
        }

        .logo-group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .app-title {
            font-size: 14px;
            font-weight: 700;
            letter-spacing: -0.3px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .app-title span {
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            background: rgba(0, 229, 255, 0.12);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
        }

        .meta-badges {
            display: flex;
            gap: 8px;
            align-items: center;
        }

        .badge {
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .badge strong {
            color: var(--text-primary);
        }

        .badge.highlight {
            border-color: rgba(0, 230, 118, 0.4);
            background: rgba(0, 230, 118, 0.08);
            color: var(--accent-lime);
        }

        .coord-selector-capsule {
            display: flex;
            align-items: center;
            gap: 5px;
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            padding: 3px 8px;
            border-radius: 6px;
            flex-shrink: 0;
        }

        .coord-selector-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
            white-space: nowrap;
        }

        .coord-field-group {
            display: flex;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            padding: 2px 4px;
            gap: 2px;
        }

        .coord-field-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            font-weight: 800;
            color: var(--text-secondary);
        }

        .coord-field-input {
            width: 42px;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 700;
            text-align: center;
            outline: none;
            -moz-appearance: textfield;
        }

        .coord-field-input::-webkit-outer-spin-button,
        .coord-field-input::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }

        .coord-field-input:focus {
            color: var(--accent-cyan);
        }

        .btn-load-coords {
            background: rgba(0, 229, 255, 0.15);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s ease;
        }

        .btn-load-coords:hover {
            background: var(--accent-cyan);
            color: #051016;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.4);
        }

        .nudge-group {
            display: flex;
            gap: 2px;
            align-items: center;
            border-left: 1px solid var(--border-subtle);
            padding-left: 5px;
            margin-left: 2px;
        }

        .btn-nudge {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            color: var(--text-secondary);
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 9px;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.1s ease;
        }

        .btn-nudge:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            background: var(--bg-surface);
        }

        .header-actions {
            display: flex;
            gap: 8px;
            align-items: center;
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

        /* Workspace Layout - Balanced 55% / 45% Split */
        .workspace {
            display: grid;
            grid-template-columns: minmax(480px, 1.18fr) minmax(400px, 0.82fr);
            height: calc(100vh - 54px);
            gap: 10px;
            padding: 8px 10px;
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

        /* Bottom Floating Seeds Dock */
        .seeds-dock {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 8px 12px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-height: 130px;
            overflow-y: auto;
            flex-shrink: 0;
        }

        .dock-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
        }

        .seeds-chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .fiber-chip {
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            border-radius: 5px;
            padding: 3px 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
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

        /* 3D Waypoint / Midpoint Tool Card */
        .waypoint-card {
            background: var(--bg-card);
            border: 1px solid var(--border-accent);
            border-radius: 6px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .waypoint-row {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }

        .coord-input-group {
            display: flex;
            align-items: center;
            gap: 4px;
            background: var(--bg-surface-raised);
            border: 1px solid var(--border-subtle);
            border-radius: 4px;
            padding: 3px 6px;
        }

        .coord-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: bold;
            color: var(--accent-cyan);
        }

        .coord-num-input {
            width: 44px;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 600;
            text-align: center;
            outline: none;
        }

        .btn-insert-waypoint {
            background: linear-gradient(135deg, #00e5ff 0%, #00b0ff 100%);
            color: #051016;
            border: none;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 11px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.15s ease;
        }

        .btn-insert-waypoint:hover {
            filter: brightness(1.15);
            transform: translateY(-1px);
        }

        .btn-slice-pick {
            background: var(--bg-surface-raised);
            color: var(--text-primary);
            border: 1px solid var(--border-accent);
            border-radius: 4px;
            padding: 5px 8px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        }

        .btn-slice-pick:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        .chip-wp-tag {
            background: rgba(0, 229, 255, 0.15);
            border: 1px dashed var(--accent-cyan);
            color: var(--accent-cyan);
            border-radius: 3px;
            padding: 1px 4px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }

        /* Right Panel: MIPs & Slice Viewer */
        .preview-panel {
            background: var(--bg-surface);
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            overflow-y: auto;
        }

        .panel-title {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
        }

        .mip-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .mip-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .mip-header {
            padding: 3px 6px;
            font-size: 10px;
            font-weight: 600;
            color: var(--text-secondary);
            background: var(--bg-surface-raised);
            border-bottom: 1px solid var(--border-subtle);
        }

        .mip-img {
            width: 100%;
            aspect-ratio: 1 / 1;
            image-rendering: pixelated;
            background: #000;
        }

        .slice-section {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .slice-controls {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .slice-slider {
            flex: 1;
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }

        .slice-views-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
        }

        .slice-view-box {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 2px;
        }

        .slice-label {
            font-size: 10px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .slice-img {
            width: 100%;
            aspect-ratio: 1 / 1;
            image-rendering: pixelated;
            border-radius: 4px;
            background: #000;
            border: 1px solid var(--border-subtle);
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
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="logo-group">
            <div class="app-title">
                COLLAGEN GT CURATOR <span>96³ 3D BOX RESOLVER</span>
            </div>
            <div class="meta-badges">
                <div class="badge" id="origin-badge">Origin: <strong>(Z=0, Y=0, X=0)</strong></div>
                <div class="badge" id="density-badge">Density: <strong>10.2%</strong></div>
                <div class="badge highlight" id="total-badge">💾 Curated: <strong id="curated-count">0</strong></div>
            </div>
        </div>

        <!-- Manual Coordinate Selector -->
        <div class="coord-selector-capsule" title="Manually select raw volume coordinates (Z, Y, X) to extract 96³ cube">
            <span class="coord-selector-title">📍 Origin:</span>
            <div class="coord-field-group" title="Z coordinate (depth). Scroll or type">
                <span class="coord-field-label">Z:</span>
                <input type="number" id="load-coord-z" class="coord-field-input global-coord-input" min="0" max="404" value="0" step="16" onkeydown="handleCoordKey(event)" onwheel="handleCoordWheel(event, 'load-coord-z', 16)">
            </div>
            <div class="coord-field-group" title="Y coordinate (height). Scroll or type">
                <span class="coord-field-label">Y:</span>
                <input type="number" id="load-coord-y" class="coord-field-input global-coord-input" min="0" max="404" value="0" step="16" onkeydown="handleCoordKey(event)" onwheel="handleCoordWheel(event, 'load-coord-y', 16)">
            </div>
            <div class="coord-field-group" title="X coordinate (width). Scroll or type">
                <span class="coord-field-label">X:</span>
                <input type="number" id="load-coord-x" class="coord-field-input global-coord-input" min="0" max="404" value="0" step="16" onkeydown="handleCoordKey(event)" onwheel="handleCoordWheel(event, 'load-coord-x', 16)">
            </div>
            <button class="btn-load-coords" onclick="loadManualCoordinates()" title="Extract 96³ cube at manual coordinates [Enter]">
                Load Coords ↵
            </button>
            <div class="nudge-group" title="Step subvolume along axes by 32 vx">
                <button class="btn-nudge" onclick="nudgeCoords(0, 0, -32)" title="Step X -32">◀X</button>
                <button class="btn-nudge" onclick="nudgeCoords(0, 0, 32)" title="Step X +32">X▶</button>
                <button class="btn-nudge" onclick="nudgeCoords(0, -32, 0)" title="Step Y -32">▲Y</button>
                <button class="btn-nudge" onclick="nudgeCoords(0, 32, 0)" title="Step Y +32">Y▼</button>
                <button class="btn-nudge" onclick="nudgeCoords(-32, 0, 0)" title="Step Z -32">◀Z</button>
                <button class="btn-nudge" onclick="nudgeCoords(32, 0, 0)" title="Step Z +32">Z▶</button>
            </div>
        </div>

        <div class="header-actions">
            <select id="saved-patches-select" class="dropdown-select" onchange="loadSavedPatch(this.value)" title="Load an already-saved cube to inspect or adjust annotations">
                <option value="">📂 Load Saved Cube...</option>
            </select>
            <button class="btn-secondary" onclick="fetchNewRandomPatch()" title="Hotkey: R">
                🎲 Random Cube [R]
            </button>
            <button class="btn-secondary" onclick="autoPopulateCandidates()" title="Auto-detect candidate seeds">
                ✨ Auto-Detect
            </button>
            <button class="btn-secondary" onclick="clearAllSeeds()" title="Clear all seeds">
                🗑️ Clear
            </button>
            <button class="btn-resolve" onclick="resolveConnections()" title="Hotkey: Space">
                ⚡ Resolve Paths [Space]
            </button>
            <button class="btn-save" onclick="saveAndNext()" title="Hotkey: Enter">
                ✅ Save & Next [Enter]
            </button>
        </div>
    </header>

    <!-- Workspace Layout -->
    <div class="workspace">
        
        <div class="three-stage-panel">
            <div class="stage-header">
                <!-- Viewport Toggles -->
                <div class="toggles-cluster">
                    <label class="toggle-pill" title="Transparent zero-voxels in 3D box">
                        <input type="checkbox" id="chk-transparent" checked onchange="toggleTransparency(this.checked)">
                        <span>Transparent 0s</span>
                    </label>
                    <label class="toggle-pill" title="3D collagen point cloud">
                        <input type="checkbox" id="chk-pointcloud" checked onchange="togglePointCloud(this.checked)">
                        <span>3D Points</span>
                    </label>
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

            <!-- Bottom Floating Seeds Dock -->
            <div class="seeds-dock">
                <div class="dock-header">
                    <span>Placed Fiber Seeds & Pairing Status</span>
                    <span id="seeds-count-tag" style="color: var(--accent-cyan); font-family: 'JetBrains Mono', monospace;">0 Seeds</span>
                </div>
                <div class="seeds-chip-container" id="seeds-chip-container">
                    <div style="color: var(--text-muted); font-size: 11px;">No seeds placed. Click on 3D Box faces or click Auto-Detect!</div>
                </div>
            </div>

            <div class="shortcuts-hint">
                <span><kbd>Click 3D Face</kbd> Drop Seed</span>
                <span><kbd>Space</kbd> Resolve</span>
                <span><kbd>Enter</kbd> Save & Next</span>
                <span><kbd>R</kbd> Random Cube</span>
                <span><kbd>[ / ]</kbd> Prev/Next Fiber ID</span>
            </div>
        </div>

        <!-- Right: 3D MIPs & Orthogonal Slice Scrubber -->
        <div class="preview-panel">
            <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center;">
                <div class="panel-title">Projections & Multi-Label Output</div>
                <div class="badge highlight" id="perf-badge">Zero-Model Geodesic Engine</div>
            </div>

            <!-- 3-Axis MIP Views -->
            <div class="mip-grid">
                <div class="mip-card">
                    <div class="mip-header">Top MIP (XY)</div>
                    <img id="mip-xy" class="mip-img" src="" alt="MIP XY">
                </div>
                <div class="mip-card">
                    <div class="mip-header">Front MIP (XZ)</div>
                    <img id="mip-xz" class="mip-img" src="" alt="MIP XZ">
                </div>
                <div class="mip-card">
                    <div class="mip-header">Side MIP (YZ)</div>
                    <img id="mip-yz" class="mip-img" src="" alt="MIP YZ">
                </div>
            </div>

            <!-- Direct 3D Midpoint / Waypoint Tool Card -->
            <div class="waypoint-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div class="panel-title" style="display: flex; align-items: center; gap: 6px;">
                        <span>📍 3D Midpoint Tool</span>
                        <span class="badge highlight" id="waypoint-target-fiber">Fiber #1</span>
                    </div>
                    <span style="font-size: 10px; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace;" id="slice-hover-readout">Slice Cursor: (X:-, Y:-, Z:-)</span>
                </div>

                <div class="waypoint-row">
                    <div class="coord-input-group" title="Scroll wheel or type to move X in 3D">
                        <span class="coord-label">X:</span>
                        <input type="number" id="wp-x" class="coord-num-input" min="0" max="95" value="48" oninput="onWaypointCoordInput()" onwheel="handleCoordWheel(event, 'wp-x')">
                    </div>
                    <div class="coord-input-group" title="Scroll wheel or type to move Y in 3D">
                        <span class="coord-label">Y:</span>
                        <input type="number" id="wp-y" class="coord-num-input" min="0" max="95" value="48" oninput="onWaypointCoordInput()" onwheel="handleCoordWheel(event, 'wp-y')">
                    </div>
                    <div class="coord-input-group" title="Scroll wheel or type to move Z in 3D">
                        <span class="coord-label">Z:</span>
                        <input type="number" id="wp-z" class="coord-num-input" min="0" max="95" value="48" oninput="onWaypointCoordInput()" onwheel="handleCoordWheel(event, 'wp-z')">
                    </div>

                    <button id="btn-wp-toggle" class="btn-insert-waypoint" onclick="toggleWaypoint()" title="Add or remove 3D midpoint for active fiber">
                        ➕ Add Midpoint
                    </button>
                </div>
            </div>

            <!-- Interactive Slice Scrubber -->
            <div class="slice-section">
                <div class="panel-header" style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="panel-title">Orthogonal 3D Slice Scrubber</div>
                    <span class="badge" id="slice-index-badge">Z: <strong>48 / 95</strong></span>
                </div>
                <div class="slice-controls">
                    <input type="range" id="slice-slider" class="slice-slider" min="0" max="95" value="48" step="1" oninput="handleSliceSlider(this.value)">
                </div>

                <div class="slice-views-row">
                    <div class="slice-view-box">
                        <span class="slice-label">Raw 0,1 Binary (Click to Drop Internal Seed)</span>
                        <img id="slice-img-raw" class="slice-img" src="" alt="Raw Slice" style="cursor: crosshair;" onclick="handleSliceClick(event)" onmousemove="handleSliceHover(event)" title="Click to auto-fill (X, Y, Z) coordinates or Shift+Click to add midpoint">
                    </div>
                    <div class="slice-view-box">
                        <span class="slice-label">1-Vx Centerlines</span>
                        <img id="slice-img-skel" class="slice-img" src="" alt="Centerline Slice" style="cursor: crosshair;" onclick="handleSliceClick(event)" onmousemove="handleSliceHover(event)">
                    </div>
                    <div class="slice-view-box">
                        <span class="slice-label">3D Instance Seg</span>
                        <img id="slice-img-inst" class="slice-img" src="" alt="Instance Slice" style="cursor: crosshair;" onclick="handleSliceClick(event)" onmousemove="handleSliceHover(event)">
                    </div>
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
        let autoCandidatesList = [];
        let allSlicesCache = [];
        let isTransparentMode = true;
        let latestResolvedCurves = {};

        // Three.js Global Variables
        let scene, camera, renderer, controls, raycaster, mouse;
        let faceMeshes = [];
        let offscreenCanvases = {};
        let faceTextures = {};
        let seedPinsGroup, curves3DGroup, pointCloudMesh;
        let currentSliderMax = 64;

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

            syncWaypointInputsForActiveFiber();
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

        function handleCoordWheel(event, inputId) {
            event.preventDefault();
            const input = document.getElementById(inputId);
            if (!input) return;
            let val = parseInt(input.value) || 0;
            val += (event.deltaY < 0 ? 1 : -1);
            val = Math.max(0, Math.min(95, val));
            input.value = val;
            onWaypointCoordInput();
        }

        function syncWaypointInputsForActiveFiber() {
            const targetBadge = document.getElementById('waypoint-target-fiber');
            const btnToggle = document.getElementById('btn-wp-toggle');

            if (targetBadge) {
                targetBadge.innerText = `Fiber #${activeFiberId}`;
                targetBadge.style.backgroundColor = COLOR_PALETTE[(activeFiberId - 1) % COLOR_PALETTE.length];
                targetBadge.style.color = '#051016';
            }

            const existingWp = seedsList.find(s => s.fiber_id === activeFiberId && (s.is_waypoint || s.face === 'waypoint' || s.face === 'internal'));
            if (existingWp) {
                document.getElementById('wp-x').value = existingWp.pos3d[2];
                document.getElementById('wp-y').value = existingWp.pos3d[1];
                document.getElementById('wp-z').value = existingWp.pos3d[0];
                if (btnToggle) {
                    btnToggle.innerText = '🗑️ Remove Midpoint';
                    btnToggle.style.background = 'linear-gradient(135deg, #ff1744 0%, #d500f9 100%)';
                }
            } else {
                // Default coordinates to the EXACT geometric middle of the fiber
                let midX = 48, midY = 48, midZ = 48;
                const curve = latestResolvedCurves[activeFiberId];
                if (curve && curve.length > 0) {
                    const midVoxel = curve[Math.floor(curve.length / 2)];
                    midZ = midVoxel[0];
                    midY = midVoxel[1];
                    midX = midVoxel[2];
                } else {
                    const fSeeds = seedsList.filter(s => s.fiber_id === activeFiberId);
                    if (fSeeds.length >= 2) {
                        midZ = Math.round((fSeeds[0].pos3d[0] + fSeeds[1].pos3d[0]) / 2);
                        midY = Math.round((fSeeds[0].pos3d[1] + fSeeds[1].pos3d[1]) / 2);
                        midX = Math.round((fSeeds[0].pos3d[2] + fSeeds[1].pos3d[2]) / 2);
                    }
                }
                document.getElementById('wp-x').value = midX;
                document.getElementById('wp-y').value = midY;
                document.getElementById('wp-z').value = midZ;
                if (btnToggle) {
                    btnToggle.innerText = '➕ Add Midpoint';
                    btnToggle.style.background = 'linear-gradient(135deg, #00e5ff 0%, #00b0ff 100%)';
                }
            }
        }

        function toggleWaypoint() {
            const existingIdx = seedsList.findIndex(s => s.fiber_id === activeFiberId && (s.is_waypoint || s.face === 'waypoint' || s.face === 'internal'));
            if (existingIdx >= 0) {
                // Remove waypoint
                seedsList.splice(existingIdx, 1);
                updateThreeSeeds();
                updateSeedsDock();
                syncWaypointInputsForActiveFiber();
                redrawFiberCurveLocally(activeFiberId);
            } else {
                // Create midpoint at current X, Y, Z
                onWaypointCoordInput(true);
                syncWaypointInputsForActiveFiber();
            }
        }

        function onWaypointCoordInput(forceAdd = false) {
            const x = parseInt(document.getElementById('wp-x').value);
            const y = parseInt(document.getElementById('wp-y').value);
            const z = parseInt(document.getElementById('wp-z').value);

            if (isNaN(x) || isNaN(y) || isNaN(z) || x < 0 || x >= 96 || y < 0 || y >= 96 || z < 0 || z >= 96) return;

            let wp = seedsList.find(s => s.fiber_id === activeFiberId && (s.is_waypoint || s.face === 'waypoint' || s.face === 'internal'));
            if (!wp) {
                const newWp = {
                    face: 'waypoint',
                    u: x,
                    v: y,
                    pos3d: [z, y, x],
                    fiber_id: activeFiberId,
                    is_waypoint: true
                };

                const fiberIndices = [];
                seedsList.forEach((s, idx) => {
                    if (s.fiber_id === activeFiberId) fiberIndices.push(idx);
                });
                if (fiberIndices.length >= 2) {
                    seedsList.splice(fiberIndices[fiberIndices.length - 1], 0, newWp);
                } else {
                    seedsList.push(newWp);
                }
                wp = newWp;
            } else {
                wp.pos3d = [z, y, x];
                wp.u = x;
                wp.v = y;
            }

            updateThreeSeeds();
            updateSeedsDock();
            syncWaypointInputsForActiveFiber();

            // INSTANT 3D Centerline deformation (No heavy server recomputation!)
            redrawFiberCurveLocally(activeFiberId);
        }

        function deleteWaypoint(fid, z, y, x) {
            const idx = seedsList.findIndex(s => s.fiber_id === fid && s.pos3d[0] === z && s.pos3d[1] === y && s.pos3d[2] === x);
            if (idx >= 0) {
                seedsList.splice(idx, 1);
                updateThreeSeeds();
                updateSeedsDock();
                syncWaypointInputsForActiveFiber();
                redrawFiberCurveLocally(fid);
            }
        }

        function redrawFiberCurveLocally(fid) {
            const fSeeds = seedsList.filter(s => s.fiber_id === fid);
            if (fSeeds.length < 2) return;

            // Extract 3D control points (X=x, Y=y, Z=z)
            const controlPoints = fSeeds.map(s => new THREE.Vector3(s.pos3d[2], s.pos3d[1], s.pos3d[0]));

            // Remove existing mesh for this fiber
            const oldMesh = curves3DGroup.children.find(c => c.userData && c.userData.fiberId === fid);
            if (oldMesh) {
                curves3DGroup.remove(oldMesh);
                if (oldMesh.geometry) oldMesh.geometry.dispose();
                if (oldMesh.material) oldMesh.material.dispose();
            }

            // Create smooth Catmull-Rom spline tube through start -> midpoint(s) -> end
            const curve = new THREE.CatmullRomCurve3(controlPoints);
            const tubeGeo = new THREE.TubeGeometry(curve, 32, 1.3, 8, false);
            const colorHex = parseInt(COLOR_PALETTE[(fid - 1) % COLOR_PALETTE.length].replace('#', '0x'));
            const tubeMat = new THREE.MeshStandardMaterial({
                color: colorHex,
                emissive: colorHex,
                emissiveIntensity: 0.75,
                roughness: 0.3
            });
            const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
            tubeMesh.userData = { fiberId: fid };
            curves3DGroup.add(tubeMesh);
        }

        function handleSliceHover(event) {
            const img = document.getElementById('slice-img-raw');
            if (!img) return;
            const rect = img.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;

            const S = 96;
            const x = Math.max(0, Math.min(S - 1, Math.round((clickX / rect.width) * (S - 1))));
            const y = Math.max(0, Math.min(S - 1, Math.round((clickY / rect.height) * (S - 1))));
            const z = parseInt(document.getElementById('slice-slider').value);

            const readout = document.getElementById('slice-hover-readout');
            if (readout) {
                readout.innerText = `Slice Cursor: (X:${x}, Y:${y}, Z:${z})`;
            }
        }

        function handleSliceClick(event) {
            const img = document.getElementById('slice-img-raw');
            const rect = img.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;

            const S = 96;
            const x = Math.max(0, Math.min(S - 1, Math.round((clickX / rect.width) * (S - 1))));
            const y = Math.max(0, Math.min(S - 1, Math.round((clickY / rect.height) * (S - 1))));
            const z = parseInt(document.getElementById('slice-slider').value);

            // Auto-populate the direct waypoint coordinate boxes
            document.getElementById('wp-x').value = x;
            document.getElementById('wp-y').value = y;
            document.getElementById('wp-z').value = z;

            // If Shift key held during click, directly add as midpoint
            if (event.shiftKey) {
                insertDirectWaypoint();
            }
        }

        function initThreeJS() {
            const container = document.getElementById('three-container');
            const width = container.clientWidth || 600;
            const height = container.clientHeight || 450;

            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
            camera.position.set(130, 110, 160);

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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

            // Bounding Wireframe Box
            const boxGeo = new THREE.BoxGeometry(96, 96, 96);
            const wireMat = new THREE.MeshBasicMaterial({
                color: 0x3b435d,
                wireframe: true,
                transparent: true,
                opacity: 0.55
            });
            const wireMesh = new THREE.Mesh(boxGeo, wireMat);
            wireMesh.position.set(48, 48, 48);
            scene.add(wireMesh);

            createThreeFacePlanes();

            seedPinsGroup = new THREE.Group();
            scene.add(seedPinsGroup);

            curves3DGroup = new THREE.Group();
            scene.add(curves3DGroup);

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

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        }

        function onWindowResize() {
            const container = document.getElementById('three-container');
            if (!container || !renderer || !camera) return;
            const width = container.clientWidth;
            const height = container.clientHeight;
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

        function createThreeFacePlanes() {
            faceMeshes = [];
            const S = 96;

            const faceConfigs = [
                {
                    name: 'z_min',
                    v0: [0, 0, 0], v1: [S, 0, 0], v2: [S, S, 0], v3: [0, S, 0]
                },
                {
                    name: 'z_max',
                    v0: [0, 0, S], v1: [S, 0, S], v2: [S, S, S], v3: [0, S, S]
                },
                {
                    name: 'y_min',
                    v0: [0, 0, 0], v1: [S, 0, 0], v2: [S, 0, S], v3: [0, 0, S]
                },
                {
                    name: 'y_max',
                    v0: [0, S, 0], v1: [S, S, 0], v2: [S, S, S], v3: [0, S, S]
                },
                {
                    name: 'x_min',
                    v0: [0, 0, 0], v1: [0, S, 0], v2: [0, S, S], v3: [0, 0, S]
                },
                {
                    name: 'x_max',
                    v0: [S, 0, 0], v1: [S, S, 0], v2: [S, S, S], v3: [S, 0, S]
                }
            ];

            faceConfigs.forEach(cfg => {
                const cvs = document.createElement('canvas');
                cvs.width = S;
                cvs.height = S;
                offscreenCanvases[cfg.name] = cvs;

                const tex = new THREE.CanvasTexture(cvs);
                tex.magFilter = THREE.NearestFilter;
                tex.minFilter = THREE.NearestFilter;
                tex.flipY = false;
                faceTextures[cfg.name] = tex;

                const geo = createQuadGeometry(cfg.v0, cfg.v1, cfg.v2, cfg.v3);
                const mat = new THREE.MeshBasicMaterial({
                    map: tex,
                    transparent: true,
                    opacity: 0.95,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const mesh = new THREE.Mesh(geo, mat);
                mesh.userData = { faceName: cfg.name };
                scene.add(mesh);
                faceMeshes.push(mesh);
            });
        }

        function updateThreeFaceTextures(faceImagesGrayscale, faceImagesRGBA) {
            const imagesToUse = isTransparentMode ? faceImagesRGBA : faceImagesGrayscale;
            if (!imagesToUse) return;

            Object.entries(imagesToUse).forEach(([faceName, b64Url]) => {
                if (!b64Url) return;
                const img = new Image();
                img.onload = () => {
                    const cvs = offscreenCanvases[faceName];
                    if (cvs) {
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

        function toggleTransparency(enabled) {
            isTransparentMode = enabled;
            if (currentPatchData) {
                updateThreeFaceTextures(currentPatchData.face_images, currentPatchData.face_images_rgba);
            }
        }

        function togglePointCloud(enabled) {
            if (pointCloudMesh) {
                pointCloudMesh.visible = enabled;
            }
        }

        function handleThreeFaceClick(event) {
            const rect = renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects(faceMeshes, true);

            if (intersects.length > 0) {
                const hit = intersects[0];
                const faceName = hit.object.userData.faceName;
                const pt = hit.point;

                let z = Math.max(0, Math.min(95, Math.round(pt.z)));
                let y = Math.max(0, Math.min(95, Math.round(pt.y)));
                let x = Math.max(0, Math.min(95, Math.round(pt.x)));

                if (faceName === 'z_min') z = 0;
                else if (faceName === 'z_max') z = 95;
                else if (faceName === 'y_min') y = 0;
                else if (faceName === 'y_max') y = 95;
                else if (faceName === 'x_min') x = 0;
                else if (faceName === 'x_max') x = 95;

                let u = y, v = x;
                if (faceName === 'z_min' || faceName === 'z_max') { u = y; v = x; }
                else if (faceName === 'y_min' || faceName === 'y_max') { u = z; v = x; }
                else { u = z; v = y; }

                const existingIdx = seedsList.findIndex(s => s.face === faceName && Math.hypot(s.u - u, s.v - v) < 6.0);
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

                    // QoL: Only auto-advance if we just completed the 2nd point for the HIGHEST/latest current Fiber ID
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

            // Solid background circle with bright white border
            ctx.beginPath();
            ctx.arc(64, 64, 54, 0, Math.PI * 2);
            ctx.fillStyle = colorHexStr;
            ctx.fill();
            ctx.lineWidth = 10;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();

            // Dark drop shadow circle for contrast
            ctx.beginPath();
            ctx.arc(64, 64, 46, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.40)';
            ctx.fill();

            // Crisp bold number text
            ctx.fillStyle = '#ffffff';
            ctx.font = `900 ${fiberId > 99 ? 42 : (fiberId > 9 ? 54 : 66)}px "JetBrains Mono", Arial, sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`${fiberId}`, 64, 66);

            const texture = new THREE.CanvasTexture(canvas);
            texture.minFilter = THREE.LinearFilter;
            const mat = new THREE.SpriteMaterial({
                map: texture,
                depthTest: false,
                transparent: true
            });
            const sprite = new THREE.Sprite(mat);
            sprite.scale.set(7.5, 7.5, 1.0);
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

            seedsList.forEach(s => {
                const z = s.pos3d[0];
                const y = s.pos3d[1];
                const x = s.pos3d[2];

                const colorHexStr = COLOR_PALETTE[(s.fiber_id - 1) % COLOR_PALETTE.length];
                const colorHex = parseInt(colorHexStr.replace('#', '0x'));
                const isWp = s.is_waypoint || s.face === 'waypoint' || s.face === 'internal';

                // 3D Geometry: Glowing Diamond/Octahedron for internal waypoints, Sphere for boundary pins
                const pinGeo = isWp ? new THREE.OctahedronGeometry(2.8, 0) : new THREE.SphereGeometry(2.0, 16, 16);
                const pinMat = new THREE.MeshStandardMaterial({
                    color: isWp ? 0xffffff : colorHex,
                    emissive: colorHex,
                    emissiveIntensity: isWp ? 1.4 : 0.85,
                    roughness: 0.2
                });
                const pinMesh = new THREE.Mesh(pinGeo, pinMat);
                pinMesh.position.set(x, y, z);
                seedPinsGroup.add(pinMesh);

                // 3D Number Billboard Sprite Badge
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

            if (!curvesDict) {
                syncWaypointInputsForActiveFiber();
                return;
            }

            Object.entries(curvesDict).forEach(([fidStr, pts]) => {
                const fid = parseInt(fidStr);
                const colorHex = parseInt(COLOR_PALETTE[(fid - 1) % COLOR_PALETTE.length].replace('#', '0x'));

                // Three.js vectors: X=x, Y=y, Z=z
                const vectors = pts.map(p => new THREE.Vector3(p[2], p[1], p[0]));
                if (vectors.length >= 2) {
                    const curve = new THREE.CatmullRomCurve3(vectors);
                    const tubeGeo = new THREE.TubeGeometry(curve, Math.max(20, pts.length), 1.3, 8, false);
                    const tubeMat = new THREE.MeshStandardMaterial({
                        color: colorHex,
                        emissive: colorHex,
                        emissiveIntensity: 0.75,
                        roughness: 0.3
                    });
                    const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
                    tubeMesh.userData = { fiberId: fid };
                    curves3DGroup.add(tubeMesh);
                }
            });

            syncWaypointInputsForActiveFiber();
        }

        function updateThreePointCloud(points) {
            if (pointCloudMesh) {
                scene.remove(pointCloudMesh);
                if (pointCloudMesh.geometry) pointCloudMesh.geometry.dispose();
                if (pointCloudMesh.material) pointCloudMesh.material.dispose();
                pointCloudMesh = null;
            }

            if (!points || points.length === 0) return;

            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array(points.length * 3);
            for (let i = 0; i < points.length; i++) {
                // Map numpy (z, y, x) to 3D Three.js (x, y, z)
                positions[i * 3] = points[i][2];     // X = x
                positions[i * 3 + 1] = points[i][1]; // Y = y
                positions[i * 3 + 2] = points[i][0]; // Z = z
            }
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

            const mat = new THREE.PointsMaterial({
                color: 0x00e5ff,
                size: 2.2,
                transparent: true,
                opacity: 0.45,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            });
            pointCloudMesh = new THREE.Points(geo, mat);
            pointCloudMesh.visible = document.getElementById('chk-pointcloud').checked;
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

                document.getElementById('perf-badge').innerText = `⏳ Loading 96³ Subvolume at (${z}, ${y}, ${x})...`;
                const res = await fetch('/api/patch/coords', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ z, y, x })
                });
                const data = await res.json();
                if (!res.ok) {
                    alert(data.error || "Failed to load coordinates");
                    return;
                }

                currentPatchData = data;
                seedsList = [];
                autoCandidatesList = data.auto_candidates || [];

                updateCoordinateInputs(data.origin, data.max_origin);

                document.getElementById('origin-badge').innerHTML = `Origin: <strong>(Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})</strong>`;
                document.getElementById('density-badge').innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
                document.getElementById('curated-count').innerText = data.curated_total;
                document.getElementById('perf-badge').innerText = `📍 Loaded 96³ Subvolume at Origin (Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})`;

                // Update 3D Face Canvas Textures & Point Cloud
                updateThreeFaceTextures(data.face_images, data.face_images_rgba);
                updateThreePointCloud(data.point_cloud);

                // Update 3D MIPs
                document.getElementById('mip-xy').src = data.mip_images.raw_xy;
                document.getElementById('mip-xz').src = data.mip_images.raw_xz;
                document.getElementById('mip-yz').src = data.mip_images.raw_yz;

                // Update Slices
                allSlicesCache = data.slices || [];
                handleSliceSlider(document.getElementById('slice-slider').value);

                // Clear 3D curves & seed pins & reset active ID to 1
                activeFiberId = 1;
                renderThree3DCurves(null);
                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            } catch (err) {
                console.error("Failed to load coordinates:", err);
                document.getElementById('perf-badge').innerText = '❌ Error loading coordinates';
            }
        }

        function nudgeCoords(dz, dy, dx) {
            const zInput = document.getElementById('load-coord-z');
            const yInput = document.getElementById('load-coord-y');
            const xInput = document.getElementById('load-coord-x');

            let z = (parseInt(zInput ? zInput.value : 0) || 0) + dz;
            let y = (parseInt(yInput ? yInput.value : 0) || 0) + dy;
            let x = (parseInt(xInput ? xInput.value : 0) || 0) + dx;

            if (zInput && zInput.max) z = Math.max(0, Math.min(parseInt(zInput.max), z));
            if (yInput && yInput.max) y = Math.max(0, Math.min(parseInt(yInput.max), y));
            if (xInput && xInput.max) x = Math.max(0, Math.min(parseInt(xInput.max), x));

            if (zInput) zInput.value = z;
            if (yInput) yInput.value = y;
            if (xInput) xInput.value = x;

            loadManualCoordinates(z, y, x);
        }

        async function loadSavedPatch(patchId) {
            if (!patchId) return;
            try {
                document.getElementById('perf-badge').innerText = `⏳ Loading Saved Sample #${String(patchId).padStart(4, '0')}...`;
                const res = await fetch('/api/patch/load', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ patch_id: parseInt(patchId) })
                });
                const data = await res.json();
                if (!res.ok) {
                    alert(data.error || "Failed to load patch");
                    return;
                }

                currentlyLoadedPatchIndex = parseInt(patchId);
                currentPatchData = data;
                seedsList = data.loaded_seeds || [];
                autoCandidatesList = [];

                updateCoordinateInputs(data.origin, data.max_origin);

                document.getElementById('origin-badge').innerHTML = `Sample: <strong>#${String(patchId).padStart(4, '0')}</strong> (Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})`;
                document.getElementById('density-badge').innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
                document.getElementById('curated-count').innerText = data.curated_total;
                document.getElementById('perf-badge').innerText = `⚡ Loaded Saved Sample #${String(patchId).padStart(4, '0')} with ${data.num_fibers} Fibers!`;

                // Update 3D Face Canvas Textures & Point Cloud
                updateThreeFaceTextures(data.face_images, data.face_images_rgba);
                updateThreePointCloud(data.point_cloud);

                // Update 3D MIPs
                document.getElementById('mip-xy').src = data.mip_results?.inst_xy || data.mip_images.raw_xy;
                document.getElementById('mip-xz').src = data.mip_results?.inst_xz || data.mip_images.raw_xz;
                document.getElementById('mip-yz').src = data.mip_results?.inst_yz || data.mip_images.raw_yz;

                // Update Slices
                allSlicesCache = data.slices || [];
                handleSliceSlider(document.getElementById('slice-slider').value);

                // Render 3D curves & seed pins
                renderThree3DCurves(data.curves_3d);
                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            } catch (err) {
                console.error("Failed to load saved patch:", err);
                document.getElementById('perf-badge').innerText = '❌ Error loading saved patch';
            }
        }

        async function fetchNewRandomPatch() {
            try {
                currentlyLoadedPatchIndex = null;
                const sel = document.getElementById('saved-patches-select');
                if (sel) sel.value = '';

                document.getElementById('perf-badge').innerText = '⏳ Loading 96³ Cube...';
                const res = await fetch('/api/patch/new?t=' + Date.now(), { cache: 'no-store' });
                const data = await res.json();
                currentPatchData = data;
                seedsList = [];
                autoCandidatesList = data.auto_candidates || [];

                updateCoordinateInputs(data.origin, data.max_origin);

                document.getElementById('origin-badge').innerHTML = `Origin: <strong>(Z=${data.origin[0]}, Y=${data.origin[1]}, X=${data.origin[2]})</strong>`;
                document.getElementById('density-badge').innerHTML = `Density: <strong>${(data.density * 100).toFixed(1)}%</strong>`;
                document.getElementById('curated-count').innerText = data.curated_total;
                document.getElementById('perf-badge').innerText = 'Ready for Seed Placement';

                // Update 3D Face Canvas Textures & Point Cloud
                updateThreeFaceTextures(data.face_images, data.face_images_rgba);
                updateThreePointCloud(data.point_cloud);

                // Update 3D MIPs
                document.getElementById('mip-xy').src = data.mip_images.raw_xy;
                document.getElementById('mip-xz').src = data.mip_images.raw_xz;
                document.getElementById('mip-yz').src = data.mip_images.raw_yz;

                // Update Slices
                allSlicesCache = data.slices || [];
                handleSliceSlider(document.getElementById('slice-slider').value);

                // Clear 3D curves & seed pins & reset active ID to 1
                activeFiberId = 1;
                renderThree3DCurves(null);
                updateThreeSeeds();
                updateSeedsDock();
                initIdSelector();
            } catch (err) {
                console.error("Failed to fetch patch:", err);
                document.getElementById('perf-badge').innerText = '❌ Error fetching patch';
            }
        }

        function autoPopulateCandidates() {
            if (!autoCandidatesList || autoCandidatesList.length === 0) return;
            seedsList = [];
            autoCandidatesList.forEach((cand, idx) => {
                seedsList.push({
                    face: cand.face,
                    u: cand.u,
                    v: cand.v,
                    pos3d: cand.pos3d,
                    fiber_id: Math.floor(idx / 2) + 1
                });
            });

            updateThreeSeeds();
            updateSeedsDock();
            initIdSelector();
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
                container.innerHTML = `<div style="color: var(--text-muted); font-size: 11px;">No seeds placed. Click on 3D Box faces or click Auto-Detect!</div>`;
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
                    <span class="chip-color-dot" style="background: ${color}; cursor:pointer;" onclick="setActiveId(${id})" title="Click to make Fiber #${id} active"></span>
                    <strong style="cursor:pointer;" onclick="setActiveId(${id})" title="Click to make Fiber #${id} active">Fiber ${id}</strong>
                    <span class="chip-status ${statusClass}">${statusText}</span>
                    <div style="display:inline-flex; align-items:center; flex-wrap:wrap; gap:3px;">${coordsHtml}</div>
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
                const res = await fetch('/api/resolve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ seeds: seedsList })
                });
                const data = await res.json();

                if (data.success) {
                    document.getElementById('perf-badge').innerText = `⚡ Resolved ${data.num_fibers} Fibers in ${data.resolve_time_ms} ms!`;
                    renderThree3DCurves(data.curves_3d);

                    document.getElementById('mip-xy').src = data.mip_results.inst_xy || data.mip_results.raw_xy;
                    document.getElementById('mip-xz').src = data.mip_results.inst_xz || data.mip_results.raw_xz;
                    document.getElementById('mip-yz').src = data.mip_results.inst_yz || data.mip_results.raw_yz;

                    allSlicesCache = data.slices || [];
                    handleSliceSlider(document.getElementById('slice-slider').value);
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

        function handleSliceSlider(val) {
            document.getElementById('slice-index-badge').innerHTML = `Z: <strong>${val} / 95</strong>`;
            if (!allSlicesCache || allSlicesCache.length === 0) return;

            const targetZ = parseInt(val);
            let closest = allSlicesCache[0];
            let minDiff = 999;
            allSlicesCache.forEach(s => {
                const diff = Math.abs(s.z - targetZ);
                if (diff < minDiff) {
                    minDiff = diff;
                    closest = s;
                }
            });

            if (closest) {
                document.getElementById('slice-img-raw').src = closest.raw;
                document.getElementById('slice-img-skel').src = closest.skel;
                document.getElementById('slice-img-inst').src = closest.inst;
            }
        }

        window.addEventListener('keydown', (e) => {
            // Do not trigger global hotkeys when typing inside inputs or select elements
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
    res = engine.resolve_connections(seeds)
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
    print(f" REAL COLLAGEN 3D BOX ANNOTATOR RUNNING AT: http://127.0.0.1:{port}")
    print("=" * 80 + "\n")
    app.run(host='127.0.0.1', port=port, debug=False)
