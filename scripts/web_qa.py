#!/usr/bin/env python
"""
Local web UI for the DRIVE QA pipeline.

Run:
    python scripts/web_qa.py --port 8502
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple

# Ensure src layout is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drive_qa import DriveQAPipeline, create_pipeline  # noqa: E402
from drive_qa.model_registry import list_models, is_gemini_model, MODEL_REGISTRY  # noqa: E402

DEFAULT_MODEL = "gemini-3.1-flash-lite"
MODEL_OPTIONS = list(MODEL_REGISTRY.keys())

# Database connection parameters (schema/driver are fixed; host/port/user/pass provided at runtime)
DB_SCHEMA = "dr"
DB_DRIVER = "mysql+pymysql"


INDEX_HTML = """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DRIVE QA</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #202124;
      --muted: #667085;
      --line: #d8dee7;
      --soft: #f5f7fa;
      --paper: #ffffff;
      --teal: #137f7a;
      --teal-ink: #075451;
      --amber: #e8a317;
      --coral: #d95d39;
      --violet: #6956b4;
      --shadow: 0 18px 48px rgba(20, 32, 52, 0.12);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(19,127,122,0.08), transparent 34%),
        linear-gradient(180deg, #f8fafb 0%, #eef3f4 100%);
    }

    button, input, textarea, select {
      font: inherit;
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
    }

    .query-pane {
      padding: 28px;
      background: rgba(255,255,255,0.86);
      border-right: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand h1 {
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: 0;
    }

    .status-pill {
      min-width: 94px;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      color: var(--muted);
      background: #fff;
      font-size: 0.82rem;
    }

    .status-pill.busy {
      color: var(--teal-ink);
      border-color: rgba(19,127,122,0.28);
      background: rgba(19,127,122,0.09);
    }

    .status-pill.connected {
      color: #065f46;
      border-color: rgba(5,150,105,0.35);
      background: rgba(5,150,105,0.09);
    }

    .status-pill.error {
      color: #9b2f19;
      border-color: rgba(217,93,57,0.35);
      background: rgba(217,93,57,0.09);
    }

    label {
      display: block;
      margin-bottom: 8px;
      color: #344054;
      font-weight: 650;
      font-size: 0.88rem;
    }

    textarea, input, select {
      width: 100%;
      border: 1px solid #c9d2df;
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      outline: none;
      transition: border-color .16s ease, box-shadow .16s ease;
    }

    textarea:focus, input:focus, select:focus {
      border-color: var(--teal);
      box-shadow: 0 0 0 4px rgba(19,127,122,0.12);
    }

    textarea {
      min-height: 166px;
      resize: vertical;
      padding: 14px;
      line-height: 1.45;
    }

    input, select {
      height: 42px;
      padding: 0 12px;
    }

    .settings {
      display: grid;
      gap: 16px;
    }

    .run-row {
      display: grid;
      grid-template-columns: 1fr 48px;
      gap: 10px;
      align-items: center;
    }

    .primary {
      height: 48px;
      border: 0;
      border-radius: 8px;
      color: #fff;
      background: var(--teal);
      font-weight: 750;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(19,127,122,0.23);
    }

    .primary:disabled {
      opacity: .58;
      cursor: not-allowed;
    }

    .icon-button, .tool-button {
      border: 1px solid var(--line);
      background: #fff;
      color: #344054;
      border-radius: 8px;
      cursor: pointer;
    }

    .icon-button {
      width: 48px;
      height: 48px;
      font-size: 1.1rem;
    }

    .hint {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }

    .examples {
      display: grid;
      gap: 8px;
    }

    .example {
      text-align: left;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: #344054;
      cursor: pointer;
      line-height: 1.35;
    }

    .output-pane {
      min-width: 0;
      padding: 28px clamp(22px, 4vw, 54px);
      display: grid;
      grid-template-rows: auto auto 1fr;
      gap: 18px;
    }

    .summary-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      border-bottom: 3px solid var(--teal);
      background: rgba(255,255,255,0.82);
      padding: 14px;
      min-height: 74px;
    }

    .metric:nth-child(2) { border-color: var(--amber); }
    .metric:nth-child(3) { border-color: var(--coral); }
    .metric:nth-child(4) { border-color: var(--violet); }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: .78rem;
      margin-bottom: 8px;
    }

    .metric strong {
      font-size: 1.12rem;
      word-break: break-word;
    }

    .workspace {
      min-width: 0;
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 16px;
    }

    .panel {
      background: rgba(255,255,255,0.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      min-height: 52px;
      padding: 12px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }

    .panel-header h2 {
      margin: 0;
      font-size: .98rem;
      letter-spacing: 0;
    }

    .tools {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .tool-button {
      min-height: 34px;
      padding: 0 10px;
      font-weight: 650;
      font-size: .84rem;
    }

    .panel-body {
      padding: 16px;
    }

    .answer-text {
      line-height: 1.55;
      min-height: 86px;
    }

    .answer-text p {
      margin: 0 0 10px;
    }

    .answer-text p:last-child {
      margin-bottom: 0;
    }

    .answer-text ul, .answer-text ol {
      margin: 8px 0;
      padding-left: 22px;
    }

    .answer-text li {
      margin: 6px 0;
    }

    .answer-text table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      min-width: auto;
      font-size: .88rem;
    }

    .answer-text th, .answer-text td {
      padding: 8px 10px;
      border: 1px solid var(--line);
      text-align: left;
    }

    .answer-text th {
      background: var(--soft);
      font-weight: 650;
    }

    .answer-text h3, .answer-text h4 {
      margin: 14px 0 6px;
      font-size: 1rem;
    }

    .answer-text em { font-style: italic; }

    .answer-text.empty, .empty-state {
      color: var(--muted);
    }

    pre.sql-block {
      margin: 0;
      padding: 16px;
      overflow: auto;
      background: #111827;
      color: #eef2ff;
      line-height: 1.55;
      min-height: 94px;
      max-height: 320px;
      font-size: .88rem;
      font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
      white-space: pre-wrap;
      word-break: break-word;
    }

    pre.sql-block .kw {
      color: #93c5fd;
      font-weight: 600;
    }

    .table-wrap {
      overflow: auto;
      max-height: 440px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 680px;
      background: #fff;
    }

    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid #edf1f5;
      text-align: left;
      vertical-align: top;
      font-size: .88rem;
    }

    th {
      position: sticky;
      top: 0;
      background: #f4f7fa;
      color: #344054;
      z-index: 1;
    }

    td {
      max-width: 360px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .error-box {
      display: none;
      border-left: 4px solid var(--coral);
      background: rgba(217,93,57,0.09);
      color: #7b2716;
      padding: 12px 14px;
      line-height: 1.45;
    }

    .error-box.visible { display: block; }

    /* Configuration section */
    .config-section {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: var(--soft);
      display: grid;
      gap: 14px;
    }

    .config-section.collapsed {
      padding: 12px 18px;
      gap: 0;
    }

    .config-section.collapsed .config-fields { display: none; }

    .config-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .config-header h2 {
      margin: 0;
      font-size: .95rem;
    }

    .config-badge {
      font-size: .82rem;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(5,150,105,0.35);
      color: #065f46;
      background: rgba(5,150,105,0.09);
    }

    .config-fields {
      display: grid;
      gap: 12px;
    }

    .config-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
    }

    .btn-connect {
      height: 42px;
      padding: 0 18px;
      border: 0;
      border-radius: 8px;
      color: #fff;
      background: var(--teal);
      font-weight: 700;
      cursor: pointer;
    }

    .btn-connect:disabled {
      opacity: .58;
      cursor: wait;
    }

    .btn-disconnect {
      border: 1px solid var(--line);
      background: #fff;
      color: #344054;
      border-radius: 8px;
      padding: 4px 10px;
      font-size: .82rem;
      cursor: pointer;
    }

    .config-error {
      color: #9b2f19;
      font-size: .84rem;
      line-height: 1.4;
    }

    @media (max-width: 920px) {
      .shell {
        grid-template-columns: 1fr;
      }
      .query-pane {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .summary-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 560px) {
      .query-pane, .output-pane {
        padding: 18px;
      }
      .summary-strip {
        grid-template-columns: 1fr;
      }
      .panel-header {
        align-items: flex-start;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="query-pane">
      <div class="brand">
        <h1>DRIVE QA</h1>
        <div id="status" class="status-pill">Not configured</div>
      </div>

      <!-- Configuration section -->
      <div id="configSection" class="config-section">
        <div class="config-header">
          <h2>Configuration</h2>
          <span id="configBadge" class="config-badge" style="display:none"></span>
          <button id="disconnectBtn" class="btn-disconnect" style="display:none" type="button">Disconnect</button>
        </div>
        <div id="configFields" class="config-fields">
          <div>
            <label for="dbHost">Database Host</label>
            <input id="dbHost" type="text" placeholder="e.g. 192.168.1.100" autocomplete="off" />
          </div>
          <div>
            <label for="dbPort">Database Port</label>
            <input id="dbPort" type="number" placeholder="e.g. 3306" min="1" max="65535" autocomplete="off" />
          </div>
          <div>
            <label for="dbUser">Database User</label>
            <input id="dbUser" type="text" placeholder="Username" autocomplete="off" />
          </div>
          <div>
            <label for="dbPass">Database Password</label>
            <input id="dbPass" type="password" placeholder="Password" autocomplete="off" />
          </div>
          <div class="config-row">
            <div></div>
            <button id="connectBtn" class="btn-connect" type="button">Connect</button>
          </div>
          <div id="configError" class="config-error" style="display:none"></div>
        </div>
      </div>

      <form id="qa-form">
        <label for="question">Ask your question</label>
        <textarea id="question" name="question" placeholder="Example: Does TAMOXIFEN appear as a candidate for Breast Neoplasms in GNNs, network proximity, and threshold values?" disabled></textarea>

        <div class="settings">
          <div>
            <label for="model">LLM Model</label>
            <select id="model" name="model" disabled>__MODEL_OPTIONS__</select>
          </div>
          <div>
            <label for="apiKey">LLM API Key</label>
            <input id="apiKey" type="password" placeholder="API key for the selected model" autocomplete="off" disabled />
          </div>
          <div class="run-row">
            <button id="runButton" class="primary" type="submit" disabled>Ask</button>
            <button class="icon-button" type="button" id="clearButton" title="Clear">X</button>
          </div>
        </div>
      </form>

      <div class="examples">
        <div class="hint">Quick examples</div>
        <button class="example" type="button">What drugs are suggested for rheumatoid Arthritis by both GNNs and network proximity?</button>
        <button class="example" type="button">Which drug has the highest score in the GNN REDIRECTION model for Abdominal Cramps?</button>
        <button class="example" type="button">Is there any information path between Diabetes Mellitus and IBUPROFEN? Which ones?</button>
      </div>
    </section>

    <section class="output-pane">
      <div id="errorBox" class="error-box"></div>

      <div class="summary-strip" aria-live="polite">
        <div class="metric"><span>Rows returned</span><strong id="metricRows">0</strong></div>
        <div class="metric"><span>Columns</span><strong id="metricCols">0</strong></div>
        <div class="metric"><span>Truncated</span><strong id="metricTrunc">No</strong></div>
        <div class="metric"><span>Time</span><strong id="metricTime">-</strong></div>
      </div>

      <div class="workspace">
        <section class="panel">
          <div class="panel-header">
            <h2>Answer</h2>
          </div>
          <div class="panel-body">
            <div id="answer" class="answer-text empty">Configure your database and API key, then ask a question.</div>
          </div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h2>Result Table</h2>
            <div class="tools">
              <button class="tool-button" type="button" id="downloadJson">JSON</button>
              <button class="tool-button" type="button" id="downloadCsv">CSV</button>
            </div>
          </div>
          <div id="tableWrap" class="table-wrap">
            <div class="panel-body empty-state">Results will appear here.</div>
          </div>
        </section>
        <section class="panel">
          <div class="panel-header">
            <h2>Generated SQL</h2>
            <div class="tools">
              <button class="tool-button" type="button" id="copySql">Copy SQL</button>
            </div>
          </div>
          <pre id="sql" class="sql-block">(no query yet)</pre>
        </section>
      </div>
    </section>
  </main>

  <script>
    // ─── DOM refs ──────────────────────────────────────────────────────────────
    const form = document.getElementById('qa-form');
    const statusEl = document.getElementById('status');
    const runButton = document.getElementById('runButton');
    const clearButton = document.getElementById('clearButton');
    const questionEl = document.getElementById('question');
    const modelEl = document.getElementById('model');
    const answerEl = document.getElementById('answer');
    const sqlEl = document.getElementById('sql');
    const tableWrap = document.getElementById('tableWrap');
    const errorBox = document.getElementById('errorBox');
    const metrics = {
      rows: document.getElementById('metricRows'),
      cols: document.getElementById('metricCols'),
      trunc: document.getElementById('metricTrunc'),
      time: document.getElementById('metricTime')
    };

    // Config UI refs
    const configSection = document.getElementById('configSection');
    const configFields = document.getElementById('configFields');
    const configBadge = document.getElementById('configBadge');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const connectBtn = document.getElementById('connectBtn');
    const dbHostEl = document.getElementById('dbHost');
    const dbPortEl = document.getElementById('dbPort');
    const dbUserEl = document.getElementById('dbUser');
    const dbPassEl = document.getElementById('dbPass');
    const apiKeyEl = document.getElementById('apiKey');
    const configError = document.getElementById('configError');

    // ─── State (credentials in memory only, never persisted) ───────────────────
    let dbHost = '';
    let dbPort = 0;
    let dbUser = '';
    let dbPass = '';
    let isConnected = false;
    let currentResult = { rows: [], columns: [], sql: '' };

    // ─── Helpers ───────────────────────────────────────────────────────────────
    function setStatus(text, mode) {
      statusEl.textContent = text;
      statusEl.className = 'status-pill' + (mode ? ' ' + mode : '');
    }

    function showError(message) {
      errorBox.textContent = message || '';
      errorBox.classList.toggle('visible', Boolean(message));
    }

    function showConfigError(msg) {
      configError.textContent = msg || '';
      configError.style.display = msg ? 'block' : 'none';
    }

    function setFormEnabled(enabled) {
      questionEl.disabled = !enabled;
      modelEl.disabled = !enabled;
      apiKeyEl.disabled = !enabled;
      runButton.disabled = !enabled;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[ch]));
    }

    // ─── Configuration ─────────────────────────────────────────────────────────
    connectBtn.addEventListener('click', async () => {
      const host = dbHostEl.value.trim();
      const portVal = parseInt(dbPortEl.value.trim(), 10);
      const user = dbUserEl.value.trim();
      const pass = dbPassEl.value.trim();

      if (!host) { showConfigError('Database host is required.'); return; }
      if (!dbPortEl.value.trim() || isNaN(portVal) || portVal < 1 || portVal > 65535) { showConfigError('Database port must be a number between 1 and 65535.'); return; }
      if (!user) { showConfigError('Database user is required.'); return; }
      if (!pass) { showConfigError('Database password is required.'); return; }

      showConfigError('');
      connectBtn.disabled = true;
      connectBtn.textContent = 'Connecting...';

      try {
        const resp = await fetch('/api/configure', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ db_host: host, db_port: portVal, db_user: user, db_pass: pass })
        });
        const data = await resp.json();
        if (data.ok) {
          dbHost = host;
          dbPort = portVal;
          dbUser = user;
          dbPass = pass;
          isConnected = true;
          // Collapse config section
          configSection.classList.add('collapsed');
          configBadge.textContent = '\\u2713 Connected';
          configBadge.style.display = 'inline-block';
          disconnectBtn.style.display = 'inline-block';
          setFormEnabled(true);
          setStatus('Ready', 'connected');
          // Clear sensitive inputs
          dbHostEl.value = '';
          dbPortEl.value = '';
          dbUserEl.value = '';
          dbPassEl.value = '';
        } else {
          showConfigError(data.error || 'Connection failed.');
        }
      } catch (err) {
        showConfigError('Network error: ' + (err.message || String(err)));
      } finally {
        connectBtn.disabled = false;
        connectBtn.textContent = 'Connect';
      }
    });

    disconnectBtn.addEventListener('click', () => {
      dbHost = '';
      dbPort = 0;
      dbUser = '';
      dbPass = '';
      apiKeyEl.value = '';
      isConnected = false;
      configSection.classList.remove('collapsed');
      configBadge.style.display = 'none';
      disconnectBtn.style.display = 'none';
      setFormEnabled(false);
      setStatus('Not configured', '');
      showConfigError('');
    });

    // ─── SQL Formatter ─────────────────────────────────────────────────────────
    function formatSql(sql) {
      if (!sql) return '';
      // Insert newlines before major SQL keywords (case-insensitive)
      const keywords = /\\b(SELECT|FROM|INNER JOIN|LEFT JOIN|RIGHT JOIN|CROSS JOIN|JOIN|ON|WHERE|AND|OR|GROUP BY|ORDER BY|HAVING|LIMIT|UNION ALL|UNION|SET|VALUES)\\b/gi;
      let formatted = sql.replace(keywords, (match) => {
        return '\\n' + match.toUpperCase();
      });
      // Indent subordinate clauses
      formatted = formatted.replace(/\\n(ON|AND|OR)\\b/g, '\\n  $1');
      // Clean up leading newline
      formatted = formatted.replace(/^\\n/, '');
      return formatted;
    }

    function highlightSql(text) {
      // Apply keyword highlighting (text is already escaped)
      const kws = /\\b(SELECT|DISTINCT|FROM|INNER JOIN|LEFT JOIN|RIGHT JOIN|CROSS JOIN|JOIN|ON|WHERE|AND|OR|NOT|IN|IS|NULL|AS|GROUP BY|ORDER BY|HAVING|LIMIT|UNION|ALL|SET|VALUES|INSERT|UPDATE|DELETE|COUNT|MAX|MIN|AVG|SUM|ASC|DESC|BETWEEN|LIKE|EXISTS|CASE|WHEN|THEN|ELSE|END|CTE|WITH)\\b/g;
      return text.replace(kws, '<span class="kw">$1</span>');
    }

    // ─── Markdown Renderer ─────────────────────────────────────────────────────
    function renderAnswerMarkdown(text) {
      const lines = String(text || '').split(/\\r?\\n/);
      const blocks = [];
      let listItems = [];
      let olItems = [];
      let tableRows = [];
      let paragraph = [];

      const inlineFormat = (value) => {
        let s = escapeHtml(value);
        // Bold
        s = s.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
        // Italic (single * not followed by space, avoiding conflict with bold)
        s = s.replace(/(?<!\\*)\\*(?!\\*)(.+?)(?<!\\*)\\*(?!\\*)/g, '<em>$1</em>');
        // Inline code
        s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
        return s;
      };

      const flushParagraph = () => {
        if (!paragraph.length) return;
        blocks.push('<p>' + inlineFormat(paragraph.join(' ')) + '</p>');
        paragraph = [];
      };

      const flushList = () => {
        if (!listItems.length) return;
        blocks.push('<ul>' + listItems.map(item => '<li>' + inlineFormat(item) + '</li>').join('') + '</ul>');
        listItems = [];
      };

      const flushOl = () => {
        if (!olItems.length) return;
        blocks.push('<ol>' + olItems.map(item => '<li>' + inlineFormat(item) + '</li>').join('') + '</ol>');
        olItems = [];
      };

      const flushTable = () => {
        if (tableRows.length < 2) {
          // Not a valid table, treat as paragraphs
          tableRows.forEach(r => paragraph.push(r));
          tableRows = [];
          return;
        }
        // First row is header, second is separator (---), rest are data
        const headerCells = tableRows[0].split('|').map(c => c.trim()).filter(c => c);
        const dataRows = tableRows.slice(2); // Skip separator line
        let html = '<table><thead><tr>' + headerCells.map(c => '<th>' + inlineFormat(c) + '</th>').join('') + '</tr></thead><tbody>';
        dataRows.forEach(row => {
          const cells = row.split('|').map(c => c.trim()).filter(c => c);
          html += '<tr>' + cells.map(c => '<td>' + inlineFormat(c) + '</td>').join('') + '</tr>';
        });
        html += '</tbody></table>';
        blocks.push(html);
        tableRows = [];
      };

      lines.forEach(line => {
        const trimmed = line.trim();

        // Table detection: line starts and ends with |
        if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
          flushParagraph();
          flushList();
          flushOl();
          tableRows.push(trimmed);
          return;
        } else if (tableRows.length > 0) {
          flushTable();
        }

        if (!trimmed) {
          flushParagraph();
          flushList();
          flushOl();
          return;
        }

        // Headers
        const h3 = trimmed.match(/^###\\s+(.+)$/);
        if (h3) { flushParagraph(); flushList(); flushOl(); blocks.push('<h4>' + inlineFormat(h3[1]) + '</h4>'); return; }
        const h2 = trimmed.match(/^##\\s+(.+)$/);
        if (h2) { flushParagraph(); flushList(); flushOl(); blocks.push('<h3>' + inlineFormat(h2[1]) + '</h3>'); return; }

        // Unordered list: * item or - item
        const bullet = trimmed.match(/^[\\*\\-]\\s+(.+)$/);
        if (bullet) {
          flushParagraph();
          flushOl();
          listItems.push(bullet[1]);
          return;
        }

        // Ordered list: 1. item
        const ordered = trimmed.match(/^\\d+\\.\\s+(.+)$/);
        if (ordered) {
          flushParagraph();
          flushList();
          olItems.push(ordered[1]);
          return;
        }

        flushList();
        flushOl();
        paragraph.push(trimmed);
      });

      if (tableRows.length > 0) flushTable();
      flushParagraph();
      flushList();
      flushOl();
      return blocks.join('');
    }

    // ─── Table rendering ───────────────────────────────────────────────────────
    function renderTable(columns, rows) {
      if (!rows || !rows.length) {
        tableWrap.innerHTML = '<div class="panel-body empty-state">No rows to display.</div>';
        return;
      }
      const cols = columns && columns.length ? columns : Object.keys(rows[0]);
      const thead = '<thead><tr>' + cols.map(c => '<th>' + escapeHtml(c) + '</th>').join('') + '</tr></thead>';
      const tbody = '<tbody>' + rows.map(row => {
        return '<tr>' + cols.map(c => '<td title="' + escapeHtml(formatCell(row[c])) + '">' + escapeHtml(formatCell(row[c])) + '</td>').join('') + '</tr>';
      }).join('') + '</tbody>';
      tableWrap.innerHTML = '<table>' + thead + tbody + '</table>';
    }

    function formatCell(value) {
      if (value === null || value === undefined) return '';
      if (typeof value === 'object') return JSON.stringify(value);
      return String(value);
    }

    // ─── Result update ─────────────────────────────────────────────────────────
    function updateResult(data) {
      currentResult = data;
      const rows = data.rows || [];
      const columns = data.columns || [];
      const answerText = data.answer || (data.error ? 'Could not generate a natural language answer.' : 'The query did not return a textual answer.');
      answerEl.innerHTML = renderAnswerMarkdown(answerText);
      answerEl.classList.toggle('empty', !data.answer);

      // Format and highlight SQL
      const rawSql = data.sql || '(no SQL generated)';
      if (data.sql) {
        const formatted = formatSql(rawSql);
        sqlEl.innerHTML = highlightSql(escapeHtml(formatted));
      } else {
        sqlEl.textContent = rawSql;
      }

      metrics.rows.textContent = String(data.row_count ?? rows.length ?? 0);
      metrics.cols.textContent = String(columns.length || (rows[0] ? Object.keys(rows[0]).length : 0));
      metrics.trunc.textContent = data.truncated ? 'Yes' : 'No';
      metrics.time.textContent = data.duration_ms ? data.duration_ms + ' ms' : '-';
      renderTable(columns, rows);
      showError(data.error || '');
    }

    // ─── CSV/JSON export ───────────────────────────────────────────────────────
    function toCsv(columns, rows) {
      const cols = columns && columns.length ? columns : (rows[0] ? Object.keys(rows[0]) : []);
      const escapeCsv = value => {
        const raw = formatCell(value);
        return /[",\\n\\r]/.test(raw) ? '"' + raw.replace(/"/g, '""') + '"' : raw;
      };
      return [cols.join(','), ...rows.map(row => cols.map(c => escapeCsv(row[c])).join(','))].join('\\n');
    }

    function download(filename, content, type) {
      const blob = new Blob([content], { type });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    // ─── Form submit ───────────────────────────────────────────────────────────
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (!isConnected) {
        showError('Please connect to the database first.');
        return;
      }

      const question = questionEl.value.trim();
      if (!question) {
        showError('Please enter a question before submitting.');
        questionEl.focus();
        return;
      }

      const currentApiKey = apiKeyEl.value.trim();
      if (!currentApiKey) {
        showError('LLM API key is required.');
        apiKeyEl.focus();
        return;
      }

      runButton.disabled = true;
      setStatus('Consulting', 'busy');
      showError('');

      try {
        const response = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question,
            model: modelEl.value.trim(),
            max_rows: 50000,
            db_host: dbHost,
            db_port: dbPort,
            db_user: dbUser,
            db_pass: dbPass,
            api_key: currentApiKey
          })
        });
        const data = await response.json();
        updateResult(data);
        setStatus(response.ok && !data.error ? 'Ready' : 'Warning', response.ok && !data.error ? 'connected' : 'error');
      } catch (error) {
        showError(error.message || String(error));
        setStatus('Error', 'error');
      } finally {
        runButton.disabled = false;
      }
    });

    // ─── Other events ──────────────────────────────────────────────────────────
    clearButton.addEventListener('click', () => {
      questionEl.value = '';
      showError('');
      questionEl.focus();
    });

    document.querySelectorAll('.example').forEach(button => {
      button.addEventListener('click', () => {
        questionEl.value = button.textContent.trim();
        questionEl.focus();
      });
    });

    document.getElementById('copySql').addEventListener('click', async () => {
      await navigator.clipboard.writeText(currentResult.sql || '');
      setStatus('SQL copied', 'connected');
      setTimeout(() => setStatus('Ready', 'connected'), 1300);
    });

    document.getElementById('downloadJson').addEventListener('click', () => {
      download('drive_qa_results.json', JSON.stringify(currentResult.rows || [], null, 2), 'application/json');
    });

    document.getElementById('downloadCsv').addEventListener('click', () => {
      download('drive_qa_results.csv', toCsv(currentResult.columns || [], currentResult.rows || []), 'text/csv');
    });
  </script>
</body>
</html>
"""


# ─── HTTP helpers ───────────────────────────────────────────────────────────────

def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'")
    handler.end_headers()
    handler.wfile.write(body)


def _sanitize_error(error_msg: str) -> str:
    """Remove potential credentials from error messages."""
    # Strip anything that looks like a connection string password
    sanitized = re.sub(r"://[^@]*@", "://***:***@", str(error_msg))
    return sanitized


# ─── Pipeline cache ─────────────────────────────────────────────────────────────

class PipelineCache:
    """Cache pipelines by (db_url_hash, api_key_hash, model) to avoid recreation."""

    def __init__(self, log_level: str):
        self.log_level = log_level
        self._lock = Lock()
        self._pipelines: Dict[Tuple[str, str, str], DriveQAPipeline] = {}

    def get(self, db_url: str, api_key: str, model: str) -> DriveQAPipeline:
        # Use hashes as cache keys to avoid storing raw credentials in dict keys
        db_hash = hashlib.sha256(db_url.encode()).hexdigest()[:16]
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        cache_key = (db_hash, key_hash, model)

        with self._lock:
            pipeline = self._pipelines.get(cache_key)
            if pipeline is None:
                pipeline = create_pipeline(
                    db_url=db_url,
                    model_id=model,
                    log_level=self.log_level,
                    api_key=api_key,
                )
                self._pipelines[cache_key] = pipeline
            return pipeline


# ─── Configuration validation ───────────────────────────────────────────────────

def _build_db_url(db_host: str, db_port: int, db_user: str, db_pass: str) -> str:
    """Build the full SQLAlchemy DB URL from user-provided connection parameters."""
    from urllib.parse import quote_plus
    return f"{DB_DRIVER}://{quote_plus(db_user)}:{quote_plus(db_pass)}@{db_host}:{db_port}/{DB_SCHEMA}"


def _validate_db_connection(db_url: str) -> str | None:
    """Try connecting to the database. Returns error message or None on success."""
    from sqlalchemy import create_engine, text

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return None
    except Exception as exc:
        return _sanitize_error(str(exc))


def _validate_api_key(api_key: str, model: str) -> str | None:
    """Validate an API key. For Gemini, checks via models.list(). For Azure, deferred."""
    if is_gemini_model(model):
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=10000),
            )
            next(iter(client.models.list()), None)
            return None
        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if "401" in msg or "UNAUTHENTICATED" in msg or "API_KEY_INVALID" in msg:
                return "Invalid API key. Please check your Gemini API key."
            if "403" in msg or "PERMISSION_DENIED" in msg:
                return "API key does not have sufficient permissions."
            if "timeout" in msg_lower or "timed out" in msg_lower:
                return (
                    "Gemini API validation timed out. Check your internet connection, "
                    "VPN/proxy/firewall settings, or try an Azure model."
                )
            return f"API key validation failed: {_sanitize_error(msg)}"
    # For Azure models, validation is deferred to the first request
    return None


# ─── Request handler factory ────────────────────────────────────────────────────

def make_handler(cache: PipelineCache, default_model: str) -> type[BaseHTTPRequestHandler]:
    model_options = "".join(
        f'<option value="{spec.model_id}"{" selected" if spec.model_id == default_model else ""}>{spec.display_name}</option>'
        for spec in list_models()
    )
    html = INDEX_HTML.replace("__MODEL_OPTIONS__", model_options)

    class DriveQAWebHandler(BaseHTTPRequestHandler):
        server_version = "DriveQAWeb/1.0"

        def log_message(self, format: str, *args: Any) -> None:
            # Never log request bodies (which may contain credentials)
            msg = format % args
            # Extra safety: strip anything that looks like credentials
            msg = re.sub(r"://[^@]*@", "://***:***@", msg)
            logging.getLogger("drive_qa.web").info("%s - %s", self.address_string(), msg)

        def _read_json_body(self) -> Dict[str, Any] | None:
            """Read and parse JSON body. Returns None and sends error on failure."""
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length > 1_000_000:
                    _json_response(self, HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Request too large"})
                    return None
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                return payload
            except Exception as exc:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON payload: {exc}"})
                return None

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                _html_response(self, html)
                return
            if self.path == "/health":
                _json_response(self, HTTPStatus.OK, {"ok": True, "default_model": default_model})
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:
            if self.path == "/api/configure":
                self._handle_configure()
            elif self.path == "/api/ask":
                self._handle_ask()
            else:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def _handle_configure(self) -> None:
            payload = self._read_json_body()
            if payload is None:
                return

            db_host = str(payload.get("db_host", "")).strip()
            db_port_raw = payload.get("db_port")
            db_user = str(payload.get("db_user", "")).strip()
            db_pass = str(payload.get("db_pass", "")).strip()

            if not db_host:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Database host is required."})
                return
            try:
                db_port = int(db_port_raw)
                if not (1 <= db_port <= 65535):
                    raise ValueError
            except (TypeError, ValueError):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Database port must be a number between 1 and 65535."})
                return
            if not db_user:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Database user is required."})
                return
            if not db_pass:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Database password is required."})
                return

            db_url = _build_db_url(db_host, db_port, db_user, db_pass)

            # Validate DB connection
            db_error = _validate_db_connection(db_url)
            if db_error:
                _json_response(self, HTTPStatus.OK, {"ok": False, "error": f"Database connection failed: {db_error}"})
                return

            _json_response(self, HTTPStatus.OK, {"ok": True})

        def _handle_ask(self) -> None:
            payload = self._read_json_body()
            if payload is None:
                return

            question = str(payload.get("question", "")).strip()
            model = str(payload.get("model", "")).strip() or default_model
            max_rows = _coerce_max_rows(payload.get("max_rows", 50))
            db_host = str(payload.get("db_host", "")).strip()
            db_port_raw = payload.get("db_port")
            db_user = str(payload.get("db_user", "")).strip()
            db_pass = str(payload.get("db_pass", "")).strip()
            api_key_val = str(payload.get("api_key", "")).strip()

            if not question:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Question is required."})
                return
            if not db_host or not db_user or not db_pass or not api_key_val:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Database credentials and API key are required. Please configure the connection first."})
                return
            try:
                db_port = int(db_port_raw)
                if not (1 <= db_port <= 65535):
                    raise ValueError
            except (TypeError, ValueError):
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Database port must be a number between 1 and 65535."})
                return

            db_url = _build_db_url(db_host, db_port, db_user, db_pass)

            started = time.perf_counter()
            try:
                pipeline = cache.get(db_url=db_url, api_key=api_key_val, model=model)
                result = pipeline.answer(question=question, max_answer_rows=max_rows)
                # Sanitize any potential credential leakage in error messages
                if result.get("error"):
                    result["error"] = _sanitize_error(result["error"])
                result["duration_ms"] = round((time.perf_counter() - started) * 1000)
                result["model"] = model
                result["max_rows"] = max_rows
                # Never return credentials
                result.pop("db_url", None)
                result.pop("api_key", None)
                _json_response(self, HTTPStatus.OK, result)
            except Exception as exc:
                logging.getLogger("drive_qa.web").error("Request failed: %s", _sanitize_error(str(exc)))
                _json_response(
                    self,
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "question": question,
                        "retrieved_tables": [],
                        "sql": None,
                        "valid": False,
                        "rows": [],
                        "columns": [],
                        "row_count": 0,
                        "truncated": False,
                        "answer": None,
                        "error": _sanitize_error(str(exc)),
                        "duration_ms": round((time.perf_counter() - started) * 1000),
                        "model": model,
                        "max_rows": max_rows,
                    },
                )

    return DriveQAWebHandler


def _coerce_max_rows(value: Any) -> int:
    try:
        max_rows = int(value)
    except (TypeError, ValueError):
        return 50000
    return max(1, min(max_rows, 50000))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local web UI for DRIVE QA.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8502, help="Port to bind.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Default Gemini model.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cache = PipelineCache(log_level=args.log_level)
    handler = make_handler(cache=cache, default_model=args.model)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"DRIVE QA web UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
