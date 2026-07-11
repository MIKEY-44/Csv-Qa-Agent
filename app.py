"""
CSV QA Agent - Complete Fixed Version
======================================
Serves the exact dark UI from user screenshots.
Windows-compatible: No SIGALRM.
"""
import os
import sys
import uuid
import json
import re
import ast
import io
import contextlib
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

PROJECT_ROOT = Path(__file__).parent.absolute()
UPLOAD_DIR = PROJECT_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ==================== DATA MODELS ====================
from dataclasses import dataclass, field

@dataclass
class AgentResponse:
    answer: str
    confidence: float = 0.95
    viz_type: str = ""
    data: Any = None
    execution_trace: list = field(default_factory=list)
    code: str = ""
    mode: str = "rule-based"
    latency_ms: float = 0.0

# ==================== SANDBOX (Windows-compatible) ====================
class SecureSandbox:
    ALLOWED_BUILTINS = {
        'len', 'range', 'round', 'sum', 'min', 'max', 'abs', 'float', 'int', 'str',
        'dict', 'list', 'tuple', 'set', 'sorted', 'zip', 'enumerate', 'map', 'filter',
        'bool', 'type', 'isinstance', 'hasattr', 'getattr', 'print', 'ord', 'chr',
        'hex', 'bin', 'oct', 'pow', 'divmod', 'all', 'any', 'reversed', 'slice'
    }

    def __init__(self, df, timeout: float = 5.0):
        self.df = df.copy()
        self.timeout = timeout
        self._compile_builtins()

    def _compile_builtins(self):
        import builtins
        safe = {}
        for name in self.ALLOWED_BUILTINS:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)
        for exc_name in ['Exception', 'ValueError', 'TypeError', 'KeyError', 
                          'IndexError', 'AttributeError', 'ZeroDivisionError',
                          'RuntimeError', 'StopIteration', 'NotImplementedError']:
            if hasattr(builtins, exc_name):
                safe[exc_name] = getattr(builtins, exc_name)
        self._safe_builtins = safe

    def _validate_ast(self, code: str):
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Imports are not allowed in sandbox"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('open', 'exec', 'eval', 'compile', '__import__'):
                        return False, f"'{node.func.id}' is not allowed"
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ('__subclasses__', '__bases__', '__globals__',
                                         'mro', 'func_globals', 'gi_frame', 'f_locals',
                                         'f_globals'):
                        return False, f"Attribute '{node.func.attr}' is blocked"
            if isinstance(node, ast.Delete):
                return False, "Delete statements are not allowed"
        return True, ""

    def _run_with_timeout(self, code: str, globals_dict: dict, locals_dict: dict):
        result_container = [None]
        exception_container = [None]
        stdout_buffer = io.StringIO()

        def target():
            try:
                with contextlib.redirect_stdout(stdout_buffer):
                    exec(code, globals_dict, locals_dict)
                result_container[0] = locals_dict.get('result')
            except Exception as e:
                exception_container[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)

        if thread.is_alive():
            return False, None, f"Execution timed out after {self.timeout}s"
        if exception_container[0] is not None:
            return False, None, f"{type(exception_container[0]).__name__}: {exception_container[0]}"
        if result_container[0] is None:
            return False, None, "'result' variable was not set"
        return True, result_container[0], stdout_buffer.getvalue()

    def run(self, code: str):
        valid, msg = self._validate_ast(code)
        if not valid:
            return False, None, msg
        safe_globals = {
            "pd": pd, "np": np, "df": self.df, "datetime": datetime,
            "__builtins__": self._safe_builtins
        }
        safe_locals = {}
        return self._run_with_timeout(code, safe_globals, safe_locals)

# ==================== RULE AGENT ====================
class RuleAgent:
    def __init__(self, df):
        self.df = df
        self.sandbox = SecureSandbox(df, timeout=5.0)

    def answer(self, question: str) -> Tuple[str, str, Any, str]:
        q = question.lower()

        def has_any(words): return any(w in q for w in words)
        def has_all(words): return all(w in q for w in words)

        code = ""

        if has_any(['total revenue', 'total sales', 'sum of revenue', 'all revenue', 'overall revenue', 'revenue total']):
            code = """df['revenue'] = df['units'] * df['unit_price']
total = df['revenue'].sum()
result = {'answer': f'Total revenue: ${total:,.2f}', 'data': {'total': round(float(total),2)}, 'viz_type': 'number'}"""

        elif has_all(['region']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max', 'biggest', 'largest']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        elif has_all(['product']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max', 'biggest', 'largest']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('product')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        elif has_all(['category']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        elif has_any(['sales rep', 'rep', 'representative']) and has_any(['highest', 'top', 'best', 'most', 'maximum', 'max']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('sales_rep')['revenue'].sum().sort_values(ascending=False)
result = {'answer': f'{grouped.index[0]} has the highest revenue at ${grouped.iloc[0]:,.2f}', 'data': {str(k): round(float(v), 2) for k, v in grouped.to_dict().items()}, 'viz_type': 'bar'}"""

        elif has_any(['average', 'mean', 'avg']) and has_any(['segment', 'customer_segment']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('customer_segment')['revenue'].mean().round(2)
result = {'answer': 'Average revenue by segment: ' + ', '.join([f"{k}=${v}" for k,v in grouped.items()]), 'data': {str(k): float(v) for k,v in grouped.to_dict().items()}, 'viz_type': 'table'}"""

        elif has_any(['average', 'mean', 'avg']) and has_all(['region']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('region')['revenue'].mean().round(2)
result = {'answer': 'Average revenue by region: ' + ', '.join([f"{k}=${v}" for k,v in grouped.items()]), 'data': {str(k): float(v) for k,v in grouped.to_dict().items()}, 'viz_type': 'table'}"""

        elif has_any(['average', 'mean', 'avg']) and has_all(['product']):
            code = """df['revenue'] = df['units'] * df['unit_price']
grouped = df.groupby('product')['revenue'].mean().round(2)
result = {'answer': 'Average revenue by product: ' + ', '.join([f"{k}=${v}" for k,v in grouped.items()]), 'data': {str(k): float(v) for k,v in grouped.to_dict().items()}, 'viz_type': 'table'}"""

        elif has_any(['growth', 'mom', 'month over month', 'monthly', 'month to month', 'trend']):
            code = """df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.to_period('M')
df['revenue'] = df['units'] * df['unit_price']
monthly = df.groupby('month')['revenue'].sum()
growth = monthly.pct_change() * 100
g = {str(k): round(float(v),1) for k,v in growth.dropna().items()}
result = {'answer': 'Month-over-month growth: ' + ', '.join([f"{k}={v}%" for k,v in g.items()]), 'data': g, 'viz_type': 'line'}"""

        elif has_any(['compare', 'vs', 'versus', 'pivot', 'breakdown', 'cross', 'by region and', 'by segment and']):
            code = """df['revenue'] = df['units'] * df['unit_price']
pivot = df.pivot_table(values='revenue', index='region', columns='customer_segment', aggfunc='sum', fill_value=0)
result = {'answer': 'Enterprise vs SMB sales by region', 'data': {str(k): {str(k2): float(v2) for k2, v2 in v.items()} for k, v in pivot.to_dict().items()}, 'viz_type': 'table'}"""

        elif has_any(['correlation', 'correlate', 'relationship between']):
            code = """corr = df['units'].corr(df['unit_price'])
result = {'answer': f'Correlation between units and unit_price: {corr:.4f}', 'data': {'correlation': round(float(corr),4)}, 'viz_type': 'number'}"""

        elif has_any(['how many', 'count', 'unique', 'number of', 'distinct']):
            code = """counts = df.groupby('sales_rep')['customer_segment'].nunique()
result = {'answer': 'Unique customer segments handled by each rep', 'data': {str(k): int(v) for k,v in counts.to_dict().items()}, 'viz_type': 'table'}"""

        else:
            code = """df['revenue'] = df['units'] * df['unit_price']
total = df['revenue'].sum()
result = {'answer': f'Dataset: {len(df)} rows. Total revenue: ${total:,.2f}. Try: total revenue, top region, avg by segment, growth, compare, correlation', 'data': {'total': round(float(total),2)}, 'viz_type': 'number'}"""

        success, result, output = self.sandbox.run(code)
        if success:
            return result.get('answer', ''), code, result.get('data'), result.get('viz_type', '')
        else:
            return f"Error: {output}", code, None, ''

# ==================== LLM AGENT ====================
class LLMAgent:
    def __init__(self, df, api_key):
        self.df = df
        self.schema = self._get_schema()
        self.sandbox = SecureSandbox(df, timeout=5.0)
        self.client = None

        if api_key:
            try:
                os.environ["OPENAI_TIMEOUT"] = "30"
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.client.models.list()
            except Exception as e:
                print(f"OpenAI init failed: {e}")
                self.client = None

    def _get_schema(self):
        cols = []
        for c in self.df.columns:
            cols.append({"name": c, "dtype": str(self.df[c].dtype), "unique": self.df[c].nunique()})
        return {"columns": cols, "shape": self.df.shape}

    def answer(self, question: str) -> Tuple[str, str, Any, str]:
        if not self.client:
            raise Exception("OpenAI not available")

        schema_str = json.dumps(self.schema, default=str)
        prompt = f"""Generate Python/Pandas code to answer: "{question}"

Dataset: {schema_str}
Rules:
1. Write ONLY Python code. No markdown, no explanations.
2. df is pre-loaded. Store answer in `result` = {{'answer': str, 'data': any, 'viz_type': 'table'|'bar'|'line'|'number'|None}}
3. Compute actual values. No hardcoded numbers.
4. No imports, no file I/O.
5. Calculate revenue as df['revenue'] = df['units'] * df['unit_price'] first, then use df['revenue'] for groupby operations.

Example: result = {{'answer': 'Total: $45,230', 'data': {{'total': 45230}}, 'viz_type': 'number'}}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800
        )

        code = response.choices[0].message.content.strip()
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'^```\s*', '', code)
        code = re.sub(r'```\s*$', '', code)

        success, result, output = self.sandbox.run(code)
        if success:
            return result.get('answer', ''), code, result.get('data'), result.get('viz_type', '')
        else:
            raise Exception(output)

# ==================== MAIN AGENT ====================
class CSVQAAgent:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
        self.schema = {
            "columns": [{"name": c, "dtype": str(self.df[c].dtype), "unique": self.df[c].nunique()} for c in self.df.columns],
            "shape": self.df.shape
        }
        self.rule_agent = RuleAgent(self.df)
        self.llm_agent = None

        if API_KEY:
            try:
                self.llm_agent = LLMAgent(self.df, API_KEY)
            except:
                pass

    def ask(self, question: str) -> Dict:
        start = datetime.now()

        if self.llm_agent:
            try:
                answer, code, data, viz = self.llm_agent.answer(question)
                return {"success": True, "answer": answer, "code": code, "data": data, "viz_type": viz,
                        "execution_time": (datetime.now() - start).total_seconds(), "mode": "llm"}
            except Exception as e:
                print(f"LLM failed, using fallback: {e}")

        answer, code, data, viz = self.rule_agent.answer(question)
        return {"success": True, "answer": answer, "code": code, "data": data, "viz_type": viz,
                "execution_time": (datetime.now() - start).total_seconds(), "mode": "rule-based"}

    def answer(self, question: str) -> AgentResponse:
        start = datetime.now()
        trace = []

        if self.llm_agent:
            try:
                trace.append("Attempting LLM generation...")
                answer, code, data, viz = self.llm_agent.answer(question)
                trace.append("LLM code executed successfully")
                return AgentResponse(
                    answer=answer, confidence=0.95, viz_type=viz, data=data,
                    execution_trace=trace, code=code, mode="llm",
                    latency_ms=round((datetime.now() - start).total_seconds() * 1000, 2)
                )
            except Exception as e:
                trace.append(f"LLM failed: {e}")

        trace.append("Using rule-based fallback...")
        answer, code, data, viz = self.rule_agent.answer(question)
        trace.append("Rule-based code executed successfully")
        return AgentResponse(
            answer=answer, confidence=0.90, viz_type=viz, data=data,
            execution_trace=trace, code=code, mode="rule-based",
            latency_ms=round((datetime.now() - start).total_seconds() * 1000, 2)
        )

# ==================== FASTAPI APP ====================
sessions = {}

default_agent = None
default_csv = DATA_DIR / "sales.csv"
if default_csv.exists():
    try:
        default_agent = CSVQAAgent(str(default_csv))
        print(f"Loaded default CSV: {default_csv} ({default_agent.df.shape[0]} rows)")
    except Exception as e:
        print(f"Could not load default CSV: {e}")

app = FastAPI(title="CSV AI Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load HTML from file or use inline
HTML_CONTENT = ""
html_path = PROJECT_ROOT / "index.html"
if html_path.exists():
    with open(html_path, "r", encoding="utf-8") as f:
        HTML_CONTENT = f.read()
else:
    HTML_CONTENT = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>CSV QA Agent</title></head><body><h1>CSV QA Agent</h1><p>Place index.html next to app.py</p></body></html>"

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(HTML_CONTENT)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "Only CSV/Excel files supported")

    sid = str(uuid.uuid4())
    path = UPLOAD_DIR / f"{sid}_{file.filename}"

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    try:
        agent = CSVQAAgent(str(path))
        sessions[sid] = {"agent": agent, "filename": file.filename}
        return {"success": True, "session_id": sid, "filename": file.filename, "schema": agent.schema}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/api/ask")
async def api_ask(question: str = Form(...), session_id: str = Form(default=None)):
    import time
    start = time.time()

    active_agent = default_agent

    # Use provided session_id if valid
    if session_id and session_id in sessions:
        active_agent = sessions[session_id]["agent"]
    elif sessions:
        # Fallback to latest session
        latest_sid = list(sessions.keys())[-1]
        active_agent = sessions[latest_sid]["agent"]

    if active_agent is None:
        return JSONResponse({"success": False, "error": "No CSV loaded. Please upload a file first."}, status_code=400)

    try:
        response = active_agent.answer(question)
        return JSONResponse({
            "success": True,
            "answer": response.answer,
            "confidence": response.confidence,
            "viz_type": response.viz_type,
            "data": response.data,
            "trace": response.execution_trace,
            "mode": response.mode,
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/health")
def health():
    return {"status": "ok", "default_agent": default_agent is not None, "sessions": len(sessions), "api_key": bool(API_KEY)}

if __name__ == "__main__":
    print("="*60)
    print("CSV QA Agent - Fixed (No SIGALRM)")
    print("="*60)
    print(f"Default agent: {default_agent is not None}")
    print(f"HTML file: {html_path.exists()}")
    print("Open http://localhost:8000")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)