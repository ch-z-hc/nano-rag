"""nano-rag CLI: ingest documents, query knowledge base, serve via HTTP or WeChat."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from aiohttp import web

from rag.ingest import ingest_directory, ingest_file
from rag.query import ask, retrieve, reset_bm25_index
from rag.store import reset_store


# ============================================================
# CLI commands
# ============================================================


def cmd_ingest(args):
    path = Path(args.path)
    if not path.exists():
        print(f"路径不存在: {path}")
        return

    if path.is_file():
        print(f"Ingesting single file: {path.name}")
        n = ingest_file(path)
        print(f"Done. {n} chunks added.")
    else:
        print(f"Ingesting directory: {path}")
        files, chunks = ingest_directory(path, recursive=not args.no_recursive)
        print(f"Done. {files} files -> {chunks} chunks added.")

    # Reset BM25 index so next query rebuilds it with new documents
    reset_bm25_index()


def cmd_query(args):
    question = args.question
    print(f"检索中: {question}")
    print("-" * 50)

    if args.retrieve_only:
        chunks = retrieve(question, args.top_k or None)
        if not chunks:
            print("未找到相关内容")
        for i, c in enumerate(chunks):
            src = c["metadata"].get("source_name", "?")
            rrf = c.get("rrf_score")
            dist = c.get("distance")
            score_info = f"rrf={rrf:.4f}" if rrf else f"distance={dist:.4f}" if dist else ""
            has_parent = "parent_content" in c.get("metadata", {})
            parent_tag = " [parent-child]" if has_parent else ""
            print(f"\n--- [{i+1}] {src} ({score_info}){parent_tag} ---")
            print(c["content"][:300])
    else:
        answer = ask(question, args.top_k or None)
        print(answer)


def cmd_reset(args):
    reset_store()


def cmd_wechat_login(args):
    asyncio.run(_wechat_login())


async def _wechat_login():
    from channels.wechat import login_wechat
    await login_wechat()


def cmd_serve(args):
    """Start HTTP API server for RAG queries (+ WeChat bot if session exists)."""
    host = args.host
    port = args.port

    # Eager-init the vector store (loads embedding model) before starting
    # the event loop, otherwise it blocks aiohttp handlers.
    from rag.store import get_store
    print("Loading embedding model...")
    get_store()
    print("Ready.")

    app = web.Application()
    app.add_routes([
        web.get("/", _root_handler),
        web.get("/api/health", _health_handler),
        web.get("/api/kb", _kb_list_handler),
        web.delete("/api/kb/{hash}", _kb_delete_handler),
        web.post("/api/query", _query_handler),
        web.post("/api/ingest", _ingest_handler),
    ])

    # Start WeChat bot as background task if a valid session exists
    _wechat_task: asyncio.Task | None = None

    async def _start_wechat(app):
        nonlocal _wechat_task
        from channels.wechat import run_wechat_bot, login_wechat
        from rag.query import ask as rag_ask
        from channels.wechat import _load_session

        session = _load_session()
        if not session.get("token"):
            print("[wechat] No saved session, starting login...")
            await login_wechat()
            session = _load_session()
            if not session.get("token"):
                print("[wechat] Login failed or cancelled, skipping WeChat bot.")
                return

        async def handle(from_user, content, msg_id):
            print(f"[wx] {from_user}: {content}")
            try:
                answer = await asyncio.to_thread(rag_ask, content)
                return answer
            except Exception as e:
                return f"出错了: {e}"

        _wechat_task = asyncio.create_task(run_wechat_bot(handle))
        print("[wechat] Bot started as background task")

    async def _stop_wechat(app):
        if _wechat_task:
            _wechat_task.cancel()
            try:
                await _wechat_task
            except asyncio.CancelledError:
                pass

    app.on_startup.append(_start_wechat)
    app.on_cleanup.append(_stop_wechat)

    print(f"Starting nano-rag on http://{host}:{port}")
    print(f"  Web UI + API       http://{host}:{port}")
    print(f"  WeChat auto-start  有 session 时自动启动")
    web.run_app(app, host=host, port=port)


SEARCH_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nano-rag · 本地知识库</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.21/dist/contrib/auto-render.min.js"></script>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font:15px/1.6 system-ui,-apple-system,sans-serif;background:#f8f9fa;color:#1a1a2e;min-height:100vh}
  .topbar{background:#fff;border-bottom:1px solid #e5e7eb;padding:0 24px;display:flex;align-items:center;height:56px;gap:24px;position:sticky;top:0;z-index:10}
  .topbar h1{font-size:18px;font-weight:700;display:flex;align-items:center;gap:8px;white-space:nowrap}
  .topbar h1 .dot{width:8px;height:8px;border-radius:50%;background:#22c55e}
  .topbar nav{display:flex;gap:4px}
  .topbar nav button{padding:8px 16px;border:none;background:none;font-size:14px;cursor:pointer;border-radius:6px;color:#666;font-weight:500;transition:all .15s}
  .topbar nav button.active{color:#2563eb;background:#eff6ff}
  .topbar nav button:hover{color:#2563eb;background:#f5f5f5}
  main{max-width:860px;margin:0 auto;padding:24px}
  .tab{display:none}
  .tab.active{display:block}

  /* Search */
  .search-box{display:flex;gap:10px;margin-bottom:20px}
  .search-box input{flex:1;padding:12px 16px;border:1px solid #d4d4d8;border-radius:10px;font-size:15px;outline:none;transition:border-color .15s,box-shadow .15s}
  .search-box input:focus{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.12)}
  .search-box button{padding:12px 22px;background:#2563eb;color:#fff;border:none;border-radius:10px;font-size:15px;cursor:pointer;font-weight:600;transition:background .15s;white-space:nowrap}
  .search-box button:hover{background:#1d4ed8}
  .search-box button:disabled{opacity:.5;cursor:default}
  .status{color:#888;font-size:13px;margin-bottom:16px;min-height:20px}
  .result-card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:22px 26px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
  .result-card .answer{line-height:1.75;font-size:15px}
  .result-card .answer p{margin:6px 0}
  .result-card .answer code{background:#f0f0f0;padding:2px 6px;border-radius:4px;font-size:13px;font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
  .result-card .answer pre{background:#1e1e2e;color:#cdd6f4;padding:14px 18px;border-radius:8px;overflow-x:auto;margin:10px 0;font-size:13px;line-height:1.5}
  .result-card .answer pre code{background:none;padding:0;color:inherit;font-size:inherit}
  .result-card .answer ul,.result-card .answer ol{padding-left:22px;margin:6px 0}
  .result-card .answer li{margin:2px 0}
  .result-card .answer h1,.result-card .answer h2,.result-card .answer h3{font-size:16px;font-weight:600;margin:10px 0 4px}
  .result-card .answer table{border-collapse:collapse;width:100%;margin:8px 0}
  .result-card .answer td,.result-card .answer th{border:1px solid #e5e7eb;padding:6px 10px;font-size:13px}
  .result-card .answer th{background:#f5f5f5;font-weight:600}
  .result-card .answer blockquote{border-left:3px solid #2563eb;margin:8px 0;padding:4px 14px;color:#666;background:#f8faff}
  .result-card .sources{margin-top:16px;padding-top:14px;border-top:1px solid #f0f0f0;font-size:12px;color:#888;display:flex;align-items:center;gap:6px}
  .result-card .sources .icon{opacity:.5}
  .error{color:#dc2626;background:#fef2f2;border:1px solid #fecaca;padding:16px 20px;border-radius:10px;font-size:14px}
  .spinner{display:inline-block;width:16px;height:16px;border:2px solid #e5e5e5;border-top-color:#2563eb;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* KB tab */
  .drop-zone{background:#fff;border:2px dashed #d4d4d8;border-radius:12px;padding:40px 20px;text-align:center;cursor:pointer;transition:all .15s;margin-bottom:24px}
  .drop-zone:hover,.drop-zone.drag{border-color:#2563eb;background:#f8faff}
  .drop-zone .icon{font-size:36px;margin-bottom:10px;opacity:.6}
  .drop-zone p{color:#888;font-size:14px}
  .drop-zone .hint{font-size:12px;color:#aaa;margin-top:4px}
  .drop-zone input{display:none}
  .upload-status{margin-bottom:16px;min-height:24px}
  .upload-status .file-progress{margin-bottom:8px}
  .upload-status .file-progress .fname{font-size:13px;color:#555;margin-bottom:4px;display:flex;justify-content:space-between}
  .upload-status .file-progress .fname .pct{font-weight:600;color:#2563eb}
  .progress-bar{height:6px;background:#e5e7eb;border-radius:3px;overflow:hidden}
  .progress-bar .fill{height:100%;background:#2563eb;border-radius:3px;transition:width .2s;width:0}
  .progress-bar .fill.done{background:#22c55e}
  .progress-bar .fill.fail{background:#ef4444}
  .progress-bar .fill.indeterminate{width:30%!important;background:linear-gradient(90deg,#2563eb,#a78bfa);animation:slideBar 1.2s ease-in-out infinite}
  @keyframes slideBar{0%{transform:translateX(0)}50%{transform:translateX(233%)}100%{transform:translateX(0)}}
  .file-list{display:flex;flex-direction:column;gap:8px}
  .file-card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:14px;transition:box-shadow .15s}
  .file-card:hover{box-shadow:0 2px 8px rgba(0,0,0,.06)}
  .file-card .file-icon{width:40px;height:40px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
  .file-card .file-icon.pdf{background:#fef2f2;color:#ef4444}
  .file-card .file-icon.md{background:#f0fdf4;color:#16a34a}
  .file-card .file-icon.txt{background:#f5f5f5;color:#737373}
  .file-card .file-icon.code{background:#eff6ff;color:#2563eb}
  .file-card .file-info{flex:1;min-width:0}
  .file-card .file-name{font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .file-card .file-meta{font-size:12px;color:#999;margin-top:2px}
  .file-card .file-chunks{font-size:12px;color:#666;background:#f5f5f5;padding:4px 10px;border-radius:20px;white-space:nowrap}
  .file-card .del-btn{background:none;border:none;color:#ccc;cursor:pointer;padding:6px 8px;border-radius:6px;font-size:16px;transition:all .15s;flex-shrink:0}
  .file-card .del-btn:hover{color:#ef4444;background:#fef2f2}
  .empty-state{text-align:center;padding:48px 20px;color:#aaa}
  .empty-state .icon{font-size:40px;margin-bottom:12px;opacity:.5}
  .empty-state p{font-size:14px}

  .toast{position:fixed;bottom:24px;right:24px;background:#1a1a2e;color:#fff;padding:12px 20px;border-radius:10px;font-size:14px;z-index:100;animation:fadeIn .2s;box-shadow:0 4px 12px rgba(0,0,0,.15)}
  @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<div class="topbar">
  <h1><span class="dot"></span>nano-rag</h1>
  <nav>
    <button id="tab-search" class="active" onclick="switchTab('search')">搜索</button>
    <button id="tab-kb" onclick="switchTab('kb');loadKB()">知识库</button>
  </nav>
</div>
<main>
  <!-- SEARCH TAB -->
  <div id="panel-search" class="tab active">
    <div class="search-box">
      <input id="q" type="text" placeholder="输入问题，搜索本地知识库…" autofocus>
      <button id="btn" onclick="search()">搜索</button>
    </div>
    <div id="status" class="status"></div>
    <div id="output"></div>
  </div>

  <!-- KNOWLEDGE BASE TAB -->
  <div id="panel-kb" class="tab">
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="icon">📁</div>
      <p>点击上传或拖拽文件到此处</p>
      <div class="hint">支持 PDF · Markdown · TXT · 代码文件</div>
      <input type="file" id="fileInput" multiple accept=".pdf,.md,.txt,.py,.js,.ts,.go,.rs,.java,.cpp,.c,.h,.json,.yaml,.yml,.toml,.xml,.html,.css,.sql,.sh">
    </div>
    <div class="upload-status" id="uploadStatus"></div>
    <div class="file-list" id="fileList"></div>
    <div class="empty-state" id="emptyState">
      <div class="icon">📚</div>
      <p>知识库为空，上传一些文档开始吧</p>
    </div>
  </div>
</main>
<div id="toast" class="toast" style="display:none"></div>

<script>
// ---- Tab switching ----
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  document.getElementById('panel-'+name).classList.add('active');
  document.getElementById('tab-'+name).classList.add('active');
}

// ---- Toast ----
let toastTimer;
function showToast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(()=>t.style.display='none', 2500);
}

// ---- Search ----
const inp = document.getElementById('q');
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const out = document.getElementById('output');
inp.addEventListener('keydown', e => { if (e.key==='Enter') search(); });

async function search() {
  const q = inp.value.trim();
  if (!q) return;
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>检索中…';
  out.innerHTML = '';
  try {
    const res = await fetch('/api/query', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const data = await res.json();
    if (data.error) { out.innerHTML = '<div class="error">'+esc(data.error)+'</div>'; status.textContent=''; }
    else { status.textContent = ''; renderAnswer(data.answer); }
  } catch(e) { out.innerHTML = '<div class="error">请求失败: '+esc(e.message)+'</div>'; status.textContent=''; }
  btn.disabled = false;
}

function renderAnswer(text) {
  const parts = text.split(/---\\s*参考来源/);
  const answer = parts[0];
  const sources = parts.length > 1 ? parts[1].trim() : '';

  // 1. Protect ALL LaTeX before marked touches it.
  //    Order matters: $$ before $, \[ before \(
  const mathBlocks = [];
  const PH = '%%MATH%%';
  let p = answer;

  // $$ display math $$
  p = p.replace(/\$\$([\s\S]*?)\$\$/g, (_, f) => {
    mathBlocks.push('$$' + f.trim() + '$$');
    return PH + (mathBlocks.length - 1) + '%%';
  });
  // \[ display math \]
  p = p.replace(/\\\[([\s\S]*?)\\\]/g, (_, f) => {
    mathBlocks.push('$$' + f.trim() + '$$');
    return PH + (mathBlocks.length - 1) + '%%';
  });
  // $ inline math $
  p = p.replace(/\$([^$\s][^$]*?)\$/g, (_, f) => {
    mathBlocks.push('$' + f.trim() + '$');
    return PH + (mathBlocks.length - 1) + '%%';
  });
  // \( inline math \)
  p = p.replace(/\\\(([\s\S]*?)\\\)/g, (_, f) => {
    mathBlocks.push('$' + f.trim() + '$');
    return PH + (mathBlocks.length - 1) + '%%';
  });

  // 2. CJK spacing (math is protected by placeholders)
  p = p
    .replace(/([一-鿿㐀-䶿])([a-zA-Z0-9])/g, '$1 $2')
    .replace(/([a-zA-Z0-9])([一-鿿㐀-䶿])/g, '$1 $2');

  // 3. Markdown
  const md = typeof marked !== 'undefined' ? marked.parse(p.trim()) : '<pre>'+esc(p.trim())+'</pre>';

  // 4. Restore math (just the raw LaTeX string, KaTeX handles it)
  let final = md;
  for (let i = 0; i < mathBlocks.length; i++) {
    final = final.replace(PH + i + '%%', esc(mathBlocks[i]));
  }

  let html = '<div class="result-card"><div class="answer">'+final+'</div>';
  if (sources) html += '<div class="sources"><span class="icon">📎</span> '+esc(sources)+'</div>';
  html += '</div>';
  out.innerHTML = html;

  // 5. KaTeX
  if (typeof renderMathInElement !== 'undefined') {
    try {
      renderMathInElement(out.querySelector('.answer'), {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
        ],
        throwOnError: false,
      });
    } catch(e) {}
  }
}
function esc(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

// ---- KB ----
let _filesCache = {};
async function loadKB() {
  try {
    const res = await fetch('/api/kb');
    const data = await res.json();
    const files = data.files||[];
    _filesCache = {};
    files.forEach(f => { if(f.hash) _filesCache[f.hash] = f; });
    renderKB(files);
  } catch(e) { console.error(e); }
}

function iconFor(name) {
  const ext = name.split('.').pop().toLowerCase();
  if (ext==='pdf') return {cls:'pdf', icon:'📕'};
  if (ext==='md') return {cls:'md', icon:'📝'};
  if (['py','js','ts','go','rs','java','cpp','c','h','sql','sh'].includes(ext)) return {cls:'code', icon:'💻'};
  return {cls:'txt', icon:'📄'};
}

function renderKB(files) {
  const list = document.getElementById('fileList');
  const empty = document.getElementById('emptyState');
  if (!files.length) { list.innerHTML=''; empty.style.display='block'; return; }
  empty.style.display='none';
  let html = '';
  for (const f of files) {
    const {cls, icon} = iconFor(f.name);
    const size = f.size ? (f.size<1024?f.size+'B':f.size<1048576?(f.size/1024).toFixed(1)+'KB':(f.size/1048576).toFixed(1)+'MB') : '';
    html += '<div class="file-card">'
      + '<div class="file-icon '+cls+'">'+icon+'</div>'
      + '<div class="file-info"><div class="file-name" title="'+esc(f.name)+'">'+esc(f.name)+'</div>'
      + '<div class="file-meta">'+(size?size+' · ':'')+esc(f.path||'')+'</div></div>'
      + '<div class="file-chunks">'+f.chunks+' 片段</div>'
      + '<button class="del-btn" title="删除" onclick="event.stopPropagation();delDoc(\''+esc(f.hash)+'\')">✕</button>'
      + '</div>';
  }
  list.innerHTML = html;
}

async function delDoc(hash) {
  const name = (_filesCache[hash]||{}).name || hash;
  if (!confirm('确定要从知识库中删除 "'+name+'" 吗？')) return;
  try {
    const res = await fetch('/api/kb/'+encodeURIComponent(hash), {method:'DELETE'});
    const data = await res.json();
    if (data.ok) { showToast('已删除: '+name); loadKB(); }
    else { showToast('删除失败: '+(data.error||'unknown')); }
  } catch(e) { showToast('请求失败: '+e.message); }
}

// ---- Upload ----
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');

dropZone.addEventListener('dragover', e=>{e.preventDefault();dropZone.classList.add('drag')});
dropZone.addEventListener('dragleave', ()=>{dropZone.classList.remove('drag')});
dropZone.addEventListener('drop', e=>{
  e.preventDefault();
  dropZone.classList.remove('drag');
  if (e.dataTransfer.files.length) uploadFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', ()=>{if(fileInput.files.length) uploadFiles(fileInput.files)});

async function uploadFiles(files) {
  let ok=0, fail=0;
  uploadStatus.innerHTML = '';
  for (let i=0; i<files.length; i++) {
    const f = files[i];
    const overall = files.length>1 ? (' ('+(i+1)+'/'+files.length+')') : '';

    // Build progress DOM
    const div = document.createElement('div');
    div.className = 'file-progress';
    div.innerHTML = '<div class="fname"><span>'+esc(f.name)+'</span><span class="pct">0%</span></div>'
      +'<div class="progress-bar"><div class="fill"></div></div>';
    uploadStatus.appendChild(div);
    const fill = div.querySelector('.fill');
    const pctEl = div.querySelector('.pct');

    try {
      await uploadOne(f, (pct) => {
        if (pct >= 98) {
          fill.classList.add('indeterminate');
          pctEl.textContent = '分析中…';
        } else {
          fill.style.width = pct+'%';
          pctEl.textContent = Math.round(pct)+'%';
        }
      });
      fill.classList.add('done');
      pctEl.textContent = '完成';
      ok++;
    } catch(e) {
      fill.classList.add('fail');
      pctEl.textContent = '失败';
      fail++;
      console.error(f.name, e);
    }
  }
  setTimeout(()=>{uploadStatus.innerHTML=''}, 2000);
  showToast('导入完成: '+ok+' 成功'+(fail?', '+fail+' 失败':''));
  fileInput.value = '';
  loadKB();
}

function uploadOne(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    form.append('file', file);

    // Upload progress: file bytes transferred
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.min(e.loaded / e.total * 100, 99);
        onProgress(pct);
      }
    };

    // Download progress: server is still processing (ingest)
    xhr.onprogress = () => {
      onProgress(99); // keep near-complete while server works
    };

    xhr.onload = () => {
      try {
        const data = JSON.parse(xhr.responseText);
        if (data.ok || data.ingested?.length>0) resolve(data);
        else reject(new Error(data.error||'ingest failed'));
      } catch(e) { reject(e); }
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.open('POST', '/api/ingest');
    xhr.send(form);
  });
}
</script>
</body>
</html>"""


async def _root_handler(_req):
    return web.Response(text=SEARCH_PAGE, content_type="text/html", charset="utf-8")


async def _health_handler(_req):
    return web.json_response({"status": "ok"})


async def _query_handler(req):
    try:
        body = await req.json()
        question = body.get("question", "")
        top_k = body.get("top_k")
        if not question:
            return web.json_response({"error": "question is required"}, status=400)
        answer = await asyncio.to_thread(ask, question, top_k)
        return web.json_response({"answer": answer})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def _kb_list_handler(_req):
    """List all unique documents in the knowledge base."""
    try:
        from rag.store import get_store
        collection, _ = get_store()
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return web.json_response({"files": []})

        # Deduplicate by source_hash
        seen: dict[str, dict] = {}
        for meta in metadatas:
            h = meta.get("source_hash", "")
            if h and h not in seen:
                src_path = meta.get("source", "")
                # Get file size if exists
                size = None
                try:
                    from pathlib import Path as _P
                    p = _P(src_path)
                    if p.exists():
                        size = p.stat().st_size
                except Exception:
                    pass
                seen[h] = {
                    "hash": h,
                    "name": meta.get("source_name", src_path),
                    "path": src_path,
                    "chunks": 0,
                    "size": size,
                }
            if h in seen:
                seen[h]["chunks"] += 1

        files = sorted(seen.values(), key=lambda x: x["name"].lower())
        return web.json_response({"files": files})
    except Exception as e:
        return web.json_response({"error": str(e), "files": []}, status=500)


async def _kb_delete_handler(req):
    """Delete a document and all its chunks from the knowledge base."""
    source_hash = req.match_info.get("hash", "")
    if not source_hash:
        return web.json_response({"ok": False, "error": "hash required"}, status=400)
    try:
        from rag.store import get_store
        from pathlib import Path as _Path
        collection, _ = get_store()
        results = collection.get(where={"source_hash": source_hash})
        if results["ids"]:
            collection.delete(ids=results["ids"])
            # Also delete the file on disk if in knowledge/
            for meta in (results.get("metadatas") or []):
                src = meta.get("source", "")
                if src:
                    p = _Path(src)
                    if p.exists() and "knowledge" in str(p):
                        p.unlink(missing_ok=True)
        return web.json_response({"ok": True})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def _ingest_handler(req):
    """Upload and ingest one or more files into the knowledge base."""
    import shutil
    from pathlib import Path as _Path

    knowledge_dir = _Path(__file__).resolve().parent / "knowledge"
    knowledge_dir.mkdir(exist_ok=True)

    try:
        reader = await req.multipart()
        ingested = []
        errors = []

        while True:
            part = await reader.next()
            if part is None:
                break
            filename = part.filename
            if not filename:
                continue

            # Only allow safe extensions
            safe_exts = {
                ".pdf", ".md", ".txt", ".py", ".js", ".ts", ".go", ".rs",
                ".java", ".cpp", ".c", ".h", ".json", ".yaml", ".yml",
                ".toml", ".xml", ".html", ".css", ".sql", ".sh",
            }
            suffix = _Path(filename).suffix.lower()
            if suffix not in safe_exts:
                errors.append(f"{filename}: unsupported format")
                continue

            dest = knowledge_dir / filename
            with open(dest, "wb") as f:
                while True:
                    chunk = await part.read_chunk(65536)
                    if not chunk:
                        break
                    f.write(chunk)

            try:
                from rag.ingest import ingest_file

                # Run blocking ingest in a thread so it doesn't freeze the
                # event loop while chunking + embedding.
                n = await asyncio.to_thread(ingest_file, dest)
                ingested.append({"name": filename, "chunks": n})
            except Exception as e:
                errors.append(f"{filename}: {e}")

        # Reset BM25 index so next query rebuilds with new documents
        if ingested:
            reset_bm25_index()

        ok = len(ingested) > 0
        return web.json_response({
            "ok": ok or len(errors) == 0,
            "ingested": ingested,
            "errors": errors,
        })
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


def cmd_wechat_serve(args):
    """Start WeChat bot + RAG."""
    asyncio.run(_wechat_serve())


async def _wechat_serve():
    from channels.wechat import run_wechat_bot
    from rag.query import ask as rag_ask

    async def handle_message(from_user: str, content: str, msg_id: str) -> Optional[str]:
        print(f"[wx] {from_user}: {content}")
        try:
            answer = await asyncio.to_thread(rag_ask, content)
            return answer
        except Exception as e:
            return f"出错了: {e}"

    await run_wechat_bot(handle_message)


# ============================================================
# Args
# ============================================================


def build_parser():
    parser = argparse.ArgumentParser(description="nano-rag: local RAG + WeChat bot")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p = sub.add_parser("ingest", help="Index files into the knowledge base")
    p.add_argument("path", help="File or directory to ingest")
    p.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirs")
    p.set_defaults(func=cmd_ingest)

    # query
    p = sub.add_parser("query", help="Ask a question against the knowledge base")
    p.add_argument("question", help="Question to ask")
    p.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    p.add_argument("--retrieve-only", action="store_true", help="Only retrieve, skip LLM")
    p.set_defaults(func=cmd_query)

    # reset
    p = sub.add_parser("reset", help="Delete the knowledge base")
    p.set_defaults(func=cmd_reset)

    # serve (HTTP API)
    p = sub.add_parser("serve", help="Start HTTP API server")
    p.add_argument("--host", default="127.0.0.1", help="Bind host")
    p.add_argument("--port", type=int, default=8899, help="Bind port")
    p.set_defaults(func=cmd_serve)

    # wechat-login
    p = sub.add_parser("wechat-login", help="Login to WeChat (QR code)")
    p.set_defaults(func=cmd_wechat_login)

    # wechat-serve
    p = sub.add_parser("wechat-serve", help="Start WeChat RAG bot")
    p.set_defaults(func=cmd_wechat_serve)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
