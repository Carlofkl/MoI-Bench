#!/usr/bin/env python3
"""Read-only browser for the finalized MoI-Bench package.

The page organizes samples on two independent axes: operation on the
left and target at the top.  It reads ``data/samples.json``,
``data/targets.json`` and ``data/images/sources.csv``; it has no write endpoints.

Usage:
    python scripts/browse_moi.py
    python scripts/browse_moi.py --language en
    python scripts/browse_moi.py --language zh --port 8802 --no-open
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import socket
import threading
import urllib.parse
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

OPERATION_LABELS = {
    "en": {
        "all": "All Operations",
        "gen": "Generation",
        "local/add": "Local · Add",
        "local/remove": "Local · Remove",
        "local/replace": "Local · Replace",
        "local/adjust": "Local · Adjust",
        "global/background": "Global · Background",
        "global/stylize": "Global · Stylization",
        "global/restore": "Global · Restoration & Enhancement",
        "global/camera": "Global · Camera Transformation",
        "understand/comprehend": "Understanding · Semantic Edit",
        "understand/inpaint": "Understanding · Inpainting",
        "understand/outpaint": "Understanding · Outpainting",
        "ref/r1": "Reference · R1",
        "ref/b1r1": "Reference · B1+R1",
        "ref/b1r2": "Reference · B1+R2",
        "ref/b1r3": "Reference · B1+R3",
    },
    "zh": {
        "all": "全部操作",
        "gen": "生成",
        "local/add": "局部 · 增加",
        "local/remove": "局部 · 去除",
        "local/replace": "局部 · 替换",
        "local/adjust": "局部 · 调整",
        "global/background": "全局 · 背景",
        "global/stylize": "全局 · 风格化",
        "global/restore": "全局 · 修复增强",
        "global/camera": "全局 · 相机变化",
        "understand/comprehend": "理解 · 语义编辑",
        "understand/inpaint": "理解 · 填充",
        "understand/outpaint": "理解 · 扩图",
        "ref/r1": "参考 · R1",
        "ref/b1r1": "参考 · B1+R1",
        "ref/b1r2": "参考 · B1+R2",
        "ref/b1r3": "参考 · B1+R3",
    },
}

OBJECT_LABELS = {
    "en": {
        "animal": "Animals",
        "object": "Object",
        "object/concrete": "Concrete Objects",
        "object/abstract": "Abstract Objects",
        "portrait": "Portrait",
        "portrait/body": "Body & Limbs",
        "portrait/face": "Face & Hair",
        "portrait/multi": "Multiple Subjects",
        "portrait/attire": "Clothing & Accessories",
        "text": "Text & Documents",
    },
    "zh": {
        "animal": "动物 / 宠物",
        "object": "物体",
        "object/concrete": "具体物体",
        "object/abstract": "抽象物体",
        "portrait": "人像",
        "portrait/body": "躯体 / 四肢",
        "portrait/face": "五官 / 发型",
        "portrait/multi": "人像多主体",
        "portrait/attire": "衣物 / 饰品",
        "text": "文字 / 文档",
    },
}

DOMAIN_LABELS = {
    "en": {
        "object": "Object",
        "animal": "Animals",
        "portrait": "Portrait",
        "text": "Text & Documents",
    },
    "zh": {
        "object": "物体",
        "animal": "动物 / 宠物",
        "portrait": "人像",
        "text": "文字 / 文档",
    },
}

DOMAIN_ORDER = ["object", "animal", "portrait", "text"]
OBJECT_MINOR_ORDER = {
    "object": ["object/concrete", "object/abstract"],
    "portrait": [
        "portrait/face",
        "portrait/body",
        "portrait/multi",
        "portrait/attire",
    ],
}

UI_TEXT = {
    "en": {
        "html_lang": "en",
        "title": "MoI-Bench Browser",
        "loading": "Loading…",
        "search": "Search task, prompt, or sample ID",
        "object_category": "Target",
        "subcategory": "Subtype",
        "all_objects": "All",
        "general_scope": "General ({scope})",
        "general_mixed": "General (Mixed Objects)",
        "scope_separator": ", ",
        "operation_hint": "Select an operation to show its target categories.",
        "subcategory_hint": "Select Object or Portrait to view its subtypes.",
        "all_in_group": "All {label}",
        "load_more": "Load More",
        "summary": "{total} samples · Operation × Target",
        "status": "Showing {visible} of {filtered} matching samples ({total} total)",
        "empty": "No samples match this combination.",
        "load_error": "Failed to load: ",
        "input_image": "Input Image",
        "missing_image": "Image unavailable; check the downloaded file:",
        "open_image": "Open full-size image",
        "reference_image": "Reference Image {number}",
        "image": "Image {number}",
        "source_page": "Source Page",
        "show_chinese_prompt": False,
    },
    "zh": {
        "html_lang": "zh-CN",
        "title": "MoI-Bench 浏览器",
        "loading": "载入中…",
        "search": "搜索 task、Prompt 或样本 ID",
        "object_category": "对象类型",
        "subcategory": "细分类",
        "all_objects": "全部",
        "general_scope": "通用（{scope}）",
        "general_mixed": "通用（混合对象）",
        "scope_separator": "、",
        "operation_hint": "选择一个编辑行为后显示该行为的实际对象分区。",
        "subcategory_hint": "选择物体或人像后可查看细分类。",
        "all_in_group": "全部{label}",
        "load_more": "继续加载",
        "summary": "{total} 条样本 · Operation × Target",
        "status": "当前显示 {visible} / {filtered} 条筛选结果（全库 {total} 条）",
        "empty": "这个组合下没有样本。",
        "load_error": "载入失败：",
        "input_image": "输入图",
        "missing_image": "图片未找到，请检查下载文件：",
        "open_image": "查看原图",
        "reference_image": "参考图 {number}",
        "image": "图{number}",
        "source_page": "来源页",
        "show_chinese_prompt": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8802)
    parser.add_argument("--language", "--lang", choices=["en", "zh"], default="en",
                        help="Interface language. English is the default; use zh for Chinese.")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def object_domains(object_name: str, operation_objects: set[str]) -> list[str]:
    """Return the conceptual object domains covered by one benchmark bucket."""
    if object_name == "all":
        represented: set[str] = set()
        for other in operation_objects - {"all", "mixed"}:
            represented.update(object_domains(other, operation_objects))
        remaining = [domain for domain in DOMAIN_ORDER if domain not in represented]
        return remaining or list(DOMAIN_ORDER)
    if object_name == "mixed":
        return list(DOMAIN_ORDER)
    if object_name == "object":
        # General restoration/understanding samples cover ordinary objects and
        # animals/pets; portraits and text have their own benchmark buckets.
        return ["object", "animal"]
    if object_name.startswith("object/"):
        return ["object"]
    if object_name.startswith("portrait/") or object_name == "portrait":
        return ["portrait"]
    if object_name in {"animal", "text"}:
        return [object_name]
    return []


def build_object_groups(samples: list[dict[str, Any]], language: str) -> dict[str, list[dict[str, Any]]]:
    """Build operation-specific object partitions for the browser controls."""
    labels = OBJECT_LABELS[language]
    domains = DOMAIN_LABELS[language]
    ui = UI_TEXT[language]

    def all_group(count: int) -> dict[str, Any]:
        return {
            "id": "all_objects",
            "label": ui["all_objects"],
            "count": count,
            "match_objects": [],
            "match_domains": [],
            "children": [],
        }

    groups_by_operation: dict[str, list[dict[str, Any]]] = {}
    groups_by_operation["all"] = [all_group(len(samples))]
    for domain in DOMAIN_ORDER:
        object_names = sorted({
            sample["target"] for sample in samples
            if (
                (domain == "object" and (
                    sample["target"] == "object" or sample["target"].startswith("object/")
                ))
                or (domain == "portrait" and (
                    sample["target"] == "portrait" or sample["target"].startswith("portrait/")
                ))
                or sample["target"] == domain
            )
        })
        count = sum(sample["target"] in object_names for sample in samples)
        if object_names:
            groups_by_operation["all"].append({
                "id": f"domain:{domain}",
                "label": domains[domain],
                "count": count,
                "match_objects": object_names,
                "match_domains": [],
                "children": [],
            })

    operation_ids = list(dict.fromkeys(sample["operation"] for sample in samples))
    for operation in operation_ids:
        subset = [sample for sample in samples if sample["operation"] == operation]
        counts = Counter(sample["target"] for sample in subset)
        groups = [all_group(len(subset))]

        for broad in ("all", "object", "mixed"):
            if broad not in counts:
                continue
            covered = next(
                sample["object_domains"] for sample in subset if sample["target"] == broad
            )
            scope = ui["scope_separator"].join(domains[domain] for domain in covered)
            label = ui["general_mixed"] if broad == "mixed" else ui["general_scope"].format(scope=scope)
            groups.append({
                "id": f"exact:{broad}",
                "label": label,
                "count": counts[broad],
                "match_objects": [broad],
                "match_domains": [],
                "children": [],
            })

        object_children = [name for name in OBJECT_MINOR_ORDER["object"] if name in counts]
        if object_children:
            groups.append({
                "id": "major:object",
                "label": domains["object"],
                "count": sum(counts[name] for name in object_children),
                "match_objects": object_children,
                "match_domains": [],
                "children": [
                    {"id": name, "label": labels[name], "count": counts[name]}
                    for name in object_children
                ],
            })

        if "animal" in counts:
            groups.append({
                "id": "exact:animal",
                "label": domains["animal"],
                "count": counts["animal"],
                "match_objects": ["animal"],
                "match_domains": [],
                "children": [],
            })

        portrait_children = [name for name in OBJECT_MINOR_ORDER["portrait"] if name in counts]
        if portrait_children:
            groups.append({
                "id": "major:portrait",
                "label": domains["portrait"],
                "count": sum(counts[name] for name in portrait_children),
                "match_objects": portrait_children,
                "match_domains": [],
                "children": [
                    {"id": name, "label": labels[name], "count": counts[name]}
                    for name in portrait_children
                ],
            })
        elif "portrait" in counts:
            groups.append({
                "id": "exact:portrait",
                "label": domains["portrait"],
                "count": counts["portrait"],
                "match_objects": ["portrait"],
                "match_domains": [],
                "children": [],
            })

        if "text" in counts:
            groups.append({
                "id": "exact:text",
                "label": domains["text"],
                "count": counts["text"],
                "match_objects": ["text"],
                "match_domains": [],
                "children": [],
            })
        groups_by_operation[operation] = groups

    return groups_by_operation


def license_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["local_path"]: row for row in csv.DictReader(handle)}


def build_payload(data_root: Path, language: str = "en") -> dict[str, Any]:
    operation_labels = OPERATION_LABELS[language]
    document = json.loads((data_root / "samples.json").read_text(encoding="utf-8"))
    licenses = license_rows(data_root / "images" / "sources.csv")
    targets = json.loads((data_root / "targets.json").read_text(encoding="utf-8"))["items"]
    if set(targets) != set(document["items"]):
        raise ValueError("Sample IDs differ between samples.json and targets.json")
    samples = []
    for sample_id, record in document["items"].items():
        assignment = targets[sample_id]
        operation, object_name = assignment["operation"], assignment["target"]
        if language == "en":
            operation_labels[operation] = assignment["operation_label"]
        source = sample_id if sample_id.startswith("images/") else ""
        references = [str(value) for value in record.get("reference_images") or []]
        if not references and record.get("reference_image"):
            references = [str(record["reference_image"])]
        image_paths = ([source] if source else []) + references
        samples.append({
            "id": sample_id,
            "task": record["task"],
            "operation": operation,
            "target": object_name,
            "instruction": record.get("instruction", ""),
            "instruction_zh": record.get("instruction_zh", "") if language == "zh" else "",
            "source": source,
            "references": references,
            "licenses": [licenses.get(value, {}) for value in image_paths],
        })
    operation_objects: dict[str, set[str]] = {}
    for sample in samples:
        operation_objects.setdefault(sample["operation"], set()).add(sample["target"])
    for sample in samples:
        sample["object_domains"] = object_domains(
            sample["target"], operation_objects[sample["operation"]]
        )
    operation_counts = Counter(sample["operation"] for sample in samples)
    operation_order = [key for key in operation_labels if key == "all" or key in operation_counts]

    return {
        "samples": samples,
        "operations": [
            {"id": key, "label": operation_labels.get(key, key),
             "count": len(samples) if key == "all" else operation_counts[key]}
            for key in operation_order
        ],
        "object_groups_by_operation": build_object_groups(samples, language),
        "total": len(samples),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="__HTML_LANG__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
:root{--ink:#172033;--muted:#667085;--line:#d8dee9;--paper:#fff;--wash:#f3f6fa;--blue:#155eef;--nav:268px;--header:160px}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--wash);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--line);padding:14px 22px 12px}
.title-row{display:flex;align-items:center;gap:18px;margin-bottom:11px}.title-row h1{font-size:20px;margin:0;white-space:nowrap}.summary{color:var(--muted);font-size:14px}.search{margin-left:auto;width:min(430px,38vw);padding:9px 12px;border:1px solid #b8c2d3;border-radius:8px;font-size:14px}
.object-axis{display:grid;gap:7px}.axis-row{display:grid;grid-template-columns:112px minmax(0,1fr);align-items:center;gap:10px;min-height:34px}.axis-row.hidden{display:none}.axis-label{font-size:12px;font-weight:700;color:#475467;text-transform:uppercase;letter-spacing:.035em}.object-buttons{display:flex;gap:8px;overflow-x:auto;padding-bottom:1px;min-height:34px;align-items:center}.filter{border:1px solid #c6d0e0;background:#fff;color:#344054;border-radius:8px;padding:8px 11px;cursor:pointer;white-space:nowrap;font-size:13px}.filter:hover{border-color:#84adff}.filter.active{color:#fff;background:var(--blue);border-color:var(--blue)}.filter small{opacity:.72;margin-left:4px}.minor-filter{background:#f8faff}.object-hint{color:var(--muted);font-size:12px;font-style:italic}
.layout{display:grid;grid-template-columns:var(--nav) 1fr;min-height:calc(100vh - var(--header))}aside{position:sticky;top:var(--header);align-self:start;height:calc(100vh - var(--header));overflow:auto;background:#fff;border-right:1px solid var(--line);padding:14px}.operation{width:100%;text-align:left;margin-bottom:7px;padding:10px 11px}.operation span{float:right;opacity:.72}
main{padding:18px 20px}.status{font-size:14px;color:var(--muted);margin:0 0 14px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}.card{background:var(--paper);border:1px solid var(--line);border-radius:11px;overflow:hidden;box-shadow:0 1px 2px rgba(16,24,40,.04)}.images{display:flex;flex-direction:column;gap:12px;padding:10px;background:#e8edf4}.image-wrap{min-width:0;background:#fff;border-radius:6px;overflow:hidden}.image-link{display:block}.image-wrap img{width:100%;height:auto;display:block}.tag{display:block;padding:7px 9px;color:#344054;background:#f8fafc;border-bottom:1px solid #e4e7ec;font-size:12px}.missing-image::after{content:attr(data-missing-label);display:block;padding:24px 10px;color:#637581;font-size:13px;overflow-wrap:anywhere}.prompt-only{height:210px;display:flex;align-items:center;justify-content:center;padding:28px;background:linear-gradient(145deg,#eaf1ff,#f8fbff);font-weight:650;line-height:1.55;text-align:center}
.body{padding:13px 14px}.meta{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:12px;margin-bottom:9px}.task{color:#155eef;font-weight:650}.en{font-size:15px;line-height:1.48;margin:0 0 7px}.zh{font-size:13px;line-height:1.48;color:#475467;margin:0}.licenses{border-top:1px solid #edf0f5;margin-top:11px;padding-top:9px;font-size:11px;color:var(--muted)}.license{margin-top:4px;overflow-wrap:anywhere}.license a{color:#155eef;text-decoration:none}.empty{padding:50px;text-align:center;color:var(--muted)}.more{display:none;margin:20px auto 6px;padding:10px 24px;border:1px solid #84adff;background:#fff;color:#155eef;border-radius:8px;cursor:pointer}.more.show{display:block}
@media(max-width:800px){:root{--nav:178px}.title-row{flex-wrap:wrap}.search{order:3;margin-left:0;width:100%}.axis-row{grid-template-columns:1fr;gap:4px}.grid{grid-template-columns:1fr}}

</style>
</head>
<body>
<header>
  <div class="title-row">
    <h1>MoI-Bench</h1>
    <div class="summary" id="summary">__LOADING__</div>
    <input id="search" class="search" placeholder="__SEARCH_PLACEHOLDER__">
  </div>
  <div class="object-axis">
    <div class="axis-row"><div class="axis-label">__OBJECT_CATEGORY__</div><div id="object-majors" class="object-buttons"></div></div>
    <div class="axis-row" id="subcategory-row"><div class="axis-label">__SUBCATEGORY__</div><div id="object-minors" class="object-buttons"></div></div>
  </div>
</header>
<div class="layout"><aside id="operations"></aside><main><p class="status" id="status"></p><div class="grid" id="grid"></div><button id="more" class="more">__LOAD_MORE__</button></main></div>
<script>
const UI=__UI_JSON__;
const state={operation:'all',objectGroup:'all_objects',objectMinor:'all_subobjects',query:'',limit:60};
let DATA=null;
const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const imageUrl=p=>'/image?path='+encodeURIComponent(p);
const fill=(template,values)=>Object.entries(values).reduce((text,[key,value])=>text.replaceAll(`{${key}}`,String(value)),template);
const operationMatch=(sample,id=state.operation)=>id==='all'||sample.operation===id;
const activeObjectGroups=()=>DATA.object_groups_by_operation[state.operation]||DATA.object_groups_by_operation.all;
const selectedObjectGroup=()=>activeObjectGroups().find(group=>group.id===state.objectGroup);
function objectMatch(sample){
  if(state.objectGroup==='all_objects')return state.objectMinor==='all_subobjects'||sample.target===state.objectMinor;
  const group=selectedObjectGroup();
  if(!group)return false;
  const groupMatch=group.match_objects.length
    ?group.match_objects.includes(sample.target)
    :group.match_domains.some(domain=>sample.object_domains.includes(domain));
  return groupMatch&&(state.objectMinor==='all_subobjects'||sample.target===state.objectMinor);
}
function matches(sample){
  const q=state.query.toLowerCase();
  const searchable=UI.show_chinese_prompt?[sample.id,sample.task,sample.instruction,sample.instruction_zh]:[sample.id,sample.task,sample.instruction];
  return operationMatch(sample)&&objectMatch(sample)&&(!q||searchable.join(' ').toLowerCase().includes(q));
}
function syncHeaderHeight(){document.documentElement.style.setProperty('--header',`${document.querySelector('header').offsetHeight}px`)}
function renderObjectControls(){
  const groups=activeObjectGroups();
  const majorRoot=document.getElementById('object-majors');
  majorRoot.innerHTML=groups.map(group=>`<button class="filter" data-object-group="${esc(group.id)}">${esc(group.label)}<small>${group.count}</small></button>`).join('');
  majorRoot.querySelectorAll('[data-object-group]').forEach(button=>button.onclick=()=>{
    state.objectGroup=button.dataset.objectGroup;
    state.objectMinor='all_subobjects';
    state.limit=60;
    render();
  });

  const row=document.getElementById('subcategory-row');
  const root=document.getElementById('object-minors');
  const group=selectedObjectGroup();
  if(state.operation==='all'){
    row.classList.remove('hidden');
    root.innerHTML=`<span class="object-hint">${esc(UI.operation_hint)}</span>`;
    return;
  }
  if((!group||!group.children.length)&&state.objectGroup==='all_objects'&&groups.some(item=>item.children.length)){
    row.classList.remove('hidden');
    root.innerHTML=`<span class="object-hint">${esc(UI.subcategory_hint)}</span>`;
    return;
  }
  if(!group||!group.children.length){
    row.classList.add('hidden');
    root.innerHTML='';
    return;
  }
  row.classList.remove('hidden');
  const allLabel=fill(UI.all_in_group,{label:group.label});
  const options=[{id:'all_subobjects',label:allLabel},...group.children];
  root.innerHTML=options.map(item=>`<button class="filter minor-filter" data-object-minor="${esc(item.id)}">${esc(item.label)}<small>${item.id==='all_subobjects'?group.count:item.count}</small></button>`).join('');
  root.querySelectorAll('[data-object-minor]').forEach(button=>button.onclick=()=>{
    state.objectMinor=button.dataset.objectMinor;
    state.limit=60;
    render();
  });
}
function controls(){
  document.querySelectorAll('[data-operation]').forEach(button=>button.classList.toggle('active',button.dataset.operation===state.operation));
  document.querySelectorAll('[data-object-group]').forEach(button=>button.classList.toggle('active',button.dataset.objectGroup===state.objectGroup));
  document.querySelectorAll('[data-object-minor]').forEach(button=>button.classList.toggle('active',button.dataset.objectMinor===state.objectMinor));
}
function licenseHTML(licenses){
  return licenses.map((license,index)=>{
    if(!license||!license.provider)return'';
    const author=license.author?` · ${esc(license.author)}`:'';
    const link=license.source_url?`<a href="${esc(license.source_url)}" target="_blank" rel="noreferrer">${esc(UI.source_page)}</a>`:'';
    return `<div class="license">${esc(fill(UI.image,{number:index+1}))} · ${esc(license.provider)}${author} · ${esc(license.license)} ${link}</div>`;
  }).join('');
}
function card(sample){
  const imageItems=[];
  if(sample.source){
    const sourceLabel=sample.operation==='ref/r1'?fill(UI.reference_image,{number:1}):UI.input_image;
    imageItems.push({path:sample.source,label:sourceLabel});
  }
  sample.references.forEach((path,index)=>imageItems.push({path,label:fill(UI.reference_image,{number:index+1})}));
  const visual=imageItems.length
    ?'<div class="images">'+imageItems.map(item=>`<div class="image-wrap" data-missing-label="${esc(UI.missing_image+' '+item.path)}"><span class="tag">${esc(item.label)}</span><a class="image-link" href="${imageUrl(item.path)}" target="_blank" rel="noopener" title="${esc(UI.open_image)}"><img loading="lazy" onerror="this.hidden=true;this.parentElement.parentElement.classList.add('missing-image')" src="${imageUrl(item.path)}" alt="${esc(item.label)}"></a></div>`).join('')+'</div>'
    :`<div class="prompt-only">${esc(sample.instruction)}</div>`;
  const licenses=licenseHTML(sample.licenses);
  const chinese=UI.show_chinese_prompt&&sample.instruction_zh?`<p class="zh">${esc(sample.instruction_zh)}</p>`:'';
  return `<article class="card">${visual}<div class="body"><div class="meta"><span class="task">${esc(sample.task)}</span><span>${esc(sample.id)}</span></div><p class="en">${esc(sample.instruction)}</p>${chinese}${licenses?`<div class="licenses">${licenses}</div>`:''}</div></article>`;
}
function render(){
  renderObjectControls();
  controls();
  const rows=DATA.samples.filter(matches),visible=rows.slice(0,state.limit);
  document.getElementById('status').textContent=fill(UI.status,{visible:visible.length,filtered:rows.length,total:DATA.total});
  document.getElementById('grid').innerHTML=visible.length?visible.map(card).join(''):`<div class="empty">${esc(UI.empty)}</div>`;
  document.getElementById('more').classList.toggle('show',visible.length<rows.length);
  requestAnimationFrame(syncHeaderHeight);
}
fetch('/api/data?v=2',{cache:'no-store'}).then(response=>response.json()).then(data=>{
  DATA=data;
  document.getElementById('summary').textContent=fill(UI.summary,{total:data.total});
  document.getElementById('operations').innerHTML=data.operations.map(item=>`<button class="filter operation" data-operation="${esc(item.id)}">${esc(item.label)}<span>${item.count}</span></button>`).join('');
  document.querySelectorAll('[data-operation]').forEach(button=>button.onclick=()=>{state.operation=button.dataset.operation;state.objectGroup='all_objects';state.objectMinor='all_subobjects';state.limit=60;render()});
  document.getElementById('search').oninput=event=>{state.query=event.target.value.trim();state.limit=60;render()};
  document.getElementById('more').onclick=()=>{state.limit+=60;render()};
  window.addEventListener('resize',syncHeaderHeight);
  render();
}).catch(error=>{document.getElementById('summary').textContent=UI.load_error+error});
</script></body></html>'''


def make_html(language: str) -> str:
    ui = UI_TEXT[language]
    replacements = {
        "__HTML_LANG__": ui["html_lang"],
        "__PAGE_TITLE__": ui["title"],
        "__LOADING__": ui["loading"],
        "__SEARCH_PLACEHOLDER__": ui["search"],
        "__OBJECT_CATEGORY__": ui["object_category"],
        "__SUBCATEGORY__": ui["subcategory"],
        "__LOAD_MORE__": ui["load_more"],
        "__UI_JSON__": json.dumps(ui, ensure_ascii=False, separators=(",", ":")),
    }
    html = HTML_TEMPLATE
    for marker, value in replacements.items():
        html = html.replace(marker, str(value))
    return html


def make_handler(data_root: Path, payload: dict[str, Any], language: str):
    encoded_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    encoded_html = make_html(language).encode("utf-8")
    resolved_data_root = data_root.resolve()

    allowed_images = {path for sample in payload["samples"] for path in ([sample["source"]] if sample["source"] else []) + sample["references"]}

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            no_store = content_type.startswith("text/") or content_type.startswith("application/json")
            self.send_header("Cache-Control", "no-store" if no_store else "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(encoded_html, "text/html; charset=utf-8")
                return
            if parsed.path == "/api/data":
                self.send_bytes(encoded_payload, "application/json; charset=utf-8")
                return
            if parsed.path == "/image":
                query = urllib.parse.parse_qs(parsed.query)
                relative = (query.get("path") or [""])[0]
                candidate = (resolved_data_root / relative).resolve()
                if (relative not in allowed_images or candidate == resolved_data_root or
                        resolved_data_root not in candidate.parents or not candidate.is_file()):
                    self.send_bytes(b"image not found", "text/plain", 404)
                    return
                self.send_bytes(candidate.read_bytes(), mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
                return
            self.send_bytes(b"not found", "text/plain", 404)

        def log_message(self, format: str, *args: Any) -> None:
            if args and str(args[1]) != "200":
                super().log_message(format, *args)

    return Handler


class IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    payload = build_payload(data_root, args.language)
    handler = make_handler(data_root, payload, args.language)
    server_class = IPv6Server if ":" in args.host else ThreadingHTTPServer
    server = server_class((args.host, args.port), handler)
    url = f"http://{'[' + args.host + ']' if ':' in args.host else args.host}:{args.port}/"
    print(f"MoI-Bench browser ({args.language}): {payload['total']} samples")
    print(url)
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
