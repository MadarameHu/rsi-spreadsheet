#!/usr/bin/env python3
"""Read-only static browser for complete diagnostic artifacts, including live runs."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
import socket
import time
from pathlib import Path


def _json(path):
    if not path.is_file():
        return {'exists': False, 'path': str(path), 'value': None}
    try:
        raw = path.read_text(encoding='utf-8')
        return {'exists': True, 'path': str(path), 'value': json.loads(raw), 'raw': raw}
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {'exists': True, 'path': str(path), 'value': None,
                'raw': locals().get('raw'), 'read_error': str(exc)}


def _text(path):
    if not path.is_file():
        return {'exists': False, 'path': str(path), 'text': None}
    try:
        return {'exists': True, 'path': str(path), 'text': path.read_text(encoding='utf-8')}
    except (OSError, UnicodeError) as exc:
        return {'exists': True, 'path': str(path), 'text': None, 'read_error': str(exc)}


def _dict(document):
    value = document.get('value')
    return value if isinstance(value, dict) else {}


def _within(ep, supplied, default):
    path = Path(supplied) if isinstance(supplied, str) else default
    if not path.is_absolute():
        path = ep / path
    return path if path.resolve().is_relative_to(ep) else default


def _terminal(directory):
    document = _json(directory/'finished.json')
    if not document['exists']:
        return {}
    finished = _dict(document)
    fields = ('agent_terminal_status', 'exception_type', 'exception_message', 'finished_at',
              'trial_result_path', 'trial_result_sha256')
    if isinstance(finished.get('terminal'), dict):
        return {key: finished['terminal'][key] for key in fields if key in finished['terminal']}
    # Older design invocations kept a hashed trial source instead of a terminal.
    # Agent execution duration and a valid candidate do not establish completion.
    try:
        execution = finished.get('agent_execution')
        source = execution.get('source') if isinstance(execution, dict) else None
        if not isinstance(source, dict) or not isinstance(source.get('path'), str):
            raise ValueError('缺少可验证的 Harbor trial 来源')
        path = Path(source['path']).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError('Harbor trial 来源不在此阶段目录内')
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source.get('sha256'):
            raise ValueError('Harbor trial 来源 SHA-256 不匹配')
        trial = json.loads(raw)
        if not isinstance(trial, dict):
            raise ValueError('Harbor trial 必须是对象')
        timestamp = trial.get('finished_at')
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError('Harbor trial 缺少结束时间，不能确认终态')
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if 'exception_info' not in trial:
            raise ValueError('Harbor trial 缺少 exception_info')
        exception = trial['exception_info']
        status, kind, message = 'completed', None, None
        if exception is not None:
            if not isinstance(exception, dict) or not isinstance(exception.get('exception_type'), str) or not exception['exception_type']:
                raise ValueError('Harbor trial exception_info 格式无效')
            kind, message = exception['exception_type'], exception.get('exception_message')
            status = 'timeout' if kind.endswith('TimeoutError') else 'error'
        return {'agent_terminal_status': status, 'exception_type': kind, 'exception_message': message,
                'finished_at': timestamp, 'trial_result_path': str(path), 'trial_result_sha256': digest}
    except (OSError, ValueError, TypeError) as exc:
        return {'agent_terminal_status': 'unknown', 'verification_error': str(exc)}


def _runtime(ep):
    """Check only a local running controller; preserve terminal records verbatim."""
    return _runtime_document(_json(ep/'runtime-status.json'))


def _runtime_document(document):
    value = _dict(document)
    recorded = value.get('status', 'unknown')
    runtime = {'recorded_status': recorded, 'status': recorded,
               'checked_at': datetime.now(timezone.utc).isoformat(),
               'pid': value.get('pid'), 'hostname': value.get('hostname'), 'started_at': value.get('started_at'),
               'updated_at': value.get('updated_at'), 'reason': value.get('reason'),
               'process_check': 'not_checked', 'path': document['path']}
    if recorded != 'running':
        if recorded not in ('interrupted', 'completed', 'failed', 'incomplete'):
            runtime.update(status='unknown', reason='没有可确认的运行状态记录')
        return runtime
    if value.get('hostname') not in (None, socket.gethostname()):
        runtime.update(status='unknown', process_check='remote', reason='运行记录来自其他主机，未检查本机同号进程；仅显示记录状态')
        return runtime
    pid = value.get('pid')
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        runtime.update(status='unknown', reason='运行记录缺少有效的本机进程 ID')
        return runtime
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        runtime.update(status='interrupted', process_check='absent',
                       reason='运行记录仍为 running，但控制进程已不存在')
    except (PermissionError, OSError) as exc:
        runtime.update(status='unknown', process_check='unavailable',
                       reason='无法检查控制进程：'+str(exc))
    else:
        runtime['process_check'] = 'alive'
    return runtime


def _watch_active(runtime, validation_runtime):
    return (runtime['status'] == 'running' or validation_runtime['status'] == 'running' or
            (validation_runtime['recorded_status'] == 'running' and
             validation_runtime['process_check'] == 'remote'))


def _effective_status(recorded, runtime):
    # Preserve stage completion/failure; controller liveness cannot certify them.
    if recorded != 'running':
        return recorded
    if runtime['status'] == 'running':
        return 'running'
    if runtime['status'] in ('interrupted', 'failed', 'completed'):
        return 'interrupted'
    return 'unknown'


def _reused_worker(ep, record, sample):
    """Read only verified local archives; never follow an external index pointer."""
    from stage_records import read_stage_commit
    directory = ep/'diagnostics/observations'/f'sample-{sample:02d}'
    empty = _json(directory/'missing.json')
    worker = {'sample_id': sample, 'recorded_status': record.get('status'),
              'status': 'failed', 'artifact_status': 'rejected', 'terminal': {},
              'raw': empty, 'derived': empty, 'input_manifest': empty, 'evidence_index': empty,
              'directory': str(directory), 'commit_id': record.get('commit_id'),
              'reused_from': record.get('reused_from'), 'reused_commit_id': record.get('reused_commit_id'),
              'input_texts': {key: _text(directory/'missing.txt') for key in
                              ('attempt', 'agent_prompt', 'original_system_prompt')}}
    try:
        commit_id = record.get('reused_commit_id')
        if not isinstance(commit_id, str) or commit_id != record.get('commit_id'):
            raise ValueError('复用 commit ID 不匹配')
        records = ep/'diagnostics/records'
        if not records.resolve().is_relative_to(ep):
            raise ValueError('复用归档目录不得指向本轮目录以外')
        expected_commit = records/'commits'/f'{commit_id}.json'
        if Path(record.get('commit_path', '')).resolve() != expected_commit.resolve():
            raise ValueError('复用 commit 必须来自本轮 records 归档')
        commit = read_stage_commit(records, commit_id)
        if commit['stage'] != 'observations' or commit['payload'].get('sample_id') != sample:
            raise ValueError('复用 commit 的阶段或 sample ID 不匹配')
        source = Path(record['reused_from'])/'observations'/f'sample-{sample:02d}'
        if not source.is_absolute():
            raise ValueError('复用来源必须是绝对路径')
        refs = {ref['original_path']: records/ref['path'] for ref in commit['references']}

        def document(relative, *, text=False):
            original = str(source/relative)
            archived = refs.get(original)
            if archived is None:
                return {'exists': False, 'path': original, 'text' if text else 'value': None}
            result = (_text if text else _json)(archived)
            result['original_path'] = original
            return result

        for key, relative in [('raw_output_path', 'work/output.json'),
                              ('output_path', 'observations.json'), ('input_manifest', 'inputs/manifest.json')]:
            if record.get(key) != str(source/relative) or str(source/relative) not in refs:
                raise ValueError('复用路径没有匹配的已归档证据：'+key)
        worker.update(status='reused', artifact_status='committed', directory=str(source),
                      raw=document('work/output.json'), derived=document('observations.json'),
                      input_manifest=document('inputs/manifest.json'), evidence_index=document('inputs/evidence_index.json'),
                      terminal=commit.get('metadata', {}).get('stage_execution', {}).get('terminal', {}),
                      input_texts={'attempt': document('inputs/attempt.md', text=True),
                                   'agent_prompt': document('task/instruction.md', text=True),
                                   'original_system_prompt': document('inputs/task/system_prompt.md', text=True)})
        task = document('inputs/task/instruction.md', text=True)
        return worker, task if task['exists'] else None
    except (OSError, ValueError, TypeError, KeyError) as exc:
        worker.update(error_type='ReuseVerificationError', error=str(exc))
        return worker, None


def build_data(episode):
    ep = Path(episode).resolve()
    if not ep.is_dir():
        raise ValueError('episode directory does not exist')
    diag = ep/'diagnostics'
    runtime = _runtime(ep)
    recoveries = [_json(path) for path in sorted(diag.glob('recovery-*/recovery.json'))
                  if path.resolve().is_relative_to(ep)]
    index_doc = _json(diag/'observations/index.json')
    index = _dict(index_doc)
    inventory = _dict(_json(diag/'observations/input_inventory.json'))
    order = index.get('sample_order', inventory.get('sample_order', []))
    if not order:
        order = sorted(int(p.name.split('-')[-1]) for p in (diag/'observations').glob('sample-*')
                       if p.is_dir() and p.name.split('-')[-1].isdigit())
    indexed = {w['sample_id']: w for w in index.get('workers', []) if isinstance(w, dict) and 'sample_id' in w}
    workers = []
    instruction = None
    for sample in order:
        directory = diag/'observations'/f'sample-{sample:02d}'
        record = dict(indexed.get(sample, {}))
        if 'reused_from' in record or 'reused_commit_id' in record:
            worker, task = _reused_worker(ep, record, sample)
            workers.append(worker)
            if instruction is None and task is not None:
                instruction = task
            continue
        record.update(_dict(_json(directory/'state.json')))
        raw_path = _within(ep, record.get('raw_output_path'), directory/'work/output.json')
        output_path = _within(ep, record.get('output_path'), directory/'observations.json')
        raw, derived = _json(raw_path), _json(output_path)
        manifest_path = _within(ep, record.get('input_manifest'), directory/'inputs/manifest.json')
        manifest = _json(manifest_path)
        evidence = _json(manifest_path.parent/'evidence_index.json')
        terminal = _terminal(directory)
        if isinstance(record.get('terminal'), dict):
            terminal.update({k: v for k, v in record['terminal'].items() if k in (
                'agent_terminal_status', 'exception_type', 'exception_message', 'finished_at',
                'trial_result_path', 'trial_result_sha256')})
        started = (directory/'started.json').is_file()
        status = record.get('status', 'pending')
        if status == 'pending' and started:
            status = 'running' if not (directory/'finished.json').exists() else 'finished_uncommitted'
        worker = {'sample_id': sample, 'recorded_status': status,
                  'status': _effective_status(status, runtime), 'artifact_status': record.get('artifact_status',
                  'written_uncommitted' if raw['exists'] else 'missing'),
                  'terminal': terminal, 'raw': raw, 'derived': derived, 'input_manifest': manifest,
                  'evidence_index': evidence, 'directory': str(directory),
                  'input_texts': {
                      'attempt': _text(manifest_path.parent/'attempt.md'),
                      'agent_prompt': _text(directory/'task/instruction.md'),
                      'original_system_prompt': _text(manifest_path.parent/'task/system_prompt.md')},
                  'commit_id': record.get('commit_id'), 'error_type': record.get('error_type'),
                  'error': record.get('error')}
        workers.append(worker)
        task = manifest_path.parent/'task/instruction.md'
        if instruction is None and task.is_file():
            instruction = {'path': str(task), 'text': task.read_text(encoding='utf-8')}
    if instruction is None:
        inputs = _dict(_json(diag/'inputs.json'))
        original = inputs.get('original', {})
        if isinstance(original, dict) and original.get('snapshot_path'):
            task = _within(ep, original['snapshot_path'], diag/'inputs')/'instruction.md'
            if task.is_file():
                instruction = {'path': str(task), 'text': task.read_text(encoding='utf-8')}
    stages = {}
    for name, raw_path in [('attribution', diag/'attribution/work/output.json'),
                           ('design', ep/'work/design.json')]:
        directory = diag/name
        raw, derived = _json(raw_path), _json(diag/f'{name}.json')
        error = _json(directory/'error.json')
        status = ('failed' if error['exists'] else 'artifact_available' if derived['exists'] else
                  'running' if (directory/'started.json').exists() else 'not_started')
        stages[name] = {'recorded_status': status, 'status': _effective_status(status, runtime), 'raw': raw, 'derived': derived,
                        'terminal': _terminal(directory), 'error': error}
    metrics = _json(diag/'metrics.json')
    audit = _json(diag/'audit.json')
    validation = _json(diag/'validation.json')
    validation['runtime'] = _runtime_document(validation)
    committed = sum(w['artifact_status'] == 'committed' for w in workers)
    failed = sum(w['status'] == 'failed' for w in workers)
    audit_status = _dict(audit).get('status')
    current = ('脚本执行验证' if _dict(validation).get('status') == 'running' else
               '审查 / 指标' if audit_status not in (None, 'missing') and
               stages['design']['derived']['exists'] else
               '脚本执行验证' if validation['exists'] else
               '出题' if stages['design']['status'] != 'not_started' else
               '主 Agent 汇总' if stages['attribution']['status'] != 'not_started' else '单路轨迹分析')
    return {'generated_at': datetime.now(timezone.utc).isoformat(), 'episode': str(ep),
            'task': instruction, 'sample_order': order, 'workers': workers, 'runtime': runtime, 'recoveries': recoveries,
            'progress': {'total': len(order), 'written': sum(w['raw']['exists'] for w in workers),
                         'committed': committed, 'failed': failed, 'reused': sum(w['status'] == 'reused' for w in workers), 'stage': current,
                         'batch_status': index.get('status', 'unknown')},
            'stages': stages, 'validation': validation, 'audit': audit, 'metrics': metrics}


HTML = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>轨迹归因 · 运行浏览</title>
<style>
:root{font:15px/1.6 system-ui,-apple-system,sans-serif;color:#213047;background:#f5f7fa}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:32px 24px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}.muted{color:#65748b;font-size:13px}.card{background:white;border:1px solid #dce3ec;border-radius:12px;padding:20px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.stat{padding:15px;background:#edf3fb;border-radius:9px}.stat strong{font-size:25px;display:block}button{font:inherit;cursor:pointer;border:1px solid #cbd5e1;background:white;border-radius:8px;padding:8px 13px}button.active{background:#1e58a5;color:white;border-color:#1e58a5}.samples{display:flex;gap:8px;flex-wrap:wrap}.pill{font-size:12px;background:#edf3fb;border-radius:5px;padding:3px 8px;margin-right:6px;display:inline-block}.failure{background:#fff0ef;color:#a73932}.success{background:#e9f5ed;color:#26723d}pre{font:13px/1.55 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:10px 0;padding:12px;background:#f5f7fa;border-radius:6px;max-height:460px;overflow:auto}details{margin:12px 0}summary{cursor:pointer;color:#285689}.finding{white-space:pre-wrap;overflow-wrap:anywhere}.observation{border-top:1px solid #e5e9f0;padding:16px 0}.note{border-left:3px solid #5081bb;padding-left:12px}.row{display:flex;gap:12px;align-items:center;justify-content:space-between}code{overflow-wrap:anywhere}@media(max-width:650px){.grid{grid-template-columns:repeat(2,1fr)}main{padding:18px 12px}}
</style><main><div class="row"><div><h1>轨迹归因 · 运行浏览</h1><div class="muted" id="time"></div></div><button onclick="location.reload()">刷新快照</button></div>
<p class="note">模型只填写判断与证据引用；身份、路径、状态和版本由脚本记录。文件写出不等于提交成功，提交成功也不等于语义正确。</p>
<section class="card" id="runtime"></section><section class="grid" id="stats"></section><div class="muted" id="stage"></div>
<section class="card" id="recoveries" hidden></section>
<section class="card"><h2>原题与输入</h2><div id="task"></div></section>
<section class="card"><h2>16 路分析 · 按采样查看</h2><div id="samples" class="samples"></div><div id="worker"></div></section>
<section class="card"><h2>主 Agent · 缺陷与能力汇总</h2><div id="attribution"></div></section>
<section class="card"><h2>出题 Agent · 目标与设计</h2><div id="design"></div></section>
<section class="card"><h2>脚本执行验证 · 独立于出题 Agent</h2><div id="validation"></div></section>
<section class="card"><h2>独立审查与正式指标</h2><p class="muted">自动审查指标是估计值；缺失指标保持未知，不记为 0。</p><div id="metrics"></div></section>
<details><summary>运行来源</summary><pre id="source"></pre></details></main>
<script type="application/json" id="data">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const label={reused:'复用已验证分析',pending:'等待开始',running:'进行中',failed:'失败',completed:'已完成',not_started:'未开始',artifact_available:'派生产物已写出',finished_uncommitted:'执行结束，尚未提交',committed:'已提交',rejected:'未通过提交',written_uncommitted:'已写出，尚未提交',missing:'尚无输出',timeout:'超时',error:'执行错误',unknown:'未知',interrupted:'已中断'};
function el(tag,text,parent,cls){let e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;if(parent)parent.appendChild(e);return e}
function json(value){return JSON.stringify(value,null,2)}
function detail(parent,title,value){let d=el('details',undefined,parent);el('summary',title,d);el('pre',typeof value==='string'?value:json(value),d);return d}
function doc(parent,title,item){if(!item.exists){el('p',title+'：尚未生成',parent,'muted');return}detail(parent,title,item.raw??item.value);if(item.read_error)el('p','当前文件尚无法解析：'+item.read_error,parent,'failure')}
function pill(parent,text,cls){return el('span',text,parent,'pill '+(cls||''))}
function body(parent,obj){el('div',obj.finding??obj.statement??obj.feature??'',parent,'finding');for(let key of ['kind','conditions','condition','supports','observation_ids','counterevidence','counterevidence_ids','evidence','uncertainty','uncertainties','claim_id','test_intent','check_plan']){if(Object.hasOwn(obj,key)){el('div',key,parent,'muted');el('pre',typeof obj[key]==='string'?obj[key]:json(obj[key]),parent)}}}
function trainingPlan(parent,stage,name){
  if(name==='attribution'){
    el('h3','训练目标建议 · 主 Agent 的优先顺序',parent);
    const derived=stage.derived.value,raw=stage.raw.value;
    const fromDerived=Array.isArray(derived?.training_goals);
    const goals=fromDerived?derived.training_goals:raw?.training_goals;
    if(!Array.isArray(goals)){el('p','此记录未提供训练目标规划；不从缺陷列表推断优先级。',parent,'muted');return}
    el('p',fromDerived?'来源：脚本派生产物；目标内容和顺序来自模型，目标 ID 由脚本生成。':'来源：模型原始输出，尚无派生目标 ID。',parent,'muted');
    el('p','这是模型的训练建议，不代表经过验证的因果排名或正确率。',parent,'note');
    if(!goals.length){el('p','已记录空的训练目标列表。',parent,'muted');return}
    const list=el('ol',undefined,parent);
    for(const goal of goals){
      const item=el('li',undefined,list,'observation');
      el('h3',(goal.id?goal.id+' · ':'')+(goal.ability??'未记录能力'),item);
      for(const [key,title] of [['claim_ids','关联归因'],['rationale','优先考虑的理由'],['success_signal','可观测的成功信号'],['difficulty_controls','难度控制']]){
        el('div',title,item,'muted');
        el('div',Object.hasOwn(goal,key)?(typeof goal[key]==='string'?goal[key]:json(goal[key])):'未记录',item,'finding');
      }
    }
  }else{
    el('h3','本题主目标与选择依据',parent);
    const value=stage.raw.value??stage.derived.value;
    if(!value||!Object.hasOwn(value,'primary_goal_id')){el('p','此记录未提供主目标；不从 targets 的数量或顺序推断。',parent,'muted');return}
    el('p','来源：'+(stage.raw.value?'模型原始输出':'脚本派生产物'),parent,'muted');
    el('div','主目标：'+(value.primary_goal_id??'未记录'),parent,'finding');
    el('div','选择依据：'+(value.selection_reason??'未记录'),parent,'finding');
  }
}
function showValidation(parent,document){
  if(!document.exists){el('p','未开始：尚无脚本执行验证记录。',parent,'muted');return}
  const value=document.value;
  if(!value||value.schema_version!=='host-validation-v1'){el('p','验证记录尚无法解析或版本不受支持，状态与分数未知。',parent,'muted');doc(parent,'完整验证记录',document);return}
  const labels={running:'执行中',completed:'执行检查已完成',incomplete:'执行不完整',failed:'执行失败',pending:'等待执行',timeout:'超时',error:'执行错误',interrupted:'已中断',missing:'缺失'};
  pill(parent,labels[value.status]??value.status??'未知');
  if(document.runtime?.reason)el('p',document.runtime.reason,parent,'note');
  if(value.error)el('pre','验证阶段错误：'+(typeof value.error==='string'?value.error:json(value.error)),parent,'failure');
  el('p','候选版本：'+(value.candidate_sha256??'未记录'),parent,'muted');
  el('p','目标模型尝试：请求 '+(value.requested_attempts??'未知')+' 次，完成 '+(value.completed_attempts??'未知')+' 次，做对 '+(value.solved_attempts??'未知')+' 次。',parent);
  const trials=[...(value.oracle?[value.oracle]:[]),...(Array.isArray(value.attempts)?value.attempts:[])];
  if(!value.oracle)el('p','参考解：尚无执行结果。',parent,'muted');
  for(const trial of trials){
    const row=el('div',undefined,parent,'observation');
    el('h3',(trial.kind==='oracle'?'参考解':'目标模型')+' · '+(trial.id??'未记录编号'),row);
    const score=typeof trial.reward==='number'&&Number.isFinite(trial.reward)?String(trial.reward):'未知（无有效评分）';
    el('div','状态：'+(labels[trial.status]??trial.status??'未知')+' · 得分：'+score,row);
    el('p','开始：'+(trial.started_at??'未记录')+' · 结束：'+(trial.finished_at??'未记录'),row,'muted');
    if(trial.error)el('pre',typeof trial.error==='string'?trial.error:json(trial.error),row,'failure');
    if(trial.detail!==undefined&&trial.detail!==null)detail(row,'执行详情',trial.detail);
    if(trial.result_path)detail(row,'结果文件位置',trial.result_path);
  }
  if(value.status!=='completed')el('p','待完成：参考解与请求的目标模型尝试尚未全部确认完成；中断或缺失评分不计为 0 分。',parent,'note');
  el('p','此阶段只报告实际执行结果；不判定归因、训练目标覆盖、评分器完整性或难度是否达标。质量结论请查看独立审查与正式指标。',parent,'note');
  doc(parent,'完整脚本验证记录',document);
}
document.getElementById('time').textContent='快照时间：'+data.generated_at+' · 以下为生成快照时的状态，并非浏览器实时进程监测';
if(data.recoveries.length){let root=document.getElementById('recoveries');root.hidden=false;el('h2','恢复历史 · '+data.recoveries.length+' 次',root);el('p','当前产物可能包含复用结果和重跑结果，不代表首次 16 路全部成功。原始失败记录保留在恢复归档中。',root,'note');for(let [i,item] of data.recoveries.entries()){let a=el('div',undefined,root,'observation');el('h3','恢复 '+(i+1),a);let r=item.value;if(r&&typeof r==='object'&&!Array.isArray(r)){el('p','记录时间：'+(r.recorded_at??r.started_at??r.updated_at??'未记录'),a,'muted');el('p','重跑 sample_ids：'+(Array.isArray(r.repeated_sample_ids)?json(r.repeated_sample_ids):'未记录'),a);el('p','复用 sample_ids：'+(Array.isArray(r.reused_sample_ids)?json(r.reused_sample_ids):'未记录'),a);el('p','原因：'+(r.reason??r.action??'未记录'),a)}doc(a,'完整恢复记录',item);detail(a,'恢复记录来源',item.path)}}
let runtimeRoot=document.getElementById('runtime');el('h2','运行环境：'+(label[data.runtime.status]??data.runtime.status),runtimeRoot);if(data.runtime.reason)el('p',data.runtime.reason,runtimeRoot);detail(runtimeRoot,'运行记录与本机进程核查',data.runtime);
__REFRESH__
for(let [title,value] of [['采样总数',data.progress.total],['原始输出已写',data.progress.written],['结构校验后提交',data.progress.committed],['失败',data.progress.failed]]){let a=el('div',undefined,document.getElementById('stats'),'stat');el('span',title,a);el('strong',String(value),a)}
document.getElementById('stage').textContent='当前阶段：'+data.progress.stage+' · 原始批次记录：'+(label[data.progress.batch_status]??data.progress.batch_status)+(data.progress.reused?' · '+data.progress.reused+' 路复用已验证的历史分析，本轮没有重跑这些分析':'');
if(data.task){el('pre',data.task.text,document.getElementById('task'));detail(document.getElementById('task'),'原题文件位置',data.task.path)}else el('p','原题输入尚未落盘',document.getElementById('task'),'muted');
function showWorker(index){let w=data.workers[index],root=document.getElementById('worker');root.replaceChildren();document.querySelectorAll('#samples button').forEach((b,i)=>b.classList.toggle('active',i===index));el('h3','Sample '+String(w.sample_id).padStart(2,'0'),root);pill(root,label[w.status]??w.status);pill(root,'产物：'+(label[w.artifact_status]??w.artifact_status),w.artifact_status==='committed'?'success':w.status==='failed'?'failure':'');pill(root,'模型终态：'+(label[w.terminal.agent_terminal_status]??w.terminal.agent_terminal_status??'尚无终态记录'));
if(w.reused_from)detail(root,'历史分析复用来源（终态属于原运行）',{reused_from:w.reused_from,reused_commit_id:w.reused_commit_id,verified:w.status==='reused'});if(w.error)el('pre',w.error_type+': '+w.error,root);el('h3','模型语义输出',root);let raw=w.raw.value,derived=w.derived.value;let observations=raw?.observations??derived?.observations??[];if(!observations.length)el('p','尚无可解析观察；已有文件可在下方完整展开。',root,'muted');for(let [i,o] of observations.entries()){let a=el('div',undefined,root,'observation');el('h3','观察 '+(i+1),a);body(a,o)}for(let key of ['limitations','evidence_limitations'])if(raw&&Object.hasOwn(raw,key))detail(root,key,raw[key]);doc(root,'完整模型原始 JSON（不截断）',w.raw);doc(root,'完整脚本派生 JSON（不截断）',w.derived);
el('h3','脚本信息与完整输入索引',root);detail(root,'身份、路径、提交和终态',{sample_id:w.sample_id,directory:w.directory,recorded_status:w.recorded_status,display_status:w.status,artifact_status:w.artifact_status,commit_id:w.commit_id,terminal:w.terminal});doc(root,'输入 manifest（完整文件清单与哈希）',w.input_manifest);doc(root,'证据索引（事件 / 行号）',w.evidence_index);for(let [key,title] of [['attempt','完整单路轨迹 attempt.md'],['agent_prompt','分析 Agent 实际提示词'],['original_system_prompt','原任务系统提示词']]){let input=w.input_texts[key];if(input.exists){detail(root,title,input.text??input.read_error);detail(root,title+' · 来源',input.path)}else el('p',title+'：尚未落盘',root,'muted')}}
data.workers.forEach((w,i)=>{let b=el('button',String(w.sample_id).padStart(2,'0')+' · '+(label[w.status]??w.status),document.getElementById('samples'));b.onclick=()=>showWorker(i)});if(data.workers.length)showWorker(0);
for(let name of ['attribution','design']){let s=data.stages[name],root=document.getElementById(name);pill(root,label[s.status]??s.status);if(s.terminal.agent_terminal_status){pill(root,'模型终态：'+(label[s.terminal.agent_terminal_status]??s.terminal.agent_terminal_status),['timeout','error'].includes(s.terminal.agent_terminal_status)?'failure':'');if(s.terminal.verification_error)el('p','终态无法核验：'+s.terminal.verification_error,root,'muted')}el('p','原始阶段记录：'+(label[s.recorded_status]??s.recorded_status),root,'muted');trainingPlan(root,s,name);let value=s.raw.value??s.derived.value;if(value){for(let group of ['claims','targets','mappings'])if(Array.isArray(value[group]))for(let o of value[group]){let a=el('div',undefined,root,'observation');body(a,o)}for(let key of ['preserves','simplifies'])if(Object.hasOwn(value,key))detail(root,key,value[key])}doc(root,'模型原始 JSON',s.raw);doc(root,'脚本派生 JSON',s.derived);if(Object.keys(s.terminal).length)detail(root,'实际终态',s.terminal);if(s.error.exists)doc(root,'阶段错误',s.error)}
showValidation(document.getElementById('validation'),data.validation);
doc(document.getElementById('metrics'),'正式指标',data.metrics);doc(document.getElementById('metrics'),'审查结果',data.audit);document.getElementById('source').textContent=data.episode;
</script></html>'''


def render(episode, output_dir, *, refresh_seconds=None):
    data = build_data(episode)
    output = Path(output_dir).resolve()
    ep = Path(episode).resolve()
    if output.is_relative_to(ep):
        raise ValueError('output must be outside the evidence episode')
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2)
    # HTML parser must never see a model-supplied closing script tag.
    embedded = encoded.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    data_tmp = output/'.data.json.tmp'
    data_tmp.write_text(encoded+'\n', encoding='utf-8')
    data_tmp.replace(output/'data.json')
    refresh = (f"setTimeout(()=>location.reload(), {int(refresh_seconds * 1000)});"
               if refresh_seconds and _watch_active(data['runtime'], data['validation']['runtime']) else '')
    page_tmp = output/'.index.html.tmp'
    page_tmp.write_text(HTML.replace('__DATA__', embedded).replace('__REFRESH__', refresh), encoding='utf-8')
    page_tmp.replace(output/'index.html')
    return output/'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--watch', action='store_true', help='周期重建快照；兼容独立验证，异机只跟踪已记录状态')
    parser.add_argument('--interval', type=float, default=10, help='快照更新间隔（秒）')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    while True:
        print(render(args.episode, args.output_dir,
                     refresh_seconds=args.interval if args.watch else None), flush=True)
        if not args.watch:
            break
        validation_runtime = _runtime_document(_json(args.episode/'diagnostics/validation.json'))
        if not _watch_active(_runtime(args.episode), validation_runtime):
            # A controller may finish during rendering; leave a terminal snapshot.
            render(args.episode, args.output_dir)
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
