"""Render presentation artifacts from saved metrics. No network required."""
from pathlib import Path
import base64
import html
import hashlib
import json
import os
import textwrap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from pipeline import FEATURES

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/lag14'
F=ROOT/'figures'
F.mkdir(exist_ok=True)
META=json.loads((R/'summary.json').read_text())
P=pd.read_csv(R/'policies.csv').set_index('policy')
CURVE=pd.read_csv(R/'budget_curve.csv')
MONTHLY=pd.read_csv(R/'monthly.csv')
VALIDATION=pd.read_csv(R/'validation_models.csv').set_index('model')
# Frozen prose is published only against the explicitly reviewed metric files.
reference=json.loads((ROOT/'results/report_reference.json').read_text())
for relative,expected_hash in reference['metric_files_sha256'].items():
    actual=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f'Review the report narrative and refresh report_reference.json: {relative} changed.')
for lag,values in reference['summary_values'].items():
    current=json.loads((ROOT/f'results/lag{lag}/summary.json').read_text())
    if any(current[key] != value for key,value in values.items()):
        raise RuntimeError(f'Review the report narrative: lag{lag} summary changed.')
# Narrative paragraphs were reviewed for this frozen experiment. Fail loudly if
# a new run changes results; otherwise static prose could silently become stale.
expected={'model':836,'history_rule':795,'high_risk':546,'longest_wait':667,
          'model_quota50':804,'model_quota100':779}
if META['audit']['raw_rows'] != 123673 or any(P.loc[k,'hits'] != v for k,v in expected.items()):
    raise RuntimeError('Results changed: update and review the fixed report narrative before exporting.')
LABELS={'model':'逻辑回归排序','history_rule':'历史不合格优先','high_risk':'历史高风险等级优先',
        'longest_wait':'最长未例检优先','random_expected':'随机选择（期望）',
        'model_quota50':'50% 名额按邮编比例分配','model_quota100':'100% 名额按邮编比例分配'}
SHORT={'model':'模型排序','history_rule':'历史经验','high_risk':'风险等级',
       'longest_wait':'等待时长','random_expected':'随机期望','model_quota50':'50% 比例配额','model_quota100':'100% 比例配额'}
FEATURE_ZH={'history_log':'历史记录数量','fail_rate':'历史不合格率','recent_fail_rate':'近一年不合格率',
 'conditions_rate':'历史附条件通过率','last_fail':'上次不合格','last_conditions':'上次附条件通过',
 'days_since_last_log':'距上次检查天数','days_since_routine_log':'距上次例行检查天数',
 'recent_count_log':'近一年检查次数','complaint_count_log':'近一年投诉检查次数','complaint_fail_log':'投诉检查不合格次数',
 'risk_high':'历史高风险等级','risk_medium':'历史中风险等级','risk_low':'历史低风险等级',
 'no_history':'无可用历史','month_sin':'月份季节项','month_cos':'月份季节项'}

font_paths=[os.environ.get('PROJECT_CJK_FONT',''),'/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
 '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc','C:/Windows/Fonts/msyh.ttf']
FONT_PATH=next((p for p in font_paths if p and Path(p).exists()),None)
if FONT_PATH:
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family']=font_manager.FontProperties(fname=FONT_PATH).get_name()
    try:
        pdfmetrics.registerFont(TTFont('CJK',FONT_PATH))
    except Exception:
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        PDF_FONT='STSong-Light'
    else: PDF_FONT='CJK'
else:
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    PDF_FONT='STSong-Light'
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.edgecolor':'#ccd4d5','text.color':'#172f38','axes.labelcolor':'#172f38',
 'xtick.color':'#546870','ytick.color':'#172f38','font.size':11,'axes.unicode_minus':False,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

def save(fig,name):
    fig.savefig(F/name,dpi=170,bbox_inches='tight')
    plt.close(fig)

def charts():
    order=['random_expected','high_risk','longest_wait','history_rule','model']
    fig,ax=plt.subplots(figsize=(8.5,3.5))
    values=P.loc[order,'precision'].to_numpy()*100
    ax.barh(range(len(order)),values,color=['#bcc8cc']*4+['#137e7c'],height=.57)
    ax.set_yticks(range(len(order)),[SHORT[k] for k in order])
    ax.set_xlim(0,max(values)*1.17)
    ax.set_xlabel('优先名单中实际 Fail 的占比（%）')
    ax.xaxis.grid(True,alpha=.15); ax.set_axisbelow(True)
    for i,v in enumerate(values): ax.text(v+.5,i,f'{v:.2f}%',va='center',weight='bold')
    save(fig,'01_policy_precision.png')
    fig,ax=plt.subplots(figsize=(8.5,3.5))
    for key,color,marker in [('model','#137e7c','o'),('history_rule','#8d9ba4','s'),('random_expected','#bbc6cb','^')]:
        d=CURVE[CURVE.policy.eq(key)]
        ax.plot(d.budget*100,d.recall*100,label=SHORT[key],color=color,marker=marker,lw=2)
    ax.set_xlabel('每月优先名额占候选池比例（%）'); ax.set_ylabel('捕获全部 Fail 的比例（%）')
    ax.legend(frameon=False); ax.grid(alpha=.15)
    save(fig,'02_budget_recall.png')
    fig,ax=plt.subplots(figsize=(8.5,3.6))
    for key,col,offset in [('model','#137e7c',(-12,14)),('model_quota50','#d57f3f',(10,8)),('model_quota100','#687e8c',(10,8))]:
        row=P.loc[key]
        ax.scatter(row.zip_tv,row.precision*100,s=95,color=col)
        ax.annotate(SHORT[key],(row.zip_tv,row.precision*100),xytext=offset,textcoords='offset points',fontsize=10)
    ax.set_xlim(.025,.355);ax.set_ylim(31.6,36.3)
    ax.set_xlabel('平均邮编构成偏离度 TV（越小越接近候选池）')
    ax.set_ylabel('命中率（%）');ax.grid(alpha=.15)
    save(fig,'03_coverage_tradeoff.png')
    fig,ax=plt.subplots(figsize=(8.5,3.2))
    months=sorted(MONTHLY.month.unique())
    for key,color in [('model','#137e7c'),('history_rule','#8d9ba4')]:
        d=MONTHLY[MONTHLY.policy.eq(key)].sort_values('month')
        ax.plot(np.arange(len(d)),d.hits/d.selected*100,label=SHORT[key],color=color,lw=1.8,marker='o',ms=3)
    ax.set_xticks([0,5,11,17,23],[months[i] for i in [0,5,11,17,23]])
    ax.set_xlabel('留出测试月份');ax.set_ylabel('命中率（%）');ax.legend(frameon=False,ncol=2);ax.grid(alpha=.15)
    save(fig,'04_monthly_stability.png')

def explain_rows():
    d=pd.read_csv(R/'scored_test.csv.gz',dtype={'zip_group':str})
    model=joblib.load(R/'model.joblib')
    contributions=model[0].transform(d[FEATURES])*model[-1].coef_[0]
    logits=model[-1].intercept_[0]+contributions.sum(axis=1)
    assert np.allclose(1/(1+np.exp(-logits)),d.score,atol=1e-12)
    reasons=[]
    for cs in contributions:
        ix=np.argsort(cs)[::-1]
        names=[]
        for i in ix:
            if cs[i] <= 0 or len(names)>=2: break
            name=FEATURE_ZH[FEATURES[i]]
            if name not in names:names.append(name)
        reasons.append('、'.join(names) or '综合历史特征')
    d['score_explanation']=reasons
    d[['venue_id','month','score','score_explanation','y','model_selected']].to_csv(R/'explained_test.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    result={}
    for month,g in d.groupby('month'):
        sub=g.sort_values(['score','venue_id'],ascending=[False,True]).head(8)
        result[month]=sub[['venue_id','zip_group','score','n_history','score_explanation','y']].to_dict('records')
    return result

def img_data(name):
    return 'data:image/png;base64,'+base64.b64encode((F/name).read_bytes()).decode()

def make_html(cases):
    data=CURVE.replace({np.nan:None}).to_dict('records')
    payload=json.dumps({'curves':data,'labels':LABELS,'cases':cases},ensure_ascii=False).replace('</','<\\/')
    template='''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>先查哪家？｜风险排序与覆盖审计</title>
<style>
:root{--ink:#17333c;--teal:#137e7c;--paper:#f4f3ee;--muted:#536970;--line:#d7dedb;--orange:#bd642e}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.75}a{color:var(--teal)}
header{background:var(--ink);color:#fff;padding:58px max(6vw,22px) 45px}.wrap{max-width:1110px;margin:auto}.eyebrow{font-size:12px;letter-spacing:2px;color:#aad9d1;font-weight:600}h1{font-size:clamp(32px,5vw,58px);letter-spacing:-1px;line-height:1.2;margin:16px 0}header p{max-width:770px;color:#d1e0df;font-size:17px}.tag{display:inline-block;border:1px solid #729393;border-radius:20px;padding:4px 13px;font-size:12px;margin:5px 7px 0 0}
main{padding:28px max(6vw,22px) 65px}nav{display:flex;gap:22px;flex-wrap:wrap;margin-bottom:24px;border-bottom:1px solid var(--line);padding-bottom:15px}nav a{text-decoration:none;font-size:14px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.metric{padding:22px 24px;background:#fff}.metric strong{font-size:38px;line-height:1.3;display:block;color:var(--teal)}.metric span{font-size:14px}.metric small{display:block;color:var(--muted);margin-top:8px}.caveat{background:#e7edeb;padding:14px 20px;font-size:14px;border-left:3px solid var(--teal);margin:20px 0}
section{margin-top:52px}h2{font-size:25px;font-weight:650;margin-bottom:12px}h2 em{font-style:normal;color:var(--orange);font-size:14px;display:block;letter-spacing:1px}h3{font-size:17px}p{max-width:950px}.two{display:grid;grid-template-columns:1fr 1fr;gap:30px}.plot{width:100%;display:block;background:#fff}.small{font-size:13px;color:var(--muted)}.panel{background:#fff;padding:24px}.controls{display:flex;align-items:center;gap:20px;flex-wrap:wrap}.controls input{width:270px;accent-color:var(--teal)}select{font:inherit;padding:7px;border:1px solid #a6b8b6;background:#fff;color:var(--ink)}output{font-weight:700;font-size:25px;color:var(--teal)}
table{border-collapse:collapse;width:100%;font-size:14px}th{text-align:left;color:var(--muted);font-weight:500;border-bottom:2px solid var(--line)}td,th{padding:12px 10px}td{border-bottom:1px solid #e6ebe9}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tr.active{background:#e9f3ef}tr.active td:first-child{color:var(--teal);font-weight:600}.table-scroll{overflow:auto}.step{padding:17px 0;border-bottom:1px solid var(--line)}.step b{display:block}.step p{margin:4px 0 0;font-size:14px;color:var(--muted)}.finding{font-size:20px;line-height:1.6;border-left:3px solid var(--orange);padding-left:18px}footer{margin-top:45px;border-top:1px solid var(--line);padding-top:24px;font-size:12px;color:var(--muted)}details{margin-top:18px}summary{font-weight:600;cursor:pointer}button{font:inherit;color:var(--teal);background:none;border:1px solid var(--teal);padding:8px 14px;cursor:pointer}button:hover{background:#e8f2ed}code{font-size:12px;background:#e9eeeb;padding:2px 5px}
@media(max-width:760px){.grid,.two{grid-template-columns:1fr}.metric{padding:17px}.metric strong{font-size:32px}section{margin-top:36px}.panel{padding:16px}td,th{padding:10px 6px}header{padding-top:35px}h1{font-size:38px}.controls{gap:12px}.controls input{width:200px}}
@media print{header{background:white;color:#17333c}header p,.eyebrow{color:#536970}nav,.controls{display:none}main,header{padding:16px}.panel{padding:0}section{break-inside:avoid}.grid{grid-template-columns:repeat(3,1fr)}}
</style></head><body>
<header><div class="wrap"><div class="eyebrow">DECISION SCIENCE / 01 · 个人研究项目</div><h1>先查哪家？</h1><p>把一个预测分数，变成可解释、可比较的资源分配决策。<br>餐饮巡检预算下的风险排序与覆盖审计。</p><span class="tag">真实官方公开数据</span><span class="tag">2019—2025</span><span class="tag">离线回顾性评估</span></div></header>
<main><div class="wrap"><nav><a href="#result">结果</a><a href="#budget">预算实验</a><a href="#coverage">覆盖取舍</a><a href="#method">信息时点</a><a href="#examples">历史候选示例</a><a href="#limits">边界与复现</a></nav>
<div class="grid"><div class="metric"><span>同等约 20% 名额的命中率</span><strong>35.05%</strong><small>随机期望 23.30%；经验规则 33.33%</small></div><div class="metric"><span>相对最强经验规则多捕获</span><strong>41 条</strong><small>836 vs 795 条 Fail；同为 2,385 个名额</small></div><div class="metric"><span>时间留出的测试样本</span><strong>11,869</strong><small>2024—2025 年 / 24 个历史月份</small></div></div>
<div class="caveat"><b>先读这一句：</b>候选池由当月实际受检记录事后构造，名单不是月初已知的真实可检查商户全集。这些数字描述历史样本的排序表现，不代表已上线效果、节省人力或减少疾病。</div>
<section id="result"><h2><em>01 / 结果</em>模型胜过随机，经验规则也很强</h2><div class="two"><div><img class="plot" src="@@POLICY@@" alt="五种策略命中率比较：模型35.05%，历史规则33.33%，随机23.30%"><p class="small">每月 ceil(20% × 候选数量)，测试期合计 2,385 个名额；目标为行政记录 Fail。</p></div><div><p class="finding">有价值的结论，是量化模型比经验多做了多少。</p><p>复杂的梯度提升树在 2023 年 1—11 月验证集上没有胜过逻辑回归，因此最终使用逻辑回归，保留 17 个可追溯的历史特征。</p><p>模型相对经验规则提升 <b>1.72 个百分点</b>。按月份配对重采样的 95% 区间为 <b>+0.29 至 +3.05 个百分点</b>，未覆盖同店跨月相关性。</p><p class="small">2024 年差值 +2.50 个百分点，2025 年 +0.93 个百分点。2025 年的观察差值较小；两年数据不足以判断长期趋势。</p></div></div></section>
<section id="budget"><h2><em>02 / 预算实验</em>如果优先名额增加，哪些结果会改变？</h2><div class="panel"><div class="controls"><label for="budget-range">每月优先名额占比</label><input id="budget-range" type="range" min="0" max="7" value="3" step="1"><output id="budget-value" for="budget-range">20%</output></div><p class="small" id="budget-caption" aria-live="polite"></p><div class="table-scroll"><table><thead><tr><th>策略</th><th class="num">捕获 Fail</th><th class="num">命中率</th><th class="num">召回率</th><th class="num">地域偏离 TV ↓</th></tr></thead><tbody id="budget-rows"></tbody></table></div><p class="small">随机项为月内等概率选择的数学期望，可为小数。预算滑块展示预先计算的情景，不重新选择模型；20% 是验证阶段预定的主指标。</p></div><img class="plot" style="margin-top:18px" src="@@CURVE@@" alt="不同优先名额比例下的Fail捕获比例曲线"></section>
<section id="coverage"><h2><em>03 / 管理取舍</em>多覆盖一些地区，要付出多少命中率？</h2><div class="two"><div><img class="plot" src="@@COVERAGE@@" alt="三种策略的命中率与邮编构成偏离度取舍"></div><div><p>先将 <b>50% 的优先名额</b>按候选池的历史邮编构成分配，在组内按分数选；其余名额全局按分数补齐。</p><p>相较纯模型排序，少捕获 <b>32 条</b> Fail，命中率从 <b>35.05% → 33.71%</b>；平均构成偏离度从 <b>0.291 → 0.237</b>。</p><p>这为管理者提供可量化的选项。是否值得，需要结合覆盖目标，而不能只由算法决定。</p><p class="small">TV = 0.5 × 各邮编入选占比与候选占比差的绝对值之和。越低表示越接近候选构成；不等于人口公平或服务需求公平。</p></div></div></section>
<section id="method"><h2><em>04 / 方法</em>把“当时能知道什么”写进数据流程</h2><div class="two"><div><div class="step"><b>123,673 条官方记录 → 29,520 条目标样本</b><p>以许可证＋地址识别对象；排除同日冲突、非评价结果。仅预测餐饮场所的例行检查，历史信息可来自其他检查类型。</p></div><div class="step"><b>每个月 1 日，冻结历史</b><p>只用严格早于“月初减 14 天”的记录。当前检查结果、违规文本、当前风险等级均不进入模型。</p></div><div class="step"><b>训练 / 验证 / 测试严格按时间分开</b><p>2020—2022 年训练，2023 年 1—11 月选择模型，2024—2025 年留出测试。验证排除标签未完整可用的 12 月；训练和重训也剔除尚不满足延迟的年末标签。2019 年用于预热。</p></div><div class="step"><b>SQL 交叉核验 + 30 天延迟复查</b><p>独立核验历史关联数量与前 20% 排名。固定逻辑回归，将信息延迟改成 30 天后，命中率为 34.42%；较经验规则的优势区间为 −0.24 至 +2.30 个百分点，包含零。</p></div></div><div><img class="plot" src="@@MONTHLY@@" alt="2024至2025各月模型与经验规则的命中率"><h3>一个没有藏起来的弱点</h3><p>无可用历史的测试样本有 <b>76 条</b>，其中 18 条 Fail。纯模型只优先选中 5 条，捕获 1 条 Fail。这部分样本太少，且召回不足，无历史不必然等于新店，不能据此认定适合新店。</p><p class="small">实际试点需要独立新店／必查名单或探索名额，并补齐未受检对象与当时可用信息的记录。</p></div></div></section>
<section id="examples"><h2><em>05 / 可解释输出</em>看看历史月份里的优先候选</h2><div class="panel"><div class="controls"><label for="month-select">历史月份</label><select id="month-select"></select></div><p class="small">展示该月分数最高的 8 条历史记录。评分为模型概率输出乘 100，仅用于排序，不是疾病概率；对象代号是哈希标识。信号是相对训练均值提高线性预测值最大的特征，不是因果解释，也不是实时检查建议。</p><div class="table-scroll"><table><thead><tr><th>对象代号</th><th>历史 ZIP</th><th class="num">评分（0—100）</th><th class="num">历史条数</th><th>主要正向信号</th><th>事后结果</th></tr></thead><tbody id="case-rows"></tbody></table></div></div></section>
<section id="limits"><h2><em>06 / 边界与复现</em>能证明什么，尚不能证明什么</h2><div class="two"><div><h3>已完成</h3><p>官方数据快照和校验清单、历史时点特征、模型与基线比较、预算与地域配额实验、月份配对重采样、30 天延迟分析，以及可运行的核验测试。</p><h3>下一步需要真实业务补充</h3><p>月初可检查对象全集、实际发布时间与历史版本、必查规则、检查员差异、位置与路线成本。补充这些后，才适合讨论前瞻试点。</p></div><div><h3>三个结论边界</h3><p>① Fail 不等于严重违规或疾病。<br>② 50.45% 是相对随机的捕获数提升，不是节省 50% 人力。<br>③ 邮编比例约束不等于解决公平性。</p><details><summary>查看复现入口</summary><p>随包 README 说明依赖与运行顺序。核心入口：<code>python src/pipeline.py</code>；敏感性：<code>python src/pipeline.py --lag 30</code>；核验：<code>python -m unittest discover -s tests -v</code>。</p></details></div></div></section>
<footer>数据来源：<a href="https://data.cityofchicago.org/d/4ijn-s7e5">Chicago Food Inspections</a>；历史导出范围 2019—2025，下载于 2026-10-10。研究背景：<a href="https://github.com/Chicago/food-inspections-evaluation">芝加哥既有开源项目</a>、<a href="https://arxiv.org/abs/1910.04906">Kannan et al. (2019)</a>、<a href="https://arxiv.org/abs/2108.05523">Singh et al. (2022)</a>。<br>个人项目 · AI 协助生成与运行核验 · 非官方产品，无真实业务部署。原始记录中的机构名和地址不在本报告展示。</footer>
</div></main><script>
const DATA=@@DATA@@;
const budgets=[.05,.1,.15,.2,.25,.3,.4,.5];
const order=['random_expected','high_risk','longest_wait','history_rule','model','model_quota50','model_quota100'];
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function renderBudget(){const b=budgets[Number(document.getElementById('budget-range').value)];const rows=DATA.curves.filter(r=>Math.abs(r.budget-b)<.00001);document.getElementById('budget-value').textContent=Math.round(b*100)+'%';document.getElementById('budget-caption').textContent='2024—2025 年固定测试集 · '+rows[0].selected.toLocaleString()+' 个优先名额 · '+rows[0].candidates.toLocaleString()+' 条候选记录';document.getElementById('budget-rows').innerHTML=order.map(k=>{const r=rows.find(x=>x.policy===k);return '<tr class="'+(k==='model'?'active':'')+'"><td>'+esc(DATA.labels[k])+'</td><td class="num">'+r.hits.toLocaleString(undefined,{maximumFractionDigits:1})+'</td><td class="num">'+(r.precision*100).toFixed(2)+'%</td><td class="num">'+(r.recall*100).toFixed(2)+'%</td><td class="num">'+(r.zip_tv===null?'—':r.zip_tv.toFixed(3))+'</td></tr>'}).join('');}
const months=Object.keys(DATA.cases).sort();const select=document.getElementById('month-select');select.innerHTML=months.map(m=>'<option>'+m+'</option>').join('');select.value=months[months.length-1];
function renderCases(){document.getElementById('case-rows').innerHTML=DATA.cases[select.value].map(r=>'<tr><td><code>'+esc(r.venue_id.slice(0,8))+'</code></td><td>'+esc(r.zip_group)+'</td><td class="num">'+(r.score*100).toFixed(1)+'</td><td class="num">'+r.n_history+'</td><td>'+esc(r.score_explanation)+'</td><td>'+(r.y?'Fail':'非 Fail')+'</td></tr>').join('');}
document.getElementById('budget-range').addEventListener('input',renderBudget);select.addEventListener('change',renderCases);renderBudget();renderCases();
</script></body></html>'''
    for token,file in [('POLICY','01_policy_precision.png'),('CURVE','02_budget_recall.png'),('COVERAGE','03_coverage_tradeoff.png'),('MONTHLY','04_monthly_stability.png')]:
        template=template.replace('@@'+token+'@@',img_data(file))
    template=template.replace('@@DATA@@',payload)
    (ROOT/'项目展示.html').write_text(template,encoding='utf-8')

INK=colors.HexColor('#17333c');TEAL=colors.HexColor('#137e7c');MUTED=colors.HexColor('#526970')
styles={
 'kicker':ParagraphStyle('kicker',fontName=PDF_FONT,fontSize=9,leading=14,textColor=TEAL,spaceAfter=10),
 'title':ParagraphStyle('title',fontName=PDF_FONT,fontSize=29,leading=39,textColor=INK,spaceAfter=14,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName=PDF_FONT,fontSize=21,leading=29,textColor=INK,spaceAfter=15,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName=PDF_FONT,fontSize=13,leading=21,textColor=TEAL,spaceBefore=13,spaceAfter=7,wordWrap='CJK'),
 'body':ParagraphStyle('body',fontName=PDF_FONT,fontSize=10,leading=17,textColor=INK,spaceAfter=9,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName=PDF_FONT,fontSize=8.5,leading=14,textColor=MUTED,spaceAfter=8,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName=PDF_FONT,fontSize=9,leading=14,textColor=INK,wordWrap='CJK')}

def para(text,style='body'): return Paragraph(text,styles[style])
def table(rows,widths):
    data=[[para(str(x),'cell') for x in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4eeea')),
     ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#c3d2cc')),
     ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dae2de')),
     ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),8),
     ('BOTTOMPADDING',(0,0),(-1,-1),8),('LEFTPADDING',(0,0),(-1,-1),8)]))
    return t

def picture(file,width=490):
    from PIL import Image as PILImage
    with PILImage.open(F/file) as im: h=width*im.height/im.width
    return Image(str(F/file),width=width,height=h,hAlign='LEFT')

def page(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setStrokeColor(colors.HexColor('#d2ddd6'));canvas.line(48,43,w-48,43)
    canvas.setFillColor(MUTED);canvas.setFont(PDF_FONT,8)
    canvas.drawString(48,29,'个人研究项目 | 历史回测，非实际部署')
    canvas.drawRightString(w-48,29,f'{doc.page} / 5')
    canvas.restoreState()

def make_pdf():
    story=[]
    def add(text,style='body'):story.append(para(text,style))
    add('DECISION SCIENCE / 数据科学 × 管理决策','kicker')
    add('先查哪家？','title')
    add('餐饮巡检预算下的风险排序与覆盖审计','h1')
    add('研究问题：优先检查名额只有约 20% 时，历史数据能否比简单经验捕获更多不合格记录？如果兼顾地区覆盖，代价是多少？')
    story.append(table([['测试样本','同等优先名额','模型捕获 Fail'],['11,869 条 / 24 个月','2,385 个','836 条 / 命中率 35.05%']],[165,165,165]))
    story.append(Spacer(1,12));story.append(picture('01_policy_precision.png'))
    add('关键结论：相对随机期望，模型多捕获约 280.3 条；相对最强经验规则，仅多捕获 41 条。后者才是判断模型是否值得维护的关键比较。','small')
    add('口径边界','h2')
    add('候选池由当月实际接受例行检查的餐饮记录事后构造。这是留出样本上的回顾性排序评估，不能表述为已提前发现风险、已节省人力或已降低疾病。','small')
    story.append(PageBreak())

    add('01 / 数据与信息时点','kicker');add('先解决“当时能知道什么”','h1')
    add('来源为芝加哥政府 Food Inspections 公共数据；下载范围 2019—2025 年，2019 年只提供预热历史。2018 年发生过检查程序变更，项目不跨用该旧阶段。')
    story.append(table([['清洗环节','数量 / 处理'],['官方导出','123,673 条'],['无有效许可证、地址或日期','排除 304 条'],['同店同日结果冲突 / 同结果重复','分别排除 1,544 / 76 条'],['有效实体去重后','121,749 条'],['停业、无法入场等非评价结果','从历史中排除 19,260 条'],['可用历史 / 最终目标','102,489 条 / 29,520 条']],[260,235]))
    add('目标样本与标签','h2')
    add('只评估 Restaurant 的 Canvass（例行检查）；同店同月保留第一次。Fail 为正类，Pass 与 Pass w/ Conditions 为负类。实体键为“许可证＋规范化地址”，共 9,628 个组合。投诉检查和复检不作为目标，但可作为旧历史。')
    add('冻结与拆分','h2')
    add('每月 1 日生成特征，只纳入严格早于“月初减 14 天”的历史事件。使用 17 个历史与月份特征；不使用本次结果、违规文本、本次风险等级、店铺标识或邮编。发布延迟是假设，现有数据没有当时的发布时间。')
    story.append(table([['环节','时段与有效样本'],['训练 / 验证',f"2020—2022：{META['splits']['train']:,}；2023 年 1—11 月：{META['splits']['validation']:,}"],['重训 / 留出测试','重训 17,510；2024—2025 测试 11,869']],[155,340]))
    add('2024 年初选模型时，12 月标签尚未完整可用，验证排除整个 12 月；训练与重训同样遵守 14 天延迟。已可用的 12 月标签可用于重训。测试期模型固定，历史特征按月更新。','small')
    story.append(PageBreak())

    add('02 / 资源分配','kicker');add('命中率与地区覆盖，不能只选一个数字','h1')
    story.append(picture('03_coverage_tradeoff.png'))
    story.append(table([['20% 总名额下的策略','捕获 Fail','命中率','平均 TV'],['全部按模型分数','836','35.05%','0.291'],['50% 比例配额 + 分数补齐','804','33.71%','0.237'],['100% 比例配额，组内按分数','779','32.66%','0.069']],[245,75,85,90]))
    add('怎么分配','h2')
    add('按当月候选池的历史邮编数量占比分配预留名额，最大余数法取整；各邮编内按分数排序，剩余名额全局补齐。所有策略总名额相同；分数并列使用哈希实体键打破平局，不看结果。')
    add('怎样读这笔账','h2')
    add('50% 配额情景比纯分数排序少捕获 32 条 Fail，命中率下降 1.34 个百分点；平均地域构成偏离度下降约 18.5%。是否值得，应由明确的覆盖目标决定。比例约束只反映候选池构成，不是人口或需求公平。')
    add('TV = 0.5 × 各邮编“入选占比与候选占比之差”的绝对值之和；按月计算后取均值。低 TV 不代表每个地区获得相同待遇。','small')
    story.append(PageBreak())

    add('03 / 稳健性与失效边界','kicker');add('把增益和弱点放在同一张桌上','h1')
    story.append(table([['2023 年 1—11 月验证','命中率 @20%','ROC-AUC'],['逻辑回归（选中）',f"{VALIDATION.loc['logistic','precision']:.2%}",f"{VALIDATION.loc['logistic','roc_auc']:.4f}"],['梯度提升树',f"{VALIDATION.loc['hist_gradient_boosting','precision']:.2%}",f"{VALIDATION.loc['hist_gradient_boosting','roc_auc']:.4f}"]],[260,120,115]))
    story.append(Spacer(1,12));story.append(picture('04_monthly_stability.png'))
    add('不确定性','h2')
    add('模型比最强经验规则的测试命中率高 1.72 个百分点。对 24 个月做 2,000 次配对重采样，95% 区间为 +0.29 至 +3.05 个百分点。此区间仅反映月份抽样波动，未处理同店跨月相关性和原始检查选择偏差。')
    add('信息延迟与时间变化','h2')
    add('固定主分析所选逻辑回归，假设延迟 30 天时命中率降至 34.42%，经验规则为 33.38%；差值的 95% 区间为 -0.24 至 +2.30 个百分点，包含零。因此不能声称优势对信息延迟假设稳健。主分析的模型优势在 2024 / 2025 年分别为 +2.50 / +0.93 个百分点，值得持续监测。')
    add('无可用历史记录的对象是明显短板','h2')
    add('测试集中 76 条无历史记录，18 条为 Fail；模型只优先选中 5 条、捕获 1 条。无历史不必然等于新店，也不能据此认定适合新店。应单列必查或探索名额，再进行前瞻验证。')
    story.append(PageBreak())

    add('04 / 交付与决策建议','kicker');add('成果可以复现，结论必须有边界','h1')
    add('已交付','h2')
    add('官方数据快照与 SHA-256 来源清单；清洗、17 个历史特征和建模流程；3 条经验基线、随机期望与 2 个模型；预算曲线、地域配额和延迟敏感性；SQL 独立核验；逐条历史评分与线性贡献解释；离线交互展示。')
    add('项目建议','h2')
    add('将历史经验规则保留为透明基线，再评估模型的维护成本是否值得这 41 条额外捕获。配额比例由管理目标决定；对无历史对象另设必查或探索机制。没有完整候选名单和当时的信息快照前，不建议声称具备实用排程能力。')
    add('主要局限','h2')
    add('① 候选池只包含已受检且有有效结果的样本，无法推广到所有商户。<br/>② Fail 是行政结果，不是疾病或严重违规。<br/>③ 14 / 30 天延迟不能恢复真实历史发布时间，导出可能含后续修订。<br/>④ 地址、许可证变化与 2019 年之前历史截断可能影响实体及特征。<br/>⑤ 地区比例不是公平性证明；未评估检查员差异、交通成本、因果效应或真实业务收益。')
    add('来源与前人工作','h2')
    add('[1] City of Chicago. Food Inspections, dataset 4ijn-s7e5. data.cityofchicago.org/d/4ijn-s7e5；下载于 2026-10-10。<br/>[2] Chicago. Food Inspection Forecasting. github.com/Chicago/food-inspections-evaluation。<br/>[3] Kannan, Shapiro &amp; Bilgic (2019). Hindsight Analysis of the Chicago Food Inspection Forecasting Model. arXiv:1910.04906。<br/>[4] Singh, Shah, Kanich &amp; Kash (2022). Fair Decision-Making for Food Inspections. arXiv:2108.05523。','small')
    add('该选题已有前人工作。本项目重点是实现并核验信息时点、强基线和覆盖取舍，不声称学术首创，也不冒充官方项目。代码、数据来源与复现说明均已纳入公开项目仓库。','small')
    doc=SimpleDocTemplate(str(ROOT/'项目报告.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,
                          topMargin=45,bottomMargin=59,title='先查哪家？餐饮巡检风险排序与覆盖审计',author='个人研究项目')
    doc.build(story,onFirstPage=page,onLaterPages=page)

if __name__=='__main__':
    charts()
    cases=explain_rows()
    make_html(cases)
    make_pdf()
    print('Generated charts, HTML and PDF from verified result files.')
