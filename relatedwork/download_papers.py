"""Download verified arXiv papers and original TeX sources. No source is executed."""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, json, re, hashlib, tarfile, io, gzip, csv

ROOT = Path(__file__).resolve().parent
# Priority is an editorial relevance judgement, not a claim of equivalent protocols.
PAPERS = [
 ('FedSA-LoRA','2410.01463','特别接近','HetSA 的共享 A、本地 B 直接基础；须对照贡献边界。',True),
 ('Fed-HeLLo','2506.12213','特别接近','按资源与层重要性分配 LoRA 层；PCLA-WL 和异构层预算的直接近邻。',True),
 ('LEGEND','2412.20004','特别接近','联合配置 LoRA 深度和秩；对照后层优先与真实设备成本。',True),
 ('Fed-pilot','2410.10200','特别接近','内存约束下的层分配背包优化及异构聚合；对照成本感知选择。',True),
 ('Fed-PLoRA','2602.16936','特别接近','并行 rank-1 模块与异构客户端；与组件级 PCLA 的分配单位直接相关。',True),
 ('FedAS-LoRA','2608.09742','特别接近','比较共享 A 或 B 并自适应选择；检查固定 Share-A 的适用条件。',True),
 ('FedSAIL','2609.32485','特别接近','输入感知共享子空间和本地系数；检查共享因子解释与个性化边界。',True),
 ('AS-LoRA','2605.05769','特别接近','逐层逐轮自适应选择 A/B 更新因子；这里的 component 不是 PCLA 的 rank-1 组件。',True),
 ('RoLoRA','2502.01755','特别接近','交替优化 A/B；对照 PCLA 交替训练机制，不能等同于保留本地 B。',True),
 ('FDLoRA','2406.07925','特别接近','全局与本地双 LoRA 个性化及融合；对照 HetSA 的本地知识保留。',True),
 ('HetLoRA','2401.06432','强相关','异构秩、自剪枝和聚合；与异构预算、组件选择相关。',False),
 ('FlexLoRA','2402.11505','强相关','全尺寸更新聚合后 SVD 重分配异构秩；异构聚合对照。',False),
 ('FLoRA','2409.05976','强相关','堆叠聚合异构 LoRA，讨论因子分别平均的聚合误差。',False),
 ('FFA-LoRA','2403.12313','强相关','冻结 A、训练并共享 B；与共享 A、本地 B 的设计区别。',False),
 ('HAFLQ','2411.06581','特别接近','重要性感知截断、冻结及仅聚合更新过的 rank-1 矩阵；组件级 PCLA 的直接近邻，另包含异构量化。',True),
 ('SLoRA','2308.06522','强相关','非 IID 下联邦 PEFT 与数据驱动初始化。',False),
 ('LoRA','2106.09685','基础','低秩适配的基本矩阵定义与参数效率。',False),
 ('AdaLoRA','2303.10512','基础','重要性感知的自适应秩预算；非联邦基础工作。',False),
 ('FedIT','2305.05644','基础','联邦指令微调背景；不要将项目中统一 rank 的 fedlora 控制等同于完整论文复现。',False),
 ('FedAvg','1602.05629','基础','联邦平均及非 IID 问题的基础。',False),
]

class MetaParser(HTMLParser):
 def __init__(self): super().__init__(); self.meta={}
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if tag=='meta' and d.get('name','').startswith('citation_'):
   self.meta.setdefault(d['name'],[]).append(d.get('content',''))

def get(url):
 r=requests.get(url,timeout=(15,60),headers={'User-Agent':'FYP-relatedwork/1.0 (academic literature collection)'})
 r.raise_for_status(); return r

def save_bytes(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and path.read_bytes()!=data:
  raise RuntimeError('Refusing to overwrite different existing file: '+str(path))
 path.write_bytes(data)

def unpack_source(data,dest):
 dest.mkdir(parents=True,exist_ok=True)
 if tarfile.is_tarfile(io.BytesIO(data)):
  with tarfile.open(fileobj=io.BytesIO(data),mode='r:*') as tf:
   for member in tf.getmembers():
    p=PurePosixPath(member.name)
    if p.is_absolute() or '..' in p.parts or ':' in member.name or '\\' in member.name:
     raise ValueError('Unsafe archive member: '+member.name)
    target=dest.joinpath(*p.parts)
    if not target.resolve().is_relative_to(dest.resolve()): raise ValueError('Unsafe target')
    if member.isdir(): target.mkdir(parents=True,exist_ok=True)
    elif member.isfile():
     if member.size>100_000_000: raise ValueError('Oversized member')
     save_bytes(target,tf.extractfile(member).read())
    # Links and special files are never extracted.
 else:
  raw=gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data
  if b'\\documentclass' not in raw and b'\\begin{document}' not in raw:
   raise ValueError('Response is not a TeX source archive')
  save_bytes(dest/'main.tex',raw)
 tex=sorted(str(p.relative_to(ROOT)).replace('\\','/') for p in dest.rglob('*.tex'))
 if not tex: raise ValueError('No .tex file in source')
 mains=[p for p in tex if b'\\documentclass' in (ROOT/p).read_bytes() and b'\\begin{document}' in (ROOT/p).read_bytes()]
 return tex,mains

def download(item):
 name,aid,priority,relation,want_tex=item
 record=dict(method=name,arxiv_id=aid,priority=priority,relation_zh=relation,checked_date='2026-09-30',reading_depth='官网元数据和摘要筛选；未完整审读或复现实验',pdf_status='pending',tex_status='not_requested')
 try:
  page=get('https://arxiv.org/abs/'+aid)
  parser=MetaParser(); parser.feed(page.text); meta=parser.meta
  if not meta.get('citation_title'): raise ValueError('Missing official title metadata')
  versions=re.findall(r'/abs/'+re.escape(aid)+r'(v\d+)',page.text)
  version=max(versions,key=lambda v:int(v[1:])) if versions else ''
  pinned=aid+version
  record.update(title=meta['citation_title'][0],authors=meta.get('citation_author',[]),date=meta.get('citation_date',[''])[0],version=version,url='https://arxiv.org/abs/'+pinned,pdf_url='https://arxiv.org/pdf/'+pinned)
  record['official_metadata']=meta
  abstract=re.search(r'<blockquote[^>]*class="abstract[^>]*>(.*?)</blockquote>',page.text,re.S)
  if abstract:
   import html
   record['abstract']=html.unescape(re.sub('<[^>]+>',' ',abstract.group(1))).strip()
  folder='01_closest' if want_tex else ('02_related' if priority=='强相关' else '03_foundations')
  pdf_path=ROOT/folder/(name+'_'+pinned+'.pdf')
  pdf=get(record['pdf_url']).content
  if not pdf.startswith(b'%PDF-') or b'%%EOF' not in pdf[-2048:]: raise ValueError('Incomplete or invalid PDF')
  save_bytes(pdf_path,pdf)
  record.update(pdf_status='ok',pdf_file=str(pdf_path.relative_to(ROOT)).replace('\\','/'),pdf_bytes=len(pdf),pdf_sha256=hashlib.sha256(pdf).hexdigest())
  if want_tex:
   record['tex_status']='pending'; record['tex_url']='https://arxiv.org/src/'+pinned
   try:
    source=get(record['tex_url']).content
    if source.lstrip().startswith(b'<'): raise ValueError('HTML returned instead of source')
    src_path=ROOT/'tex'/ (name+'_'+pinned)
    tex,mains=unpack_source(source,src_path)
    archive_path=ROOT/'source_archives'/(name+'_'+pinned+'.source')
    save_bytes(archive_path,source)
    record.update(tex_status='ok',tex_files=tex,tex_main_candidates=mains,source_archive=str(archive_path.relative_to(ROOT)).replace('\\','/'),source_sha256=hashlib.sha256(source).hexdigest(),tex_compilation='未编译；原作者多文件工程，仅下载与安全解包')
   except Exception as e: record.update(tex_status='failed',tex_error=str(e))
 except Exception as e: record['error']=str(e); record['pdf_status']='failed' if record['pdf_status']!='ok' else 'ok'
 return record

def main():
 records=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  futures={pool.submit(download,p):p for p in PAPERS}
  for future in as_completed(futures):
   record=future.result(); records.append(record)
   print(record['method']+': PDF='+record['pdf_status']+' TeX='+record['tex_status']+' '+record.get('error',record.get('tex_error','')),flush=True)
   (ROOT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 write_catalog(records)
 print('TOTAL PDF:',sum(r['pdf_status']=='ok' for r in records),'TEX papers:',sum(r['tex_status']=='ok' for r in records),flush=True)

def write_catalog(records):
 order={p[0]:i for i,p in enumerate(PAPERS)}; records.sort(key=lambda r:order[r['method']])
 (ROOT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
 with (ROOT/'paper_index.csv').open('w',newline='',encoding='utf-8-sig') as f:
  fields=['method','priority','title','arxiv_id','version','date','relation_zh','url','pdf_file','pdf_status','tex_status']
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(records)
 bib=[]
 for r in records:
  if 'title' not in r: continue
  escape=lambda s:s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
  bib.append('@misc{'+r['method'].replace('-','')+r['arxiv_id'][:2]+',\n  title={'+escape(r['title'])+'},\n  author={'+escape(' and '.join(r['authors']))+'},\n  year={'+(r['date'][:4] or '20'+r['arxiv_id'][:2])+'},\n  eprint={'+r['arxiv_id']+'},\n  archivePrefix={arXiv},\n  url={'+r['url']+'}\n}')
 (ROOT/'references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')

if __name__=='__main__': main()
