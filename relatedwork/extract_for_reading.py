"""Local page-preserving PDF extraction for the related-work audit."""
from pathlib import Path
from pypdf import PdfReader
import json, hashlib

root=Path(__file__).resolve().parent
out=root/'reading_text'
out.mkdir(exist_ok=True)
records=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
index=[]
for r in records:
 data=(root/r['pdf_file']).read_bytes()
 assert hashlib.sha256(data).hexdigest()==r['pdf_sha256']
 reader=PdfReader(root/r['pdf_file'])
 folder=out/r['method']; folder.mkdir(exist_ok=True)
 pages=[]
 for i,p in enumerate(reader.pages,1):
  text=p.extract_text() or ''
  (folder/f'p{i:02d}.txt').write_text(text,encoding='utf-8')
  pages.append(f'\n\n===== PDF PAGE {i} =====\n\n'+text)
 (out/(r['method']+'.txt')).write_text(''.join(pages),encoding='utf-8')
 index.append(dict(method=r['method'],pages=len(pages),characters=sum(len(x) for x in pages),sha256=r['pdf_sha256'],pdf_file=r['pdf_file']))
 print(r['method'],len(pages),sum(len(x) for x in pages),flush=True)
(out/'extraction_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
