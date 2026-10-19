#!/usr/bin/env python3
"""Matrix-based method figures, with one geometry source for EN/ZH and Draw.io/SVG.

python3 build_figures.py --export
"""
from __future__ import annotations
import argparse
import json
import subprocess
import shutil
from pathlib import Path
from xml.etree import ElementTree as E

HERE = Path(__file__).resolve().parent
BASE = json.loads((HERE / 'paper-palette.yaml').read_text())
C = BASE['semantic']
COMP = [C['accent'], C['success'], C['secondary'], C['danger']]
INK, MUTED, LINE, GRAY, GOLD = C['ink'], C['muted'], C['neutral'], C['stroke'], C['warning']
W, H = 1200, 650
SVG = 'http://www.w3.org/2000/svg'


def tint(color, strength=.12):
    rgb = [int(color[i:i+2], 16) for i in (1,3,5)]
    return '#' + ''.join(f'{round(v*strength+255*(1-strength)):02X}' for v in rgb)


class Scene:
    def __init__(self, name, lang, claim, height=H):
        self.name, self.lang, self.claim = name, lang, claim
        self.height = height
        self.nodes, self.edges, self.items = [], [], []
        self.font = 'Hiragino Sans GB' if lang == 'zh' else 'Arial'

    def words(self, zh, en): return zh if self.lang == 'zh' else en

    def node(self, key, x,y,w,h, shape='none', fill='none', stroke=LINE):
        self.nodes.append(dict(id=key,label=key,x=x,y=y,w=w,h=h,shape='rect',
                               draw_shape=shape,draw_fill=fill,draw_stroke=stroke,tone='neutral'))
        return key

    def text(self, x,y,text,w=150,size=18,bold=False,align='center',color=INK,parent=None):
        self.items.append(dict(kind='text',x=x,y=y,w=w,h=size*1.6,text=text,
                               size=size,bold=bold,align=align,color=color,parent=parent))

    def rect(self,x,y,w,h,fill='none',stroke=LINE,dashed=False,parent=None,round=False):
        self.items.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,
                               dashed=dashed,parent=parent,round=round))

    def edge(self,a,b,exit=(.5,1),entry=(.5,0),points=(),dashed=False,color=LINE,relation='forward',arrow=True):
        self.edges.append(dict(from_=a,to=b,exit=exit,entry=entry,points=points,dashed=dashed,
                               color=color,relation=relation,arrow=arrow))

    def lora(self,key,x,y,chosen,s=1,anchor=False,label=None):
        # A complete rank-4 adapter remains in the forward; only colored pairs train.
        width, height = 132*s, 90*s
        self.node(key,x,y,width,height)
        for k in range(4):
            color = tint(COMP[k],.28) if anchor else (COMP[k] if k in chosen else GRAY)
            self.rect(x+k*13*s,y+8*s,11*s,60*s,color,LINE,parent=key)
            self.rect(x+70*s,y+(10+k*13)*s,60*s,11*s,color,LINE,parent=key)
        self.text(x+24*s,y+70*s,'B' if label is None else label[0],w=45*s,parent=key)
        self.text(x+100*s,y+70*s,'A' if label is None else label[1],w=55*s,parent=key)
        return key

    def packet(self,key,x,y,chosen,s=1):
        # A packet contains selected factor columns/rows, not a dense delta-U.
        self.node(key,x,y,145*s,60*s)
        for j,k in enumerate(chosen):
            self.rect(x+j*14*s,y+4*s,11*s,40*s,COMP[k],LINE,parent=key)
            self.rect(x+66*s,y+(4+j*14)*s,54*s,11*s,COMP[k],LINE,parent=key)
        return key


def overview(lang):
    s=Scene('fig4_1_method',lang,'Amplify only local progress while communicating original selected factor changes.')
    t=s.text
    t(151,12,s.words('(a) 组件分配','(a) Component allocation'),w=280,size=22 if lang=='zh' else 20,bold=True)
    t(613,12,s.words('(b) 本地增量放大','(b) Amplified local forward'),w=560,size=22 if lang=='zh' else 20,bold=True)
    t(1058,12,s.words('(c) 上传与聚合','(c) Upload & aggregate'),w=260,size=22 if lang=='zh' else 20,bold=True)
    for x in [320,908]: s.rect(x,56,1,490,fill=GRAY,stroke='none')
    s.lora('global',95,90,set(range(4)),1,label=('Bₜ','Aₜ'))
    t(160,58,'Uₜ',w=70,size=20,bold=True)
    for key,x,chosen,label in [('weak1',23,{0},'rᵢ = 1'),('weak2',120,{1,2},'rᵢ = 2'),('strong',217,set(range(4)),'rᵢ = R')]:
        s.lora(key,x,275,chosen,.65)
        t(x+43,347,label,w=94,size=18)
        s.edge('global',key,points=[(161,228),(x+43,228)],relation='balanced rank-component allocation')
    # Local forward: W0*x + Ui*x + (alpha-1)*sg[(Ui-Ut)*x].
    s.node('input',584,474,26,26)
    t(597,474,'x',w=26,size=20,parent='input')
    s.node('output',584,66,26,26)
    t(597,66,'h',w=26,size=20,parent='output')
    s.node('sum',582,118,30,30,shape='ellipse',fill='#FFFFFF')
    t(597,117,'+',w=26,size=22,parent='sum')
    s.node('backbone',366,289,76,76,shape='rect',fill=GRAY)
    t(404,300,'W₀',w=65,size=20,parent='backbone')
    t(404,328,s.words('冻结','Frozen'),w=90,size=18,color=MUTED,parent='backbone')
    s.lora('local',535,293,{0},.9,label=('Bᵢ','Aᵢ'))
    t(699,280,'Uᵢ',w=65,size=20,bold=True)
    s.lora('anchor',761,354,set(),.82,anchor=True,label=('Bₜ','Aₜ'))
    t(737,322,s.words('锚点 Uₜ','Anchor Uₜ'),w=140,size=18)
    s.node('difference',801,233,30,30,shape='ellipse',fill='#FFFFFF')
    t(816,231,'−',w=26,size=22,parent='difference')
    t(858,266,'δUᵢ x',w=82,size=18)
    s.node('gain',795,112,42,42,shape='ellipse',fill=tint(GOLD,.18),stroke=GOLD)
    t(816,115,'×',w=36,size=22,parent='gain')
    t(816,71,'αᵢ − 1',w=120,size=20,bold=True)
    t(723,96,'sg',w=38,size=18,color=MUTED)
    # Stop-gradient is a barred symbol on only the extra branch.
    s.rect(715,123,1.5,20,LINE,stroke='none')
    s.rect(722,123,1.5,20,LINE,stroke='none')
    s.edge('input','backbone',exit=(.5,0),entry=(.5,1),points=[(597,451),(404,451)])
    s.edge('input','local',exit=(.5,0),entry=(.5,1))
    s.edge('input','anchor',exit=(.5,0),entry=(.5,1),points=[(597,451),(815,451)])
    s.edge('backbone','sum',exit=(.5,0),entry=(0,.5),points=[(404,133)])
    s.edge('local','sum',exit=(.5,0),entry=(.5,1))
    s.edge('local','difference',exit=(.5,0),entry=(0,.5),points=[(594,248)],relation='current local adapter output')
    s.edge('anchor','difference',exit=(.5,0),entry=(.5,1),relation='subtract cached round-start adapter output')
    s.edge('difference','gain',exit=(.5,0),entry=(.5,1),relation='current-round function increment')
    s.edge('gain','sum',exit=(0,.5),entry=(1,.5),relation='detached extra forward gain')
    s.edge('sum','output',exit=(.5,0),entry=(.5,1))
    # The selected client is expanded in panel b; the dotted edge is a zoom link.
    t(692,172,'αᵢ = R/rᵢ',w=180,size=20,bold=True)
    # Sparse uploads preserve factor/component identity across clients and server.
    for key,y,chosen in [('packet1',112,[0]),('packet2',199,[1,2]),('packet3',286,[0,1,2,3])]:
        s.packet(key,987,y,chosen,.88)
    t(1053,76,'ΔBᵢ,S       ΔAᵢ,S',w=232,size=18)
    s.node('uploads',1158,348,10,10,shape='ellipse',fill=LINE)
    # Transparent ownership container is connected to the raw-parameter uplink.
    for key in ['packet1','packet2','packet3']:
        s.edge(key,'uploads',exit=(1,.5),entry=(1,.5),points=[(1172,next(n['y']+n['h']/2 for n in s.nodes if n['id']==key)),(1172,353)],relation='selected factor changes collected by component',arrow=False)
    s.node('uplink',145,532,9,9,shape='ellipse',fill=LINE)
    for key in ['weak1','weak2','strong']:
        n=next(n for n in s.nodes if n['id']==key);cx=n['x']+n['w']/2
        s.edge(key,'uplink',exit=(1,.5),points=[(n['x']+n['w']+5, n['y']+n['h']/2),(n['x']+n['w']+5,491),(149.5,491)],relation='unamplified factor changes after local training',arrow=False,color=COMP[0])
    s.edge('uplink','uploads',exit=(1,.5),entry=(0,.65),points=[(931,536.5),(931,354.5)],relation='upload raw selected factor changes',color=COMP[0])
    t(375,503,'ΔAᵢ,S, ΔBᵢ,S',w=210,size=18,color=INK)
    s.lora('nextglobal',994,417,set(range(4)),1,label=('Bₜ₊₁','Aₜ₊₁'))
    t(955,420,s.words('聚合','Agg.'),w=70,size=18,bold=True)
    s.edge('uploads','nextglobal',exit=(.5,1),entry=(.5,0),relation='component-wise aggregation')
    s.edge('nextglobal','global',exit=(.5,1),entry=(0,.5),points=[(1060,584),(10,584),(10,135)],dashed=True,relation='broadcast next-round global adapter',color=MUTED)
    t(598,550,'t → t + 1',w=180,size=18,color=MUTED)
    # A single compact legend serves all three panels.
    s.rect(180,618,24,16,COMP[0],LINE)
    t(286,611,s.words('可训练组件','Trainable'),w=155,size=18)
    s.rect(418,618,24,16,GRAY,LINE)
    t(532,611,s.words('冻结 / 保留前向','Frozen / retained'),w=160,size=18)
    s.rect(674,618,24,16,tint(GOLD,.35),GOLD)
    t(796,611,s.words('本轮增量增益','Current-round gain'),w=180,size=18)
    return s


def aggregation(lang):
    s=Scene('fig4_2_aggregation',lang,'Coverage changes the component-wise normalization of the same raw factor uploads.',height=530)
    t=s.text
    t(193,16,s.words('(a) 相同的原始上传','(a) Same raw uploads'),w=350,size=22,bold=True)
    t(799,16,s.words('(b) 不同的归一化','(b) Different normalization'),w=700,size=22,bold=True)
    s.rect(406,65,1,395,GRAY,stroke='none')
    mask=[[1,0,0,0],[0,1,1,0],[1,1,1,1]]
    # General heterogeneous-rank branch: pi_i = r_i/R for weak clients, 1 for strong.
    hweights=[[4,0,0,0],[0,2,2,0],[1,1,1,1]]
    headings=['1','2','3','4']
    for k in range(4):t(160+k*53,78,headings[k],w=44,size=18)
    for i,row in enumerate(mask):
        t(71,145+i*65,['rᵢ=1','rᵢ=2','rᵢ=R'][i],w=90,size=18)
        for k,present in enumerate(row):
            s.rect(140+k*53,139+i*65,39,37,COMP[k] if present else '#FFFFFF',LINE,dashed=not present)
    t(233,333,'nₖ = (2, 2, 2, 1)',w=275,size=18)
    t(209,399,s.words('R = 4；等客户端权重','R = 4; equal client weights'),w=370,size=18,color=MUTED)
    # Semantic containers anchor the two normalization branches.
    s.node('raw',125,130,235,185)
    s.node('hajek',487,103,335,159)
    s.node('zerofill',487,310,335,159)
    s.node('hout',990,138,176,87)
    s.node('zout',990,345,176,87)
    s.edge('raw','hajek',exit=(1,.28),entry=(0,.5),points=[(443,181),(443,182.5)],relation='same raw uploads with inverse-inclusion weights')
    s.edge('raw','zerofill',exit=(1,.78),entry=(0,.5),points=[(443,274.3),(443,389.5)],relation='same raw uploads with missing contributions set to zero')
    for rowy,weights,title in [(111,hweights,'Hájek'),(318,mask,s.words('零填充收缩','Zero-fill shrinkage'))]:
        t(643,rowy-47,title,w=320,size=20,bold=True)
        for i,row in enumerate(weights):
            for k,value in enumerate(row):
                col=COMP[k]
                s.rect(514+k*65,rowy+i*48,49,37,tint(col,.15) if value else '#FFFFFF',LINE,dashed=not value)
                t(538.5+k*65,rowy+i*48+3,str(value),w=45,size=18)
        for k in range(4):
            s.rect(1010+k*38,rowy+38,28,55,COMP[k],LINE)
    t(899,114,'÷ (5, 3, 3, 1)',w=200,size=18)
    t(899,321,'÷ (3, 3, 3, 3)',w=200,size=18)
    s.edge('hajek','hout',exit=(1,.5),entry=(0,.5),relation='normalize by observed inverse-probability weight per component')
    s.edge('zerofill','zout',exit=(1,.5),entry=(0,.5),relation='normalize by all participant weights including unsent zeros')
    t(1056,242,s.words('分量归一化','Component-normalized'),w=230,size=18)
    t(1056,449,s.words('保留覆盖收缩','Coverage shrinkage'),w=230,size=18)
    t(625,495,s.words('矩阵内数字为加权系数；虚线格为未上传分量','Entries are aggregation weights; dashed cells are unsent components'),w=1100,size=18,color=MUTED)
    return s


def build(scene, export=False, output_dir=HERE, drawio_bin=None):
    """Build all editable/vector sources using only the Python standard library."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem=scene.name+('_zh' if scene.lang=='zh' else '_en')
    spec=dict(metadata=dict(title=scene.name,description=scene.claim,claim=scene.claim),
              paper_palette=dict(path='paper-palette.yaml'),
              figure=dict(column='double',layout='freeform',width=W,height=scene.height),
              nodes=scene.nodes,
              edges=[{'from':e['from_'],'to':e['to'],'exit':e['exit'],'entry':e['entry'],
                      'points':e['points'],'relation':e['relation']} for e in scene.edges])
    (output_dir/f'{stem}.scene.json').write_text(json.dumps(dict(**spec,language=scene.lang,font_family=scene.font,items=scene.items,relations=scene.edges),ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    figure=output_dir/f'{stem}.drawio'
    colors=[C[k] for k in ['accent','secondary','success','warning','danger']]
    mx=E.Element('mxfile', {'host':'app.diagrams.net', 'version':'26.0.0', 'type':'device',
        'data-title':scene.name, 'data-description':scene.claim, 'data-claim':scene.claim,
        'data-column':'double', 'data-width-mm':'178.0',
        'data-semantic-colors':','.join(colors), 'data-semantic-color-count':'5',
        'data-paper-palette-id':BASE['palette_id']})
    diagram=E.SubElement(mx,'diagram',id=stem,name='Figure')
    model=E.SubElement(diagram,'mxGraphModel',{'dx':'1000','dy':'700','grid':'1','gridSize':'10',
        'guides':'1','tooltips':'1','connect':'1','arrows':'1','fold':'1','page':'1','pageScale':'1',
        'pageWidth':str(W),'pageHeight':str(scene.height),'math':'0','shadow':'0','background':'#FFFFFF'})
    root=E.SubElement(model,'root')
    E.SubElement(root,'mxCell',id='0')
    E.SubElement(root,'mxCell',id='1',parent='0')
    canvas=E.SubElement(root,'mxCell',{'id':'canvas-bounds','value':'',
        'style':'rect;html=1;fillColor=#FFFFFF;strokeColor=none;pointerEvents=0;movable=0;resizable=0;connectable=0;',
        'vertex':'1','connectable':'0','parent':'1','data-canvas-boundary':'1'})
    E.SubElement(canvas,'mxGeometry',{'as':'geometry','x':'0','y':'0','width':str(W),'height':str(scene.height)})
    nodes={n['id']:n for n in scene.nodes}
    for idx,edge in enumerate(scene.edges,1):
        style=f"edgeStyle=orthogonalEdgeStyle;orthogonalLoop=1;jettySize=auto;rounded=0;html=0;strokeColor={edge['color']};strokeWidth=2.4;endArrow={'block' if edge['arrow'] else 'none'};endFill=1;startArrow=none;exitX={edge['exit'][0]};exitY={edge['exit'][1]};entryX={edge['entry'][0]};entryY={edge['entry'][1]};"+('dashed=1;dashPattern=5 4;' if edge['dashed'] else '')
        cell=E.SubElement(root,'mxCell',{'id':f'edge-{idx}','value':'','style':style,'edge':'1',
            'parent':'1','source':'node-'+edge['from_'],'target':'node-'+edge['to'],'data-relation':edge['relation']})
        geometry=E.SubElement(cell,'mxGeometry',{'as':'geometry','relative':'1','x':'0','y':'-12'})
        if edge['points']:
            points=E.SubElement(geometry,'Array',{'as':'points'})
            for x,y in edge['points']: E.SubElement(points,'mxPoint',x=str(float(x)),y=str(float(y)))
    for n in scene.nodes:
        shape=n['draw_shape']
        style=f"{'ellipse;' if shape=='ellipse' else 'rounded=0;'}container=1;collapsible=0;recursiveResize=0;html=0;fillColor={n['draw_fill']};strokeColor={n['draw_stroke'] if shape!='none' else 'none'};strokeWidth=2;shadow=0;glass=0;"
        cell=E.SubElement(root,'mxCell',{'id':'node-'+n['id'],'value':'','style':style,'vertex':'1',
            'parent':'1','data-node':'1','data-kind':'entity','data-icon':''})
        E.SubElement(cell,'mxGeometry',{'as':'geometry','x':f"{n['x']:.1f}",'y':f"{n['y']:.1f}",
            'width':f"{n['w']:.1f}",'height':f"{n['h']:.1f}"})
    for idx,item in enumerate(scene.items):
        attrs=dict(id=f'item-{idx}',vertex='1',parent='node-'+item['parent'] if item.get('parent') else '1')
        if item['kind']=='text':
            attrs.update(value=item['text'],style=f"text;html=0;whiteSpace=wrap;overflow=visible;spacing=0;align={item['align']};verticalAlign=middle;fontFamily={scene.font};fontSize={item['size']};fontStyle={1 if item['bold'] else 0};fontColor={item['color']};fillColor=none;strokeColor=none;")
        else:
            attrs['style']=f"rounded={1 if item.get('round') else 0};html=0;fillColor={item['fill']};strokeColor={item['stroke']};strokeWidth=2;"+('dashed=1;dashPattern=4 3;' if item.get('dashed') else '')
        cell=E.SubElement(root,'mxCell',attrs)
        # Text coordinates use its horizontal anchor; every label has explicit geometry.
        x=item['x']-(item['w']/2 if item.get('align')=='center' else 0)
        y=item['y']
        if item.get('parent'):
            n=nodes[item['parent']];x-=n['x'];y-=n['y']
        E.SubElement(cell,'mxGeometry',{'as':'geometry','x':str(x),'y':str(y),'width':str(item['w']),'height':str(item['h'])})
    tree=E.ElementTree(mx)
    E.indent(tree,space='  ');tree.write(figure,encoding='utf-8',xml_declaration=True)
    native_svg(scene,figure.with_suffix('.svg'),figure)
    if export:
        export_print_files(figure, drawio_bin)
    return stem


def native_svg(s,path,figure):
    """Write native SVG geometry and embed the editable, uncompressed Draw.io source."""
    E.register_namespace('',SVG)
    attrs={'version':'1.1','viewBox':f'0 0 {W} {s.height}',
           'content':figure.read_text(encoding='utf-8')}
    attrs['width']='178mm';attrs['height']=f'{178*s.height/W:.4f}mm'
    attrs['{http://www.w3.org/XML/1998/namespace}lang']='zh-CN' if s.lang=='zh' else 'en'
    root=E.Element(f'{{{SVG}}}svg',attrs)
    E.SubElement(root,f'{{{SVG}}}title').text=s.name
    E.SubElement(root,f'{{{SVG}}}desc').text=s.claim
    defs=E.SubElement(root,f'{{{SVG}}}defs')
    edgecolors=sorted(set(e['color'] for e in s.edges))
    for color in edgecolors:
        marker=E.SubElement(defs,f'{{{SVG}}}marker',id='arrow'+color[1:],viewBox='0 0 10 10',refX='9',refY='5',markerWidth='6',markerHeight='6',orient='auto-start-reverse')
        E.SubElement(marker,f'{{{SVG}}}path',d='M0 0L10 5L0 10Z',fill=color)
    E.SubElement(root,f'{{{SVG}}}rect',width=str(W),height=str(s.height),fill='#FFFFFF')
    nodes={n['id']:n for n in s.nodes}
    # Edges are behind objects, as in Draw.io.
    for e in s.edges:
        a,b=nodes[e['from_']],nodes[e['to']]
        coords=[(a['x']+a['w']*e['exit'][0],a['y']+a['h']*e['exit'][1]),*e['points'],(b['x']+b['w']*e['entry'][0],b['y']+b['h']*e['entry'][1])]
        attr={'d':'M'+'L'.join(f'{x:g},{y:g}' for x,y in coords),'fill':'none','stroke':e['color'],'stroke-width':'2.4','stroke-linejoin':'round'}
        if e['arrow']:attr['marker-end']='url(#arrow'+e['color'][1:]+')'
        if e['dashed']:attr['stroke-dasharray']='6 5'
        E.SubElement(root,f'{{{SVG}}}path',attr)
    for n in s.nodes:
        if n['draw_shape']=='none':continue
        if n['draw_shape']=='ellipse':
            E.SubElement(root,f'{{{SVG}}}ellipse',cx=str(n['x']+n['w']/2),cy=str(n['y']+n['h']/2),rx=str(n['w']/2),ry=str(n['h']/2),fill=n['draw_fill'],stroke=n['draw_stroke'],attrib={'stroke-width':'2'})
        else:E.SubElement(root,f'{{{SVG}}}rect',x=str(n['x']),y=str(n['y']),width=str(n['w']),height=str(n['h']),fill=n['draw_fill'],stroke=n['draw_stroke'],attrib={'stroke-width':'2'})
    for item in s.items:
        if item['kind']=='text':
            node=E.SubElement(root,f'{{{SVG}}}text',x=str(item['x']),y=str(item['y']+item['h']/2+item['size']*.35),fill=item['color'],attrib={'font-family':s.font+', sans-serif','font-size':str(item['size']),'font-weight':'700' if item['bold'] else '400','text-anchor':'middle' if item['align']=='center' else 'start'})
            node.text=item['text']
        else:
            attr={k:str(item[k]) for k in ['x','y','w','h']};attr['width']=attr.pop('w');attr['height']=attr.pop('h');attr.update(fill=item['fill'],stroke=item['stroke']);attr['stroke-width']='2'
            if item.get('dashed'):attr['stroke-dasharray']='4 3'
            E.SubElement(root,f'{{{SVG}}}rect',attr)
    E.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def export_print_files(figure, drawio_bin=None):
    """Optional PDF/PNG export through a public Draw.io Desktop installation."""
    executable = drawio_bin or shutil.which('drawio')
    mac_app = Path('/Applications/draw.io.app/Contents/MacOS/draw.io')
    if not executable and mac_app.is_file():
        executable = str(mac_app)
    if not executable:
        raise SystemExit('PDF/PNG export needs Draw.io Desktop: install it and put drawio on PATH, '
                         'or pass --drawio-bin. The scene JSON, Draw.io and native SVG files were built.')
    for fmt in ('pdf','png'):
        output = figure.with_suffix('.pdf' if fmt=='pdf' else '.drawio.png')
        cmd=[str(executable),'--export','--format',fmt,'--embed-diagram','--border','0']
        if fmt=='pdf': cmd += ['--crop']
        else: cmd += ['--width',str(round(178/25.4*400))]
        subprocess.run([*cmd,'--output',str(output),str(figure)],check=True,timeout=120)
        print('EXPORT', output.name, flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--export',action='store_true',help='also export PDF and 400 dpi PNG using Draw.io Desktop')
    p.add_argument('--drawio-bin',help='path to the Draw.io Desktop command (otherwise use PATH)')
    p.add_argument('--output-dir',type=Path,default=HERE,help='output directory (default: this script directory)')
    p.add_argument('--language',choices=['en','zh','both'],default='both')
    p.add_argument('--figure',choices=['overview','aggregation','both'],default='both')
    a=p.parse_args()
    for lang in ['zh','en'] if a.language=='both' else [a.language]:
        for make in ([overview,aggregation] if a.figure=='both' else [overview if a.figure=='overview' else aggregation]):
            print('BUILD',make.__name__,lang,flush=True)
            build(make(lang),a.export,a.output_dir,a.drawio_bin)

if __name__=='__main__':main()
