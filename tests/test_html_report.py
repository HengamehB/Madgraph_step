"""Exercise public report refresh and generate a portable review fixture."""
import csv
from html.parser import HTMLParser
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from report_data import load_saved_page
from report_view import refresh_report

SIGMA, ERR, BR = 1.1691955, 0.004566327258890234, 0.3392486
RANK = 'exports/reference/ranking.csv'


def save(root, name, data):
    (root / name).write_text(json.dumps(data, ensure_ascii=False),encoding='utf-8')


def sample(root, scope='selected'):
    root.mkdir(parents=True,exist_ok=True)
    (root/'exports/reference').mkdir(parents=True,exist_ok=True)
    (root/'.run.lock').touch()
    tables = {}
    def store(name,rows,path=None):
        path = path or (name + '.csv')
        with (root/path).open('w',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader()
            for row in rows:
                w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
        tables[name] = {'csv':{'path':path,'size_bytes':(root/path).stat().st_size}}
    names = ['x1','x1bar','x1 x1bar']
    channels = [dict(id='C000'+str(i+4),process='p p > '+name) for i,name in enumerate(names)]
    store('production_channels',channels)
    rates = [(SIGMA,ERR),(0.,0.),(None,None)]
    if scope=='complete': rates[2] = (.1,.003)
    if scope=='discovery': rates = [(None,None)]*3
    calc = [dict(channels[i],cross_section_pb=x,error_pb=e) for i,(x,e) in enumerate(rates) if x is not None]
    if calc: store('cross_sections',calc)
    modes = [dict(pdg=p,process=name+' > '+final,daughters=ends,width_gev=.2984155,branching_ratio=BR) for p,name,final,ends in [(9700001,'x1','u u',[2,2]),(-9700001,'x1bar','u~ u~',[-2,-2])]]
    store('decays',modes)
    store('total_widths',[dict(pdg=r['pdg'],name=r['process'].split()[0],width_gev=.879654062,omitted_probability=None) for r in modes])
    histories=[]
    for i,(tag,ends,label,history) in enumerate([('Hbenchmark',[2,2],'u u','x1 → (u u)'),('Hzero',[-2,-2],'u~ u~','x1bar → (u~ u~)'),('Hpair',[-11,12,2,-2],'e+ ve u u~','Synthetic cascade')]):
        sigma,error = rates[i]; prob=BR if i<2 else BR**2
        histories.append(dict(history_id=tag,production_process=channels[i]['process'],production_id=channels[i]['id'],final_pdgs=ends,final_state=label,history=history,branching_ratio=prob,production_cross_section_pb=sigma,production_error_pb=error,rate_pb=sigma*prob if sigma is not None else None,error_pb=error*prob if error is not None else None,flags=[],history_ref=[]))
    histories.sort(key=lambda r:(r['rate_pb'] is None,-(r['rate_pb'] or 0),r['history_id']))
    store('ranking',histories,RANK)
    save(root,'spectrum.json',dict(masses={'H':125,'X1':1500,'Y1':1200},scalar_states=[dict(label='X1',charge='4/3')]))
    save(root,'selected_operators.json',[dict(id='O001',field_content='Q² X†',coefficient_dimension=0)])
    save(root,'resolved_inputs.json',dict(coefficients={'O001':[.1,0,0,.1,0,.1]}))
    m=dict(schema='madgraph-step-run-v1',model='Triplet–sextet report fixture',status='complete' if scope=='complete' else 'histories_complete' if scope=='discovery' else 'partial',run_id='reference',requested_complete=scope!='discovery',all_rates_complete=scope=='complete',discovery_complete=True,history_counts=dict(total=3,calculated=len(calc)),current_tables=tables,uncomputed_production_ids=[channels[i]['id'] for i,r in enumerate(rates) if r[0] is None],settings=dict(collider=dict(energy_tev=13.6,dynamical_scale_choice=3,pdf='nn23lo1'),decays=dict(max_body=2,precision=.01)),failures=[])
    save(root,'manifest.json',m)
    return m


class Document(HTMLParser):
    def __init__(self,path):
        super().__init__(); self.links=[]; self.rows=[]; self.text=[]; self.row=None
        self.feed(path.read_text(encoding='utf-8'))
    def handle_starttag(self,tag,attrs):
        if tag=='a': self.links.append(dict(attrs).get('href'))
        if tag=='tr': self.row=[]
    def handle_data(self,data):
        self.text.append(data)
        if self.row is not None: self.row.append(data)
    def handle_endtag(self,tag):
        if tag=='tr' and self.row is not None:
            self.rows.append(' '.join(self.row)); self.row=None


class ReportTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); self.root=Path(tmp.name)

    def test_copied_public_cli_without_tools(self):
        old=self.root/'original'; sample(old); new=self.root/'moved run #2'; shutil.copytree(old,new); shutil.rmtree(old)
        def snapshot():
            return {p.relative_to(new):(p.read_bytes(),p.stat().st_mtime_ns) for p in new.rglob('*') if p.is_file() and p.name!='report.html'}
        before=snapshot()
        subprocess.run([str(ROOT/'run.sh'),'report','--run-dir',str(new)],env=dict(os.environ,PYTHON=sys.executable),cwd=self.root,check=True,timeout=30)
        self.assertEqual(before,snapshot()); doc=Document(new/'report.html'); self.assertIn(RANK,doc.links)
        benchmark=next(r for r in doc.rows if 'Hbenchmark' in r)
        for n in ('1.169196','33.92486%','0.3966479','0.00154912'): self.assertIn(n,benchmark)
        self.assertIn('0 ± 0',next(r for r in doc.rows if 'Hzero' in r))
        self.assertIn('Not calculated',next(r for r in doc.rows if 'Hpair' in r))

    def test_modes_and_partial_metadata(self):
        for scope,count in [('complete',3),('discovery',0)]:
            root=self.root/scope; sample(root,scope); s=refresh_report(root).read_text(); self.assertIn(f'{count} of 3',s)
        root=self.root/'failed'; root.mkdir(); save(root,'manifest.json',dict(schema='madgraph-step-run-v1',status='failed',failures=[dict(stage='input',message='<invalid>')]))
        self.assertIn('&lt;invalid&gt;',refresh_report(root).read_text())

    def test_first_page_bound_and_containment(self):
        m=sample(self.root)
        def forever(*args):
            row=dict(history_id='H',rate_pb='',branching_ratio='0.1',final_state='u u')
            for i in range(101): yield dict(row,history_id=str(i))
            raise AssertionError('Read beyond the bounded prefix')
        with patch('report_data.table_rows',forever):
            page=load_saved_page(self.root,m); self.assertEqual(len(page['rows']),100); self.assertTrue(page['more'])
        m['current_tables']['ranking']['csv']['path']='../private.csv'
        m['current_tables']['cross_sections']['csv']['path']='missing.csv'; save(self.root,'manifest.json',m)
        doc=Document(refresh_report(self.root)); self.assertNotIn('../private.csv',doc.links); self.assertNotIn('missing.csv',doc.links); self.assertNotIn('ranking.csv',doc.links)


if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--preview':
        base=Path(sys.argv[2]);
        for label in ('selected','complete','discovery'):
            folder=base/label; sample(folder,label); refresh_report(folder)
    else:
        unittest.main()
