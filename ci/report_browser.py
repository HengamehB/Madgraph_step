"""Check the generated offline report in the runner's installed Chromium."""
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import sleep
from urllib.request import Request, urlopen


def request(method, url, value=None):
    data = json.dumps(value).encode() if value is not None else None
    with urlopen(Request(url,data=data,method=method,headers={'Content-Type':'application/json'}),timeout=30) as response:
        return json.load(response)['value']


def check(folder):
    out=folder/'screens'; out.mkdir(exist_ok=True)
    server='http://127.0.0.1:9515'
    with TemporaryDirectory() as profile, subprocess.Popen(['chromedriver','--port=9515'],stdout=subprocess.DEVNULL) as driver:
        try:
            for _ in range(80):
                try:
                    if request('GET',server+'/status')['ready']: break
                except OSError: sleep(.1)
            opts={'binary':shutil.which('google-chrome'),'args':['--headless=new','--no-sandbox','--disable-dev-shm-usage','--allow-file-access-from-files','--user-data-dir='+profile]}
            session=request('POST',server+'/session',{'capabilities':{'alwaysMatch':{'browserName':'chrome','goog:chromeOptions':opts}}})['sessionId']
            base=server+'/session/'+session
            def cmd(endpoint,data): return request('POST',base+endpoint,data)
            def js(script): return cmd('/execute/sync',{'script':script,'args':[]})
            def screenshot(name): (out/(name+'.png')).write_bytes(base64.b64decode(request('GET',base+'/screenshot')))
            for name,w,h in [('desktop',1400,1000),('mobile',390,844)]:
                cmd('/goog/cdp/execute',{'cmd':'Emulation.setDeviceMetricsOverride','params':{'width':w,'height':h,'deviceScaleFactor':1,'mobile':False}})
                cmd('/url',{'url':(folder/'selected/report.html').resolve().as_uri()})
                assert js('return document.documentElement.scrollWidth <= innerWidth+1'), name+' page overflows'
                assert js('return !document.querySelector("#filters").hidden'), 'Filter controls failed to initialize'
                screenshot(name); js('document.querySelector("#ranked").scrollIntoView({behavior:"instant"})'); screenshot(name+'-ranking')
                assert js('let sel=document.querySelector("#signature");sel.value="nu";sel.dispatchEvent(new Event("change"));return document.querySelectorAll("#ranked tbody tr:not([hidden])").length')==1
                assert js('let sel=document.querySelector("#signature");sel.value="";sel.dispatchEvent(new Event("change"));let ex=document.querySelector("#exact");ex.value="-2,-2";ex.dispatchEvent(new Event("change"));return document.querySelector("#ranked tbody tr:not([hidden])").textContent.includes("Hzero")')
            cmd('/url',{'url':(folder/'selected/report.html').resolve().as_uri()})
            pdf=cmd('/goog/cdp/execute',{'cmd':'Page.printToPDF','params':{'printBackground':True,'preferCSSPageSize':True}})['data']
            (out/'report-print.pdf').write_bytes(base64.b64decode(pdf))
            print('Desktop, mobile, signed-particle and neutrino filters, and print rendering passed.')
        finally:
            driver.terminate()


if __name__=='__main__': check(Path(sys.argv[1]))
