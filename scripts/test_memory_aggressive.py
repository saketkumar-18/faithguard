import os
os.environ.setdefault('FG_LOG_LEVEL', 'WARNING')
os.environ.setdefault('FG_USE_DENSE', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('FG_NLI_BATCH', '1')
os.environ.setdefault('FG_TOP_K', '2')
os.environ.setdefault('FG_MAX_DOCS', '15')
os.environ.setdefault('FAITHGUARD_DATA_DIR', 'data')
os.environ.setdefault('FAITHGUARD_MODELS_DIR', 'models')

import psutil, threading, time
proc = psutil.Process()
peak=[0]; done=[False]
def mon():
    while not done[0]:
        m=proc.memory_info().rss
        if m>peak[0]: peak[0]=m
        time.sleep(0.02)
threading.Thread(target=mon,daemon=True).start()

from faithguard.api.app import create_app
app = create_app(load_default_corpus=True)

from fastapi.testclient import TestClient
with TestClient(app) as client:
    print('startup: rss=%.0f MB peak=%.0f MB' % (proc.memory_info().rss/1e6, peak[0]/1e6))
    r = client.post('/ask', json={'question': 'What is the Taj Mahal?'})
    print('ask:', r.status_code, r.json().get('hallucinated_initial'))
    print('after /ask: rss=%.0f MB PEAK=%.0f MB' % (proc.memory_info().rss/1e6, peak[0]/1e6))
done[0]=True