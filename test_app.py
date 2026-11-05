from pathlib import Path
from streamlit.testing.v1 import AppTest
root=Path(__file__).parent
for name in ['resnet18','mobilenet_v2']:
 at=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
 assert not at.exception,list(at.exception)
 at.selectbox[0].select(name).run()
 at.radio[0].set_value('Example').run()
 at.button[0].click().run()
 assert not at.exception,list(at.exception)
 assert not at.error,list(at.error)
 assert len(at.metric)==3
 print(name,[(m.label,m.value) for m in at.metric],flush=True)
print('PASS: both example recognition flows')
