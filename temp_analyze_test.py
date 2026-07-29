from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app)
# Login as existing user
data={'email':'testuser1@example.com','password':'Password123'}
login_resp=client.post('/auth/login', json=data).json()
print('login status', login_resp)
headers={'Authorization':f"Bearer {login_resp['access_token']}"}
with open('temp.wav','rb') as f:
    files={'file':('temp.wav', f, 'audio/wav')}
    resp=client.post('/analyze/save', headers=headers, files=files)
    print('analyze/save status', resp.status_code)
    try:
        print(resp.json())
    except Exception as e:
        print('json error', e)
        print(resp.text)
