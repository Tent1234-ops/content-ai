from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
import random
client = TestClient(app)
username = f'user_{random.randint(1000,9999)}'
email = f'user_{random.randint(1000,9999)}@example.com'
password = 'Test1234!'
print('register', email)
r = client.post('/auth/register', json={'username': username, 'email': email, 'password': password})
print('reg status', r.status_code, r.text)
if r.status_code not in (200, 201):
    raise SystemExit('register failed')
r = client.post('/auth/login', json={'email': email, 'password': password})
print('login status', r.status_code, r.text)
if r.status_code != 200:
    raise SystemExit('login failed')
access_token = r.json()['access_token']
headers = {'Authorization': f'Bearer {access_token}'}
print('calling /contents/my')
r = client.get('/contents/my', headers=headers)
print('/contents/my', r.status_code, r.text)
print('calling /recommendations/from-content/1')
r2 = client.get('/recommendations/from-content/1?source=youtube&profile_limit=150', headers=headers)
print('/recommendations/from-content/1', r2.status_code, r2.text)
path = Path('temp.wav')
if path.exists():
    print('uploading temp.wav', path.stat().st_size)
    with path.open('rb') as f:
        resp = client.post('/analyze/save', headers=headers, files={'file':('temp.wav', f, 'audio/wav')})
    print('/analyze/save', resp.status_code)
    try:
        print(resp.json())
    except Exception as e:
        print('json error', e)
        print(resp.text)
else:
    print('temp.wav missing')
