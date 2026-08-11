import requests, json, time
BASE='http://127.0.0.1:8000'
email=f'test.chat.{int(time.time())}@example.com'
password='TestPass123!'
# Register
try:
    r=requests.post(f'{BASE}/auth/register', json={'email':email,'password':password,'full_name':'Chat Test','role':'patient'}, timeout=10)
    print('register', r.status_code, r.text)
except Exception as e:
    print('register error', e)
# Login
try:
    r=requests.post(f'{BASE}/auth/login', json={'email':email,'password':password}, timeout=10)
    print('login', r.status_code)
    token = r.json().get('access_token')
    headers={'Authorization':f'Bearer {token}'}
    # Start chat
    r2=requests.post(f'{BASE}/chat/start', json={'session_type':'symptom_collection','context':{}}, headers=headers, timeout=10)
    print('start chat', r2.status_code)
    print(json.dumps(r2.json(), indent=2))
    # Send message
    r3=requests.post(f'{BASE}/chat/message', params={'session_id': r2.json().get('session_id'),'message':'I have fatigue and jaundice'}, headers=headers, timeout=10)
    print('send', r3.status_code)
    print(json.dumps(r3.json(), indent=2))
except Exception as e:
    print('login/chat error', e)
