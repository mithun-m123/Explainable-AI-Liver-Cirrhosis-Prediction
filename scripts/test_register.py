import requests, json
url = 'http://127.0.0.1:8000/auth/register'
payload = {
    'email':'test+reg@example.com',
    'password':'Short',
    'full_name':'Test Reg',
    'role':'patient'
}
try:
    r = requests.post(url, json=payload, timeout=10)
    print('status', r.status_code)
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)
except Exception as e:
    print('error', e)
