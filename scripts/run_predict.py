import requests, json, os

# Login
r = requests.post('http://127.0.0.1:8000/auth/login', json={'email':'testuser@example.com','password':'Password123'})
print('login', r.status_code)
if r.status_code != 200:
    print(r.text)
    raise SystemExit(1)

token = r.json().get('access_token')
headers = {'Authorization': f'Bearer {token}'}

symptoms = {
  'age': 45,
  'sex': 1,
  'fatigue_level': 5,
  'alcohol_consumption': 2.0,
  'weight_loss_kg': 0.0,
  'abdominal_swelling': False,
  'appetite_loss': False,
  'jaundice': False,
  'fever': False,
  'ascites': 0,
  'hepatomegaly': 0,
  'spiders': 0,
  'edema': 0,
  'bilirubin': 0.8,
  'cholesterol': 210,
  'albumin': 3.5,
  'copper': 50,
  'alk_phos': 75,
  'ast': 40,
  'triglycerides': 150,
  'platelets': 250000,
  'prothrombin': 11
}

files = {'symptoms': (None, json.dumps(symptoms))}
resp = requests.post('http://127.0.0.1:8000/predict', headers=headers, files=files)
print('predict', resp.status_code)
print(resp.text)

print('reports exist:', os.path.exists('reports'))
if os.path.exists('reports'):
    print(os.listdir('reports'))
else:
    print('no reports folder')
