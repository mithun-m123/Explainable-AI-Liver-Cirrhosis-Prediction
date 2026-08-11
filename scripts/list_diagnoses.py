from pathlib import Path
import sys
project_root = Path('.').resolve()
sys.path.insert(0, str(project_root / 'backend'))
sys.path.insert(0, str(project_root))
from database import SessionLocal, DiagnosisRecord
s=SessionLocal()
ds = s.query(DiagnosisRecord).all()
for d in ds:
    print(d.id, d.user_id, d.created_at)
