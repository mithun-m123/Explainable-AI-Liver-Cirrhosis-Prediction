import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database import User
from pathlib import Path

# Use the backend folder's DB so we inspect the same file the server uses
project_root = Path(__file__).resolve().parents[1]
db_path = project_root / 'backend' / 'liver_cirrhosis.db'
db_url = f"sqlite:///{db_path.as_posix()}"

engine = create_engine(db_url, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
s = Session()
users = s.query(User).all()
for u in users:
    print(u.id, u.email, u.full_name, u.role, u.is_active)
