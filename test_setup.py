#!/usr/bin/env python
"""
Quick test to verify the project structure and all imports work
"""
import sys

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        import fastapi
        print("✓ fastapi")
        import pydantic
        print("✓ pydantic")
        import sqlalchemy
        print("✓ sqlalchemy")
        import jose
        print("✓ python-jose")
        import passlib
        print("✓ passlib")
        print("\nAll imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_project_structure():
    """Test if project structure is correct"""
    import os
    required_dirs = [
        "backend",
        "backend/models",
        "backend/uploads",
        "backend/utils",
        "ai_pipeline",
        "ai_pipeline/datasets",
        "frontend",
        "frontend/src/components",
        "frontend/src/pages"
    ]
    
    print("\nChecking project structure...")
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - MISSING")
    
    required_files = [
        "requirements.txt",
        "README.md",
        "backend/main.py",
        "backend/models_handler.py",
        "backend/schemas.py",
        "backend/database.py",
        "backend/auth.py",
        "backend/utils/pdf_gen.py",
        "backend/utils/chatbot_api.py",
        "backend/utils/xai_logic.py",
    ]
    
    print("\nChecking required files...")
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")

if __name__ == "__main__":
    test_project_structure()
    if test_imports():
        print("\n✓ Project is ready to run!")
        print("\nTo start the backend server, run:")
        print("  cd backend")
        print("  python main.py")
        print("\nOr use uvicorn directly:")
        print("  uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("\n✗ Please install dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
