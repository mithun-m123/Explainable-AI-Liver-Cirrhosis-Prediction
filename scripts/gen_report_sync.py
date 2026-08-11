import asyncio
from pathlib import Path
import sys
# Ensure backend package imports work when running from project root
project_root = Path('.').resolve()
sys.path.insert(0, str(project_root / 'backend'))
sys.path.insert(0, str(project_root))
from pdf_gen import report_generator

async def main():
    try:
        path = await report_generator.generate_report(6, 8)
        print('Generated:', path)
    except Exception as e:
        print('Error during generation:', e)

if __name__ == '__main__':
    asyncio.run(main())
