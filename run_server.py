import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent / 'backend'))

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    # IIS HttpPlatformHandler provides HTTP_PLATFORM_PORT
    # Fall back to PORT, then default to 8000
    port = int(
        os.environ.get('HTTP_PLATFORM_PORT')
        or os.environ.get('PORT', 8000)
    )

    print(f"Unified Document Extractor API - Starting on port {port}")

    from main import app

    uvicorn.run(
        app,
        host='127.0.0.1',
        port=port,
        reload=False,
        workers=1
    )