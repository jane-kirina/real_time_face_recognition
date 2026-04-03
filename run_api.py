import uvicorn
# Custom
from app.config import settings

def main():
    uvicorn.run(
        settings.api_application,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )

if __name__ == "__main__":
    main()