import os
from openenv.core.env_server.http_server import create_app
from fastapi.responses import RedirectResponse
from sakha.models import SakhaAction, SakhaObservation
from sakha.env import SakhaEnvironment

os.environ["ENABLE_WEB_INTERFACE"] = "true"

app = create_app(SakhaEnvironment, SakhaAction, SakhaObservation, env_name="sakha")


@app.get("/")
async def root():
    return RedirectResponse(url="/web")


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
