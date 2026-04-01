from openenv.core.env_server.http_server import create_app

from sakha.env import SakhaEnvironment
from sakha.models import SakhaAction, SakhaObservation

app = create_app(SakhaEnvironment, SakhaAction, SakhaObservation, env_name="sakha")


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
