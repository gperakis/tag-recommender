import logging

import click

from tag_recommender.service.rest.factories import create_fastapi_application

logger = logging.getLogger(__name__)


@click.group()
def services():
    # placeholder for the group command
    pass


@services.command()
@click.option("--host", default="0.0.0.0", type=str, show_default=True)
@click.option(
    "--port", default="8000", type=click.IntRange(0, 65536), show_default=True
)
def run_rest(host, port):
    import uvicorn

    app = create_fastapi_application()

    uvicorn.run(app, host=host, port=port)
