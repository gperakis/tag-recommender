import logging

import click

from tag_recommender.service.rest.factories import create_fastapi_application
from tag_recommender.service.utils import run_stress_test

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


@services.command()
@click.option(
    "--path",
    required=True,
    help="Path to the parquet file with test tags. "
         "Default is data/processed/test.parquet",
    default="data/processed/test.parquet",
)
@click.option("--workers", default=4, help="Number of workers. Default is 4.")
@click.option(
    "--requests",
    default=5000,
    help="Total number of requests to send. Default is 5000.",
)
@click.option(
    "--rate", default=200, help="Rate of requests per second (Hz). Default is 200."
)
def run_rest_stress_test(path, workers, requests, rate):
    run_stress_test(path, workers, requests, rate)
