from pathlib import Path

import click

from tag_recommender.logger import initialize_logging
from tag_recommender.process.cli import data
from tag_recommender.service.cli import services


@click.group()
def cli():
    path = Path("config", "logging.yaml")
    initialize_logging(path)


# attach retrieval services cli
cli.add_command(services)
cli.add_command(data)

if __name__ == "__main__":
    cli()
