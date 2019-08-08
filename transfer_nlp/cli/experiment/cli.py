import importlib
import logging
import sys
from pathlib import Path

import click

from transfer_nlp.plugins.config import ExperimentConfig


@click.command()
@click.option(
    '--exp_file',
    help='Path to the experiment file containing the description of all objects used in the experiment'
         'If not provided, a prompt will allow you to type the path',
)
@click.option(
    '--module',
    help='Path to the module tat contains registrables',
)
@click.option(
    '--debug',
    help='Print Transfer NLP logging for debugging',
)
def experiment(exp_file, module, debug):
    # Set Transfer NLP logs
    if debug:
        click.echo("Using Transfer NLP lgs")
        logging.getLogger("transfer_nlp").setLevel("INFO")
    else:
        click.echo("Not using Transfer NLP lgs")

    # Import the registrables so that they are registered
    # TODO: an alternative would be to not use at all the decorators, put all registrables in "module" and register them here
    module = Path.cwd() / str(module)
    sys.path.append(module)
    module_name = str(module).split("/")[-1]
    try:
        click.echo("Loading registrables")
        # Credit: https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        spec = importlib.util.spec_from_file_location(module_name, module)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        click.echo("Loaded egistrables")

    except Exception as e:
        click.echo(f"Import error: {e}")

    # Load the experiment
    # TODO: see if we can have flexible number of arguments like *args for env variables
    home = str(Path.home() / 'work/transfer-nlp-data')
    if exp_file:
        e = ExperimentConfig(experiment=Path.cwd() / str(exp_file),
                             HOME=home)
    else:
        exp_file = click.prompt('Enter a text')
        e = ExperimentConfig(experiment=Path.cwd() / str(exp_file),
                             HOME=home)

    click.echo("Experiment successfully loaded!")

    # Training part
    click.echo("Launching the experiment")
    e['trainer'].train()

    click.echo("End of experiment")


if __name__ == '__main__':
    experiment()
