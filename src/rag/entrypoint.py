from .logging_config import *
import click
from .commands.deploy import deploy_cmd
from .commands.invoke import invoke_cmd
from .commands.clean import clean_cmd
from .commands.check import check_cmd


@click.group()
def cli():
    """CLI command to manage RAG"""
    pass
    
@cli.command()
def deploy():
    """Spins up AWS resources"""
    click.echo("Deploying RAG")
    deploy_cmd()

@cli.command()
def invoke():
    """Chat with the RAG system"""
    invoke_cmd()


@cli.command()
def clean():
    """Tears down AWS resources"""
    clean_cmd()

@cli.command()
def check():
    """Checks the status of the endpoints on AWS"""
    check_cmd()

cli.add_command(deploy)
cli.add_command(invoke)
cli.add_command(clean)
cli.add_command(check)

if __name__ == "__main__":
    cli()

