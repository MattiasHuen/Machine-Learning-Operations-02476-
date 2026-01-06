import typer

app = typer.Typer()

@app.command()
def hello(name: str = "Mattias", count: int = 1) -> None:
    for _ in range(count):
        typer.echo(f"Hello {name}")


if __name__ == "__main__":
    app()