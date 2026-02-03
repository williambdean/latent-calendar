# PyMC Slides

Slides for PyMC Labs talk in October 2023. Listen to the talk on YouTube: https://www.youtube.com/watch?v=k0CN-cuq724

## Usage

Build the docker image

```bash
docker build -t pymc-slides .
```

Create the slide material

```bash
docker run --rm -it -v $(pwd):/app -w /app pymc-slides python pymc-slides.py
```

Create the slide deck with [Quarto](https://quarto.org/)

```bash
quarto render pymc-slides.qmd
```

**Interactive use**

To play around with the environment used.

```bash
docker run --rm -it --entrypoint bash -v $(pwd):/app -w /app pymc-slides
```
