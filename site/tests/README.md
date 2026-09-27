# Site QA

The QA script discovers every `*.html` file below `site/` (excluding
`site/tests/`), starts a local server at `/statim/`, and checks the pages in
headless Chromium.

## Set up

Python 3.12 is used in CI. From the repository root, install the pinned browser
driver and Chromium:

```sh
python -m pip install playwright==1.63.0
python -m playwright install chromium
```

On Linux, `python -m playwright install --with-deps chromium` also installs the
required system packages (and may require elevated privileges).

## Run

Run all checks with the built-in server:

```sh
python site/tests/check.py
```

Check one site-relative page:

```sh
python site/tests/check.py --only index.html
python site/tests/check.py --only docs/index.html
```

To test an already running server, give its base URL. It must expose the site
files at that URL; for the local deployment shape, include `/statim/`:

```sh
python site/tests/check.py --url http://localhost:8000/statim/
python site/tests/check.py --url http://localhost:8000/statim/ --only index.html
```

The report lists failures by page and check, plus transferred bytes for every
page. Any failure exits with status 1. An empty or incomplete `site/` is
reported cleanly as a failed check rather than raising an exception.
