import importlib.resources as resources
import logging

logger = logging.getLogger(__name__)

# Dit module zit in whisperlivekit.web_trivias, dus __package__ klopt
PKG = __package__ or "whisperlivekit.web_trivias"


def get_web_interface_html():
    """Laadt de kale HTML (zonder inline CSS/JS)."""
    try:
        with resources.files(PKG).joinpath("index.html").open(
            "r", encoding="utf-8"
        ) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading web interface HTML: {e}")
        return "<html><body><h1>Error loading interface</h1></body></html>"


def get_inline_ui_html():
    """
    Retourneert de volledige webinterface met:
    - CSS inline
    - app.js inline
    - pcm_worklet.js & recorder_worker.js als blobs in JS
    """
    try:
        base = resources.files(PKG)

        # HTML, CSS, JS
        with base.joinpath("index.html").open("r", encoding="utf-8") as f:
            html_content = f.read()
        with base.joinpath("style.css").open("r", encoding="utf-8") as f:
            css_content = f.read()
        with base.joinpath("app.js").open("r", encoding="utf-8") as f:
            js_content = f.read()

        # Audio worklet & worker
        with base.joinpath("pcm_worklet.js").open("r", encoding="utf-8") as f:
            worklet_code = f.read()
        with base.joinpath("recorder_worker.js").open(
            "r", encoding="utf-8"
        ) as f:
            worker_code = f.read()

        # 1) CSS inline
        html_content = html_content.replace(
            '<link rel="stylesheet" href="style.css" />',
            f"<style>\n{css_content}\n</style>",
        )

        # 2) JS aanpassen zodat pcm_worklet.js en recorder_worker.js inline blobs worden
        js_content = js_content.replace(
            'await audioContext.audioWorklet.addModule("/web/pcm_worklet.js");',
            'const workletBlob = new Blob([`'
            + worklet_code
            + '`], { type: "application/javascript" });\n'
            'const workletUrl = URL.createObjectURL(workletBlob);\n'
            "await audioContext.audioWorklet.addModule(workletUrl);",
        )

        js_content = js_content.replace(
            'recorderWorker = new Worker("/web/recorder_worker.js");',
            'const workerBlob = new Blob([`'
            + worker_code
            + '`], { type: "application/javascript" });\n'
            'const workerUrl = URL.createObjectURL(workerBlob);\n'
            "recorderWorker = new Worker(workerUrl);",
        )

        # 3) JS inline (module)
        html_content = html_content.replace(
            '<script type="module" src="app.js"></script>',
            f"<script type=\"module\">\n{js_content}\n</script>",
        )

        return html_content

    except Exception as e:
        logger.error(f"Error creating embedded web interface: {e}")
        return "<html><body><h1>Error loading embedded interface</h1></body></html>"


if __name__ == "__main__":
    import pathlib

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from starlette.staticfiles import StaticFiles

    # Let op: nu verwijzen we naar het nieuwe package
    import whisperlivekit.web_trivias as webpkg

    app = FastAPI()

    web_dir = pathlib.Path(webpkg.__file__).parent
    # /web mount mag blijven; wordt niet meer gebruikt door app.js,
    # maar kan handig zijn voor toekomstige assets
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

    @app.get("/")
    async def get():
        return HTMLResponse(get_inline_ui_html())

    uvicorn.run(app=app)
