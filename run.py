# run.py
from app import create_app

app = create_app()

@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)
