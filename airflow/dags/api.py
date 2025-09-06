from pathlib import Path as _Path
from flask import Flask, request, jsonify, send_from_directory
from predictor import run_prediction

def create_app():
    app = Flask(__name__)
    
    def _project_root() -> _Path:
       return _Path(__file__).resolve().parents[2] # Goes up 3 levels

    @app.get("/health")
    def health():
        return "OK", 200

    @app.post("/predict")
    def predict_credit_risk():
        try:
            print("=== 📥 Incoming request ===")
            result = run_prediction(request.json or {}, make_pdf=True)
            print("✅ Returning response to client")
            return jsonify(result), 200
        except Exception as e:
            print("❌ ERROR OCCURRED:", str(e))
            return jsonify({"error": str(e)}), 500

    @app.get("/pdfs/<filename>")
    def download_pdf(filename):
        pdf_dir = _project_root() / "generated_pdfs"
        # print(f"pdf_dir: {pdf_dir}")
        return send_from_directory(pdf_dir, filename)

    return app

# # Check the path
# with create_app().test_client() as client:
#     response = client.get("/pdfs/sample.pdf")
#     print("Status Code:", response.status_code)

if __name__ == "__main__":
    create_app().run(port=5000)
