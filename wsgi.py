from predictor import app

# Expose the Flask application as 'app' for Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
