from flask import Flask, render_template, request, jsonify
import util

app = Flask(__name__)


@app.route("/")
def start():
    return render_template("base.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    processed_input = util.preprocess_input(data)
    predictions = util.generate_predictions(processed_input)
    return jsonify(predictions)


if __name__ == "__main__":
    app.run(debug=False)  # change to False in production
