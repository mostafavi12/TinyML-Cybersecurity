from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

def read_attacks():
    df = pd.read_csv("../logs/attack_logs.txt", names=["Attack"])
    return df.tail(10).to_dict(orient="records")

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/data')
def data():
    return jsonify(read_attacks())

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")