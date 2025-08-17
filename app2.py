from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/get_value')
def get_value():
    value = "Hello from Flask!"
    return jsonify({'value': value})

if __name__ == '__main__':
    app.run(debug=True)