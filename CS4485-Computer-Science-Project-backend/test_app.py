from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'GDP Analysis Backend is running!'
    })

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({
        'status': 'success',
        'data': 'Backend is working!'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
