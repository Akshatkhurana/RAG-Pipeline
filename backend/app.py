# from flask import Flask, request, jsonify 
# from rag_pipeline import answer_question
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.get_json()
#     query = data.get("question", "")
#     if not query:
#         return jsonify({"error": "Empty query"}), 400

#     print(f"Received question: {query}") 
#     answer = answer_question(query)
#     print(f"Answer generated: {answer}") 
#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(debug=True)






from flask import Flask, request, jsonify 
from rag_pipeline import answer_question
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "")
    mode = data.get("mode", "factual")  # Get mode from frontend

    if not query:
        return jsonify({"error": "Empty query"}), 400

    print(f"Received question: {query}, Mode: {mode}") 

    answer = answer_question(query, mode=mode)

    return jsonify({"answer": answer})
