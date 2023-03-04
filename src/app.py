from flask import Flask, request, jsonify
from mynn import MyFirstNN

NN = MyFirstNN() 

app = Flask(__name__)


@app.route('/api/myfirstnn',methods=['GET','POST'])
def mynn():
    if request.method == 'GET':
        return jsonify("This is my NN. Developed by Kavya.")
    else:
        input_json = request.json
        nn_inputs = input_json['x']
        output = NN.predict(nn_inputs)
        return jsonify(output)
    
if __name__=='__main__':
    app.run('0.0.0.0',port=8511)