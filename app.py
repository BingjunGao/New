from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# 获取当前文件所在的目录路径
current_dir = os.path.dirname(__file__)

# 加载模型，使用相对路径
model_path = os.path.join(current_dir, 'model.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 从表单获取输入数据
    ap = request.form['ap']
    crp = request.form['crp']
    ca = request.form['ca']
    ggt = request.form['ggt']
    tp = request.form['tp']
    ab = request.form['ab']
    gc = request.form['gc']
    nab = request.form['nab']

    # 模拟模型预测逻辑
    # 实际上这里应为你模型的预测代码
    # 假设模型返回结果为0, 1 或 2
    prediction = model.predict([[ap, crp, ca, ggt, tp, ab, gc, nab]])  # 使用你的模型进行预测
    
    # 将模型预测的数值转换为对应的诊断结果
    if prediction == 0:
        result = "No Arthritis"
    elif prediction == 1:
        result = "Rheumatoid arthritis"
    elif prediction == 2:
        result = "Osteoarthritis"
    
    # 将结果作为JSON返回给前端
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)