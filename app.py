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
    # 获取数据，假设数据是以 JSON 格式发送
    data = request.json
    
    # 从数据中提取对应的变量值
    input_data = [
        data['ap'],
        data['crp'],
        data['ca'],
        data['ggt'],
        data['tp'],
        data['ab'],
        data['gc'],
        data['nab']
    ]
    
    # 假设 input_data 是你从前端获取的输入数据
    input_data = [float(value) for value in input_data]  # 转换为浮点数类型
    input_data = np.array(input_data).reshape(1, -1)     # 转换为 numpy 数组

    prediction = model.predict(input_data)
    
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