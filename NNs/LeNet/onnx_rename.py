import onnx

# 加载ONNX模型
path = "/Users/z5524562/model_cnn_2.onnx"
model = onnx.load(path)

# 假设我们更改第一个输入的名称
original_input_name = model.graph.input[0].name
new_input_name = 'constant'

# 更新模型的第一个输入名称
model.graph.input[0].name = new_input_name

# 更新所有使用原始输入名称的节点，使它们指向新的输入名称
for node in model.graph.node:
    for i, input_name in enumerate(node.input):
        if input_name == original_input_name:
            node.input[i] = new_input_name

# 保存修改后的模型
onnx.save(model, 'modified_model.onnx')
