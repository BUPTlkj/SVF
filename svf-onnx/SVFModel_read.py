import numpy as np
import re
import onnx
from onnx import numpy_helper
import warnings
# import json
# from neo4j import GraphDatabase
from enum import Enum

# 目前支持ONNX基本分析
# 1. 支持分析结果转化为ONNXJSON格式
# 2. 支持分析结果转化为关系型到Neo4j图数据库
# 3. 支持与C++代码交互

# Step 1.1, 此部分是读取ONNX模型的结构

# 读取onnx网络结构 并判断是否是卷积神经网络
def read_onnx_net(net_file):
	# 加载模型
	onnx_model = onnx.load(net_file)
	# 检查模型加载是否完全准确
	onnx.checker.check_model(onnx_model)

	# 用于判断是否是卷积神经网络的字符
	is_conv = False
	# 检查神经网络结构的每一个节点
	for node in onnx_model.graph.node:
		# 是否是Conv 卷积神经网络
		# print("type", node.op_type)
		if node.op_type == 'Conv':
			is_conv = True
			break

	# 返回正确的神经网络， 并返回是否是卷积神经网络
	return onnx_model, is_conv

def gain_node(inferred_onnx_model):
	node = {}

	return node

def onnxshape_to_intlist(onnxshape):

	# lambda是一种一次性临时函数
	# 将onnx.shape.dim的值迭代的赋值给j，如果是空就是1，否则就是onnxshapee.dim
	# map(funcution, iterate)
	# result存放的是维度
	result = list(map(lambda j: 1 if j.dim_value is None else int(j.dim_value), onnxshape.dim))

	# No shape means a single value
	if not result:
		return [1]

	# NCHW 和 NHWC 是tensor的存储格式
	# convert NCHW to NHWC
	# 其中 N 表示batch size；C表示 feature maps 的数量，又称之为通道数；H 表示图片的高度，W表示图片的宽度
	if len(result) == 4:
		return [result[0], result[2], result[3], result[1]]

	# 返回NHWC维度
	return result

# 维度转换
def nchw_to_nhwc_shape(shape):

	# assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
	assert len(shape) == 4, "Unexpected shape size"
	# 将NCHW 转化为 NHWC
	return [shape[0], shape[2], shape[3], shape[1]]

# 索引转换
def nchw_to_nhwc_index(index: int) -> int:

	# 判断index是否out of bound
	# NCHW -> NHWC, 0==0 1->3 2->1 3->2
	assert 0 <= index <= 3, f"index out of range: {index}"
	if index == 0:  # batch (N)
		return 0
	elif index == 1:  # channel (C)
		return 3
	else:
		return index - 1

# nchw的矩阵转化为nhwc的矩阵
def nchw_to_nhwc(array):

	# 将NCHW transpose为 NHWC
	if array.ndim == 4:
		return array.transpose(0, 2, 3, 1)

	return array


# CHW与HWC的转化
# 输入两个矩阵的size
def reshape_nhwc(shape_in, shape_out):
	#print(shape_in, shape_out)
	# 计算输入的n-batch
	ndim_in = len(shape_in)
	# 计算输出的n-batch
	ndim_out = len(shape_out)
	# np.prod函数计算输入元素的乘积=CHW
	total_in = np.prod(shape_in[1:ndim_in])
	# np.prod函数计算输出元素的乘积=HWC
	total_out = np.prod(shape_out[1:ndim_out])
	# 保证输入输出的总数量一致的
	# 输出他们并不是数量一致的神经元计算而来
	assert total_in == total_out, "Reshape doesn't have same number of neurons before and after"
	# np.asarray类似于np.array 将其转化为数组
	# range()函数0 -> total_in-1
	# 将其转化为输入形式的数组 CHW
	array = np.asarray(range(total_in)).reshape(shape_in[1:ndim_in])
	# ndim()函数返回
	if array.ndim == 3:
		# WCH
		array = array.transpose((2, 0, 1))
	# 转为输出的维度形式的数组
	# HWC
	array = array.reshape(shape_out[1:ndim_out])
	if array.ndim == 3:
		# WCH
		return array.transpose((1, 2, 0))
	else:
		return array

# 该function的主要功能是提取神经元的分析要用到的内部信息
def prepare_model(model):

	shape_map = {} # 所有提取的shape{名称：shape}
	constants_map = {} # 常数
	output_node_map = {} # {输出节点名：节点}
	input_node_map = {} # {输入节点名：节点}

	# constants_map 获得每一个node的名字，以及他的常数值
	# shape_map 获得每一个node的名字，以及他的形状
	# initializer 存放模型的所有权重
	for initial in model.graph.initializer:
		# .copy函数保证了原数据的变化与const变化一致
		const = nchw_to_nhwc(numpy_helper.to_array(initial)).copy()
		# 获得神经网络的权重
		constants_map[initial.name] = const
		# 获得神经网络的权重的shape
		shape_map[initial.name] = const.shape

	# 所有输入节点的name
	placeholdernames = []
	# 获得所有节点的输入input
	#print("graph ", model.graph.node)
	for node_input in model.graph.input:
		# 获得每一个input中的name到placeholdernames列表中
		placeholdernames.append(node_input.name)
		if node_input.name not in shape_map:
			# 调用onnxshape_to_intlist函数提取node.input的输入shape
			shape_map[node_input.name] = onnxshape_to_intlist(node_input.type.tensor_type.shape)
			# 将输入节点加入input_node_map矩阵
			input_node_map[node_input.name] = node_input

	# 枚举nn结构中node节点的内容
	# node存放所有的计算节点
	for node in model.graph.node:
		#print(node.op_type)
		# output_node_map字典 保存节点的信息
		output_node_map[node.output[0]] = node
		# 检索当前节点中的input部分
		for node_input in node.input:
			# 将node_input节点信息保存到字典中
			input_node_map[node_input] = node

		# 以下是用来当前节点的类型attribute

		# 如果是flatten，拉成一维向量
		if node.op_type == "Flatten":
			#shape_map[node.output[0]] = shape_map[node.input[0]]
			# 其输出的矩阵的shape_map 如下，以输出节点为检索名，输出的一维矩阵的维度为[1, HWC的乘积]
			shape_map[node.output[0]] = [1,] + [np.prod(shape_map[node.input[0]][1:]),]
		# 如果是常数
		elif node.op_type == "Constant":
			# 获取节点的属性
			const = node.attribute
			const = nchw_to_nhwc(numpy_helper.to_array(const[0].t)).copy()
			constants_map[node.output[0]] = const
			shape_map[node.output[0]] = const.shape
		# 如果是矩阵乘法
		# transA 和 transB 分别代表不同的矩阵
		elif node.op_type in ["MatMul", "Gemm"]:
			transA = 0
			transB = 0
			for attribute in node.attribute:
				if 'transA' == attribute.name:
					transA = attribute.i
				elif 'transB' == attribute.name:
					transB = attribute.i
			input_shape_A = ([1] if len(shape_map[node.input[0]])==1 else []) + list(shape_map[node.input[0]])
			input_shape_B =  list(shape_map[node.input[1]]) + ([1] if len(shape_map[node.input[1]])==1 else [])
			M = input_shape_A[transA]
			N = input_shape_B[1 - transB]
			shape_map[node.output[0]] = [M, N]

		# 如果是加减乘除
		elif node.op_type in ["Add", "Sub", "Mul", "Div"]:
			# 上述操作不会改变大小
			shape_map[node.output[0]] = shape_map[node.input[0]]
			if node.input[0] in constants_map and node.input[1] in constants_map:
				if node.op_type == "Add":
					result = np.add(constants_map[node.input[0]], constants_map[node.input[1]])
				elif node.op_type == "Sub":
					result = np.subtract(constants_map[node.input[0]], constants_map[node.input[1]])
				elif node.op_type == "Mul":
					result = np.multiply(constants_map[node.input[0]], constants_map[node.input[1]])
				elif node.op_type == "Div":
					result = np.divide(constants_map[node.input[0]], constants_map[node.input[1]])
				constants_map[node.output[0]] = result
		elif node.op_type in ["Conv", "MaxPool", "AveragePool"]:
			output_shape = []
			input_shape = shape_map[node.input[0]]

			require_kernel_shape = node.op_type in ["MaxPool", "AveragePool"]
			if not require_kernel_shape:
				filter_shape = shape_map[node.input[1]]
				kernel_shape = filter_shape[1:-1]

			strides = [1, 1]
			padding = [0, 0, 0, 0]
			auto_pad = 'NOTSET'
			dilations = [1, 1]
			group = 1
			ceil_mode = 0
			for attribute in node.attribute:
				if attribute.name == 'strides':
					strides = attribute.ints
				elif attribute.name == 'pads':
					padding = attribute.ints
				elif attribute.name == 'auto_pad':
					auto_pad = attribute.s
				elif attribute.name == 'kernel_shape':
					kernel_shape = attribute.ints
				elif attribute.name == 'dilations':
					dilations = attribute.ints
				elif attribute.name == 'group':
					group = attribute.i
				elif attribute.name == 'ceil_mode':
					ceil_mode = attribute.i

			effective_kernel_shape = [(kernel_shape[i] - 1) * dilations[i] + 1 for i in range(len(kernel_shape))]

			output_shape.append(input_shape[0])

			for i in range(len(kernel_shape)):
				effective_input_size = input_shape[1 + i]
				effective_input_size += padding[i]
				effective_input_size += padding[i + len(kernel_shape)]
				if ceil_mode == 1:
					strided_kernel_positions = int(np.ceil((effective_input_size - effective_kernel_shape[i]) / float(strides[i])))
				else:
					strided_kernel_positions = int(np.floor((effective_input_size - effective_kernel_shape[i]) / strides[i]))
				output_shape.append(1 + strided_kernel_positions)

			if require_kernel_shape:
				output_shape.append(input_shape[3])
			else:
				output_shape.append(filter_shape[0])

			shape_map[node.output[0]] = output_shape
		elif node.op_type in ["Relu", "Sigmoid", "Tanh", "Softmax", "BatchNormalization", "LeakyRelu"]:
			shape_map[node.output[0]] = shape_map[node.input[0]]

		# Gather is for the moment solely for shapes
		elif node.op_type == "Gather":
			axis = 0
			for attribute in node.attribute:
				axis = attribute.i
			if node.input[0] in constants_map and node.input[1] in constants_map:
				data = constants_map[node.input[0]]
				indexes = constants_map[node.input[1]]
				constants_map[node.output[0]] = np.take(data, indexes, axis)

			if node.input[0] in shape_map and node.input[1] in shape_map:
				r = len(shape_map[node.input[0]])
				q = len(shape_map[node.input[1]])
				out_rank = q + r - 1
				if out_rank == 0:
					shape_map[node.output[0]] = shape_map[node.input[1]]
				else:
					output_shape = []
					for i in range(out_rank):
						if i < axis:
							output_shape.append(shape_map[node.input[0]][i]) # i < axis < r
						elif i >= axis and i < axis + q:
							output_shape.append(shape_map[node.input[0]][i-axis]) # i - axis < q
						else:
							output_shape.append(shape_map[node.input[0]][i - q + 1]) # i < out_rank < q + r - 1
					shape_map[node.output[0]] = output_shape
		elif node.op_type == "Shape":
			if node.input[0] in shape_map:
				constants_map[node.output[0]] = shape_map[node.input[0]]
				shape_map[node.output[0]] = [len(shape_map[node.input[0]])]

		elif node.op_type == "Reshape":
			#print("RESHAPE ", node.input, node.output)
			if node.input[1] in constants_map:
				total = 1
				replace_index = -1
				for index in range(len(constants_map[node.input[1]])):
					if constants_map[node.input[1]][index] == -1:
						replace_index = index
					else:
						total *= constants_map[node.input[1]][index]

				if replace_index != -1:
					constants_map[node.input[1]][replace_index] = np.prod(shape_map[node.input[0]]) / total

				if len(constants_map[node.input[1]]) == 4:
					shape_map[node.output[0]] = [constants_map[node.input[1]][0], constants_map[node.input[1]][2], constants_map[node.input[1]][3], constants_map[node.input[1]][1]]
				else:
					shape_map[node.output[0]] = constants_map[node.input[1]]

		elif node.op_type == "Unsqueeze":
			if node.input[0] in shape_map:
				axis = node.attribute[0].ints
				output_shape = list(shape_map[node.input[0]])
				if node.input[0] in constants_map:
					constants_map[node.output[0]] = constants_map[node.input[0]]
				for i in axis:
					output_shape.insert(i, 1)
					if node.input[0] in constants_map:
						constants_map[node.output[0]] = np.expand_dims(constants_map[node.output[0]], axis=i)
				shape_map[node.output[0]] = output_shape

		elif node.op_type == "Concat":
			all_constant = True
			n_dim = len(shape_map[node.input[0]])
			if n_dim > 2:
				axis = nchw_to_nhwc_index(node.attribute[0].i)
			else:
				axis = node.attribute[0].i
			for node_input in node.input:
				if not node_input in constants_map:
					all_constant = False
					break
			if all_constant:
				constants_map[node.output[0]] = np.concatenate([constants_map[input] for input in node.input], axis=axis)
			all_shape_known = True
			for node_input in node.input:
				if not node_input in shape_map:
					all_shape_known = False
					break
			assert all_shape_known, "Unknown shape for at least one node input!"
			new_axis_size = 0
			for node_input in node.input:
				new_axis_size += shape_map[node_input][axis]
			shape_map[node.output[0]] = [shape_map[node.input[0]][i] if i != axis else new_axis_size for i in range(len(shape_map[node.input[0]]))]
			if not all_constant:
				assert axis == n_dim-1, " supports concatenation on the channel dimension"

		elif node.op_type == "Tile":
			repeats = nchw_to_nhwc_shape(constants_map[node.input[1]])
			input_shape = list(shape_map[node.input[0]])
			assert len(repeats) == len(input_shape), "Expecting one repeat factor per dimension"
			output_shape = [factor * size for factor, size in zip(repeats, input_shape)]
			shape_map[node.output[0]] = output_shape

			repeat_index = np.where(np.array(repeats) != 1)[0]
			assert len(repeat_index) == 1, "only supports repeats for one dimension"
			repeat_index = repeat_index.item()
			assert repeat_index == 1, " only supports repeats for the first dimension"
			assert input_shape[0] == 1, "currently only supports repeats for dimensions of size 1"

		elif node.op_type == "Expand":
			if node.input[1] in constants_map:
				if len(constants_map[node.input[1]]) == 4:
					shape_map[node.output[0]] = [constants_map[node.input[1]][0], constants_map[node.input[1]][2], constants_map[node.input[1]][3], constants_map[node.input[1]][1]]
				else:
					shape_map[node.output[0]] = constants_map[node.input[1]]

				result = np.zeros(shape_map[node.output[0]]) + constants_map[node.input[0]]
				constants_map[node.output[0]] = result
		elif node.op_type == "Pad":
			input_shape = np.array(shape_map[node.input[0]])
			for attribute in node.attribute:
				if attribute.name == "pads":
					padding = np.array(attribute.ints)
				if attribute.name == "mode":
					assert attribute.s == bytes(b'constant'), "only zero padding supported"
				if attribute.name == "value":
					assert attribute.f == 0, "only zero padding supported"
			output_shape = np.copy(input_shape)
			input_dim = len(input_shape)
			assert len(padding) == 2* input_dim
			for i in range(2,input_dim): # only pad spatial dimensions
				output_shape[i-1] += padding[i]+padding[i+input_dim]
			shape_map[node.output[0]] = list(output_shape)
		else:
			assert 0, f"Operations of type {node.op_type} are not yet supported."

	return shape_map, constants_map, output_node_map, input_node_map, placeholdernames


def get_resource(shape_map, constants_map, node_input_name_list):
	list_info = []
	aa = {}
	for item in node_input_name_list:
		if item in shape_map:
			if item in constants_map:
				list_info = []
				list_info.append(shape_map[item])
				list_info.append(constants_map[item].tolist())
				aa[item] = list_info
	return aa

def output_node_map_2_graph(shape_map, constants_map, output_node_map):
	output_graph = {}
	for current in output_node_map:
		node_info = []
		current_node_name = ""
		current_node_input_name = []
		current_node_input_dim = []
		current_node_output_name = []
		current_node_output_dim = ""
		current_node_op = ""
		current_node_name = current
		newstr = str(output_node_map[current_node_name]).split('\n')
		for ii in newstr:
			ii_ = ii.split(':')
			if 'input' in ii_:
				current_node_input_name.append(eval(ii_[1]))
			if 'output' in ii_:
				current_node_output_name.append(eval(ii_[1]))
			if 'name' in ii_:
				current_node_name = ii_[1]
			if 'op_type' in ii_:
				current_node_op = ii_[1].replace("\"", "").strip()
		current_node_name = current_node_name.split('_')[1].replace("\"","") + "_" + current_node_name.split('_')[0].replace("\"","").strip()
		node_info.append(current_node_name)
		node_info.append(current_node_op)
		node_info.append(current_node_input_name)
		dic = get_resource(shape_map, constants_map, current_node_input_name)
		node_info.append(dic)
		node_info.append(current_node_output_name)
		dic1 = get_resource(shape_map, constants_map, current_node_output_name)
		node_info.append(dic1)
		current_node_name = current_node_name.split('_')[1].replace("\"","") + "_" + current_node_name.split('_')[0].replace("\"","").strip()
		output_graph[current_node_name] = node_info

	return output_graph



def input_node_map_2_graph(input_node_map):
	input_graph = {}
	for current in input_node_map:
		node_info = []
		current_node_name = ""
		current_node_input_name = []
		current_node_input_dim = []
		current_node_output_name = []
		current_node_output_dim = ""
		current_node_op = ""
		# 当前节点名
		current_node_name = current
		newstr = str(input_node_map[current_node_name]).split('\n')
		#print("1111", newstr)
		for ii in newstr:
			ii_ = ii.split(':')
			#print("2222", ii_)
			if 'input' in ii_:
				current_node_input_name.append(ii_[1])
			if 'output' in ii_:
				current_node_output_name.append(ii_[1])
			if 'name' in ii_:
				current_node_name = ii_[1]
			if 'op_type' in ii_:
				current_node_op = ii_[1]
		node_info.append(current_node_name)
		node_info.append(current_node_op)
		node_info.append(current_node_input_name)
		node_info.append(current_node_output_name)
		#print(node_info)
		input_graph[current_node_name] = node_info

	# print(graph)
	return input_graph

def to_json(path):
	model_path = path
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	gain = output_node_map_2_graph(shape_map, constants_map, output_node_map)
	aa = path.replace('.onnx','.json')
	print("pppppppppp", aa)
	json_str = json.dumps(gain)
	# 将JSON字符串写入文件中
	with open("data22l.json", "w", encoding="utf-8") as f:
		f.write(json_str)


def to_neo4j(onnx_path, neo4j_uri, neo4j_user, neo4j_password):
	# 创建连接
	driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
	model_path = onnx_path
	pre_string = "_" + onnx_path.replace(".onnx", "")
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	gain = output_node_map_2_graph(shape_map, constants_map, output_node_map)
	name_list = []
	for na in gain.keys():
		name_list.append(na)

	#print(name_list)
	def create_directed_graph(tx, name_list):
		# 创建节点和关系
		for i in range(len(name_list) - 1):
			tx.run(
				"MERGE (a:Node {name: $source}) "
				"MERGE (b:Node {name: $target}) "
				"MERGE (a)-[:CONNECTED_TO]->(b)",
				source=name_list[i],
				target=name_list[i + 1]
			)

	# 使用Neo4j事务创建节点关系
	with driver.session() as session:
		session.execute_write(create_directed_graph, name_list)

	def add_type(tx, node_name, node_type):
		tx.run("MATCH (node:Node {name: $node_name}) "
			   "SET node.type = $node_type",
			   node_name=node_name,
			   node_type=node_type)

	def add_property(tx, node_name, property1_value, property2_value):
		# 为节点添加属性
		tx.run("MATCH (node:Node {name: $node_name}) "
			   "SET node.dims = $property1_value, node.dim_val = $property2_value",
			   node_name=node_name,
			   property1_value=property1_value,
			   property2_value=property2_value)

	for key in gain:
		node_name = key
		node_type = gain[key][1]
		with driver.session() as session:
			session.execute_write(add_type, key, node_type)
		node_input_list = gain[key][2]
		for i in gain[key][2]:
			for inj in gain[key][3].keys():
				if i in gain[key][3].keys():
					with driver.session() as session:
						session.execute_write(add_property, key, str(gain[key][3][i][0]), str(gain[key][3][i][1]))

		node_output_list = gain[key][4]
		for i in gain[key][4]:
			for inj in gain[key][5].keys():
				if i in gain[key][5].keys():
					absd = i

def InteractwithCpp(model_path):
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	output_graph = {}
	node_info = []
	node_info.append(get_strides(model_path))
	output_graph["SepcialInfo"] = node_info
	for current in output_node_map:
		node_info = []
		current_node_name = ""
		current_node_input_name = []
		current_node_input_dim = []
		current_node_output_name = []
		current_node_output_dim = ""
		current_node_op = ""
		current_node_name = current
		newstr = str(output_node_map[current_node_name]).split('\n')
		for ii in newstr:
			ii_ = ii.split(':')
			if 'input' in ii_:
				current_node_input_name.append(eval(ii_[1]))
			if 'output' in ii_:
				current_node_output_name.append(eval(ii_[1]))
			if 'name' in ii_:
				current_node_name = ii_[1]
			if 'op_type' in ii_:
				current_node_op = ii_[1].replace("\"", "").strip()
		current_node_name = current_node_name.split('_')[1].replace("\"","") + "_" + current_node_name.split('_')[0].replace("\"","").strip()
		parts = current_node_name.split("_")
		if len(parts) == 2 and parts[0].isdigit():
			number = int(parts[0])
			if 0 <= number <= 9:
				pre_st = f"{number:02}_{parts[1]}"
				current_node_name = pre_st
		node_info.append(current_node_op)
		node_info.append(current_node_input_name)
		dic = get_resource(shape_map, constants_map, current_node_input_name)
		node_info.append(dic)
		node_info.append(current_node_output_name)
		output_graph[current_node_name] = node_info

	return output_graph

def conv_filter(model_path):
	info = []
	res = {}
	info = []
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	# 拓扑顺序排列的节点的输入输出节点的依存关系的node列表
	nodes = model.graph.node
	for node_idx, node in enumerate(nodes):
		if node.op_type == "Conv":
			# 获得filter
			filters = constants_map[node.input[1]].transpose(1, 2, 3, 0)
			if node_idx < 10:
				conv_name = "0" + str(node_idx) + "_" + node.op_type
				info.append(filters.shape)
				info.append(filters.tolist())
				res[conv_name] = info

	return res

def get_strides(model_path):
	res = {}
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	# 拓扑顺序排列的节点的输入输出节点的依存关系的node列表
	nodes = model.graph.node
	for node_idx, node in enumerate(nodes):
		info = []
		pads = []
		if node.op_type == "Conv":
			strides = [1, 1]
			for attribute in node.attribute:
				if attribute.name == 'strides':
					strides = attribute.ints
					info.append("strides:")
					info.append(strides)
				elif attribute.name == 'pads':
					pads = attribute.ints
					info.append("pads:")
					info.append(pads)

			if node_idx < 10:
				conv_name = "0" + str(node_idx) + "_" + node.op_type
			else:
				conv_name = str(node_idx) + "_" + node.op_type

			res[conv_name] = info

		if node.op_type == "MaxPool":
			strides = [1, 1]
			for attribute in node.attribute:
				if attribute.name == 'strides':
					strides = attribute.ints
					info.append("strides:")
					info.append(strides)
				elif attribute.name == 'kernel_shape':
					kernel_shape = attribute.ints
					info.append("Windows:")
					info.append(kernel_shape)
				if attribute.name == 'pads':
					pads = attribute.ints
					info.append("pads:")
					info.append(pads)

			if node_idx < 10:
				maxpool_name = "0" + str(node_idx) + "_" + node.op_type
			else:
				maxpool_name = str(node_idx) + "_" + node.op_type
			res[maxpool_name] = info

	return res




