import numpy as np
import re
import onnx
from onnx import numpy_helper
import warnings
# import json
# from neo4j import GraphDatabase
from enum import Enum

# Currently supporting ONNX basic analysis
# 1 Support converting analysis results to ONNXJSON format
# 2 Support converting analysis results into relational data into Neo4j graph database
# 3 Support interaction with C++code
# Step 1.1, this section reads the structure of the ONNX model
# Read the ONNX network structure and determine if it is a convolutional neural network

def read_onnx_net(net_file):
	# Load model
	onnx_model = onnx.load(net_file)
	# Check model loading
	onnx.checker.check_model(onnx_model)

	# Used to determine whether they are convolutional neural networks
	is_conv = False
	# Check each node of the neural network structure
	for node in onnx_model.graph.node:
		# Whether is Conv
		if node.op_type == 'Conv':
			is_conv = True
			break

	# Returns the correct neural network and whether it is a convolutional neural network
	return onnx_model, is_conv

def gain_node(inferred_onnx_model):
	node = {}

	return node

def onnxshape_to_intlist(onnxshape):

	# Lambda is a one-time temporary function
	# Iteratively assign the value of onnx.shape.dim to j, if it is empty, it is 1, otherwise it is onnxshape.dim
	# map(funcution, iterate)
	# result stores dimensions
	result = list(map(lambda j: 1 if j.dim_value is None else int(j.dim_value), onnxshape.dim))

	# No shape means a single value
	if not result:
		return [1]

	# NCHW & NHWC
	# convert NCHW to NHWC
	# N: batch size; C, the number of feature maps, Channels. H:height，W: weight.
	if len(result) == 4:
		return [result[0], result[2], result[3], result[1]]

	# return NHWC dims
	return result

# Dims convert
def nchw_to_nhwc_shape(shape):

	# Assert is used to determine an expression and trigger an exception when the expression condition is false
	assert len(shape) == 4, "Unexpected shape size"
	# convert NCHW into NHWC
	return [shape[0], shape[2], shape[3], shape[1]]

# index convert
def nchw_to_nhwc_index(index: int) -> int:

	# Determine whether index is out of bound
	# NCHW -> NHWC, 0==0 1->3 2->1 3->2
	assert 0 <= index <= 3, f"index out of range: {index}"
	if index == 0:  # batch (N)
		return 0
	elif index == 1:  # channel (C)
		return 3
	else:
		return index - 1

# Convert nchw into nhwc matrix
def nchw_to_nhwc(array):

	# NCHW transpose NHWC
	if array.ndim == 4:
		return array.transpose(0, 2, 3, 1)

	return array

# Conversion of CHW and HWC
# Enter the dims of two matrices
def reshape_nhwc(shape_in, shape_out):
	#print(shape_in, shape_out)
	# Calculate the input  n-batch
	ndim_in = len(shape_in)
	# Calculate the output n-batch
	ndim_out = len(shape_out)
	# np.prod fun is used to calculate the product of input elements =CHW
	total_in = np.prod(shape_in[1:ndim_in])
	# np.prodfun is used to calculate the product of output elements=HWC
	total_out = np.prod(shape_out[1:ndim_out])
	# Ensure consistent total quantity of inputs and outputs
	# The output is not calculated from the same number of neurons
	assert total_in == total_out, "Reshape doesn't have same number of neurons before and after"
	# np.asarray is similar to np.array. Convert it to an array
	# range() 0 -> total_in-1
	# Convert it into an array in input form CHW
	array = np.asarray(range(total_in)).reshape(shape_in[1:ndim_in])
	# ndim() return
	if array.ndim == 3:
		# WCH
		array = array.transpose((2, 0, 1))
	# Convert an array in the form of output dimensions
	# HWC
	array = array.reshape(shape_out[1:ndim_out])
	if array.ndim == 3:
		# WCH
		return array.transpose((1, 2, 0))
	else:
		return array

# This function is to extract internal information needed for analyzing neurons
def prepare_model(model):

	shape_map = {} # all the shape{name: shape}
	constants_map = {} # constant
	output_node_map = {} # {input node name: node}
	input_node_map = {} # {input node name: node}

	# constants_map gain each node name and constant
	# shape_map gain each node name and shape
	# initializer store the model's weight
	for initial in model.graph.initializer:
		# .copyThe function ensures that the changes in the original data are consistent with the const changes
		const = nchw_to_nhwc(numpy_helper.to_array(initial)).copy()
		# gain weight
		constants_map[initial.name] = const
		# gain the weight's shape
		shape_map[initial.name] = const.shape

	# all input nodes' name
	placeholdernames = []
	# all nodes' input
	for node_input in model.graph.input:
		# Obtain the name from each input and add it to the placeholder's names list
		placeholdernames.append(node_input.name)
		if node_input.name not in shape_map:
			# Call onnxshape_to_intlist fun to extract node.input's shape
			shape_map[node_input.name] = onnxshape_to_intlist(node_input.type.tensor_type.shape)
			# Add input nodes to the input_node_map matrix
			input_node_map[node_input.name] = node_input

	# Enumerating the content of node nodes in the nn structure
	# Node stores all computing nodes
	for node in model.graph.node:
		# output_node_map dict, store node info
		output_node_map[node.output[0]] = node
		# Retrieve the input section in the current node
		for node_input in node.input:
			input_node_map[node_input] = node

		# The following is the type attribute used for the current node

		# flatten，pull it into a one-dimensional vector
		if node.op_type == "Flatten":
			# The shape_map of the output matrix is as follows, with the output node as the search name,
			# and the dimension of the output one-dimensional matrix is [1, the product of HWC]
			shape_map[node.output[0]] = [1,] + [np.prod(shape_map[node.input[0]][1:]),]
		# If it is constant
		elif node.op_type == "Constant":
			# Get node properties
			const = node.attribute
			const = nchw_to_nhwc(numpy_helper.to_array(const[0].t)).copy()
			constants_map[node.output[0]] = const
			shape_map[node.output[0]] = const.shape
		# If it is matrix multiplication
		# transA & transB represents different matrix
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

		# If add, sub, mul, div
		elif node.op_type in ["Add", "Sub", "Mul", "Div"]:
			# The above operation will not change dims
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
		# Current Node_name
		current_node_name = current
		newstr = str(input_node_map[current_node_name]).split('\n')
		for ii in newstr:
			ii_ = ii.split(':')
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
		input_graph[current_node_name] = node_info

	return input_graph

def to_json(path):
	model_path = path
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	gain = output_node_map_2_graph(shape_map, constants_map, output_node_map)
	aa = path.replace('.onnx','.json')
	json_str = json.dumps(gain)
	# Load JSON into files
	with open("data22l.json", "w", encoding="utf-8") as f:
		f.write(json_str)


def to_neo4j(onnx_path, neo4j_uri, neo4j_user, neo4j_password):
	# Create connection
	driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
	model_path = onnx_path
	pre_string = "_" + onnx_path.replace(".onnx", "")
	model, is_conv = read_onnx_net(model_path)
	shape_map, constants_map, output_node_map, input_node_map, placeholdernames = prepare_model(model)
	gain = output_node_map_2_graph(shape_map, constants_map, output_node_map)
	name_list = []
	for na in gain.keys():
		name_list.append(na)

	def create_directed_graph(tx, name_list):
		# Create nodes and relationships
		for i in range(len(name_list) - 1):
			tx.run(
				"MERGE (a:Node {name: $source}) "
				"MERGE (b:Node {name: $target}) "
				"MERGE (a)-[:CONNECTED_TO]->(b)",
				source=name_list[i],
				target=name_list[i + 1]
			)

	# Creating node relationships using Neo4j transactions
	with driver.session() as session:
		session.execute_write(create_directed_graph, name_list)

	def add_type(tx, node_name, node_type):
		tx.run("MATCH (node:Node {name: $node_name}) "
			   "SET node.type = $node_type",
			   node_name=node_name,
			   node_type=node_type)

	def add_property(tx, node_name, property1_value, property2_value):
		# Add attributes to nodes
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
	# Node list of dependency relationships between input and output nodes arranged in topological order
	nodes = model.graph.node
	for node_idx, node in enumerate(nodes):
		if node.op_type == "Conv":
			# Gain filter
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
	# Node list of dependency relationships between input and output nodes arranged in topological order
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




