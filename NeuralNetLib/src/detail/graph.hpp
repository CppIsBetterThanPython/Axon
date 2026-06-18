#pragma once

#include <cassert>

using NodeID = size_t;

#define LEAF_OPS(X) \
	X(Input) \
	X(Parameter) \
	X(Constant)

#define UNARY_OPS(X) \
	X(Sigmoid) \
	X(Relu) \
	X(Tanh)

#define ELEMENT_WISE_OPS(X) \
	X(Add) \
	X(Sub) \
	X(Mul) \
	X(Div)

enum class Operation : uint8_t {
#define ENUM_ENTRY(name) name,
	LEAF_OPS(ENUM_ENTRY)
	UNARY_OPS(ENUM_ENTRY)
	ELEMENT_WISE_OPS(ENUM_ENTRY)
	Reduce,
	MatMul,
#undef ENUM_ENTRY
};

class Shape {
	std::vector<size_t> dimensionSizes;

public:
	Shape(std::initializer_list<size_t> list) : dimensionSizes(list) {}
	Shape(std::vector<size_t> vec) : dimensionSizes(vec) {}

	size_t getTotalSize() {
		size_t totalSize = 0;

		for (size_t dim : dimensionSizes) {
			totalSize += dim;
		}

		return totalSize;
	}

	size_t operator[](size_t dim) const {
		return dimensionSizes[dim];
	}

	size_t getRank() const {
		return dimensionSizes.size();
	}

	bool operator==(Shape other) const {
		return this->dimensionSizes == other.dimensionSizes;
	}
};

enum class DataType {
	float32
};

struct TensorInfo {
	DataType type;
	Shape shape;
};

class Node {
public:
	Operation op;
	TensorInfo tensorInfo;

	std::vector<NodeID> inputs;
	std::vector<NodeID> outputs;

	bool requires_grad;
};

class Graph;

class Tensor {
public:
	Graph* graph;
	NodeID nodeID;


	const Shape& getShape() {
		Node& node = graph->getNode(nodeID);

		return node.tensorInfo.shape;
	}

	size_t getRank() {
		getShape().getRank();
	}
};

class Graph {
private:
	std::vector<Node> nodes;
	std::vector<NodeID> inputs;
	std::vector<NodeID> outputs;
	std::vector<NodeID> params;
	std::vector<NodeID> constants;

	Node& getNode(NodeID id) {
		return nodes[id];
	}

	bool nodeRequiresGrad(Operation op, std::vector<NodeID> inputs) {
		switch (op) {
		case Operation::Parameter:
			return true;
		default:
			break;
		}

		for (NodeID id : inputs) {
			if (getNode(id).requires_grad)
				return true;
		}

		return false;
	}

	Node constructNode(Operation op, TensorInfo tensorInfo, std::vector<NodeID> inputs, std::vector<NodeID> outputs = {}) {
		bool requires_grad = nodeRequiresGrad(op, inputs);

		return Node(op, tensorInfo, inputs, outputs, requires_grad);
	}

	NodeID addNode(Operation op, TensorInfo tensorInfo, std::vector<NodeID> inputs) {
		nodes.push_back(
			constructNode(
				op,
				tensorInfo,
				inputs,
				{}
			)
		);

		NodeID nodeID = nodes.size() - 1;

		for (NodeID id : inputs) {
			getNode(id).outputs.push_back(nodeID);
		}

		return nodeID;
	}

	Tensor ElementWiseOp(Operation op, Tensor a, Tensor b) {
		assert(a.getShape() == b.getShape());

		NodeID nodeID = addNode(
			op,
			TensorInfo(DataType::float32, a.getShape()),
			{a.nodeID, b.nodeID}
		);

		return Tensor(this, nodeID);
	}

	Tensor LeafOp(Operation op, Shape shape) {
		NodeID nodeID = addNode(
			op,
			TensorInfo(DataType::float32, shape),
			{}
		);

		return Tensor(this, nodeID);
	}

	Tensor UnaryOp(Operation op, Tensor a) {
		NodeID nodeID = addNode(
			op,
			TensorInfo(DataType::float32, a.getShape()),
			{a.nodeID}
		);

		return Tensor(this, nodeID);
	}

#define LEAF_OP_FUNC(name) \
	Tensor name(Shape shape) { \
		return LeafOp(Operation::name, shape); \
	}

	LEAF_OPS(LEAF_OP_FUNC)

#undef LEAF_OP_FUNC
	
#define ELEMENT_WISE_OP_FUNC(name) \
	Tensor name(Tensor a, Tensor b) { \
		return ElementWiseOp(Operation::name, a, b); \
	}
	
	ELEMENT_WISE_OPS(ELEMENT_WISE_OP_FUNC)

#undef ELEMENT_WISE_OP_FUNC

#define UNARY_OP_FUNC(name) \
	Tensor name(Tensor a) { \
		return UnaryOp(Operation::name, a); \
	}

	UNARY_OPS(UNARY_OP_FUNC)

#undef UNARY_OP_FUNC

	Tensor MatMul(Tensor a, Tensor b) {
		assert(a.getRank() == 2);
		assert(b.getRank() == 2);
		assert(a.getShape()[1] == b.getShape()[0]);

		NodeID nodeID = addNode(
			Operation::MatMul,
			TensorInfo(DataType::float32, Shape({a.getShape()[0], b.getShape()[1]})),
			{a.nodeID, b.nodeID}
		);

		return Tensor(this, nodeID);
	}

	Tensor Reduce(Tensor a) {
		assert(a.getRank() == 1);

		NodeID nodeID = addNode(
			Operation::MatMul,
			TensorInfo(DataType::float32, Shape({1})),
			{a.nodeID}
		);

		return Tensor(this, nodeID);
	}

};
