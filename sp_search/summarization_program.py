from graphviz import Digraph


class Node:
    def __init__(self, operation, sentence, index, score, height):
        self.operation = operation
        self.sentence = sentence
        self.index = index
        self.score = score
        self.height = height
        self.left = None
        self.right = None


def draw_tree_level_order(node, networkx_graph, curr_index):
    if node is None:
        return
    if node.left is None and node.right is None:
        networkx_queue = [node.sentence.replace(":", '') + " 0"]
        networkx_graph.node(node.sentence.replace(':', '') + " 0")
    else:
        networkx_queue = [node.sentence.replace(":", '') + " " + str(curr_index)]
    queue = [node]

    while len(queue) > 0:
        curr_sentence = networkx_queue.pop(0)
        curr_node = queue.pop(0)
        if curr_node.left is not None:
            if curr_node.left.left is None and curr_node.left.right is None:
                new_sentence = curr_node.left.sentence.replace(':', '') + " 0"
            else:
                curr_index += 1
                new_sentence = curr_node.left.sentence.replace(':', '') + " " + str(curr_index)
            networkx_graph.edge(new_sentence, curr_sentence, label=curr_node.operation)
            networkx_queue.append(new_sentence)
            queue.append(curr_node.left)
        if curr_node.right is not None:
            if curr_node.right.left is None and curr_node.right.right is None:
                new_sentence = curr_node.right.sentence.replace(':', '') + " 0"
            else:
                curr_index += 1
                new_sentence = curr_node.right.sentence.replace(':', '') + " " + str(curr_index)
            networkx_graph.edge(new_sentence, curr_sentence, label=curr_node.operation)
            networkx_queue.append(new_sentence)
            queue.append(curr_node.right)

    return curr_index


def draw_sp(sp):
    networkx_graph = Digraph()
    curr_index = 1
    for tree in sp:
        temp_curr_index = draw_tree_level_order(tree.root, networkx_graph, curr_index)
        curr_index = temp_curr_index + 1

    return networkx_graph


class Tree:
    def __init__(self, root):
        self.root = root

    def get_all_nodes(self, node):
        if node is None:
            return []
        left_nodes = self.get_all_nodes(node.left)
        right_nodes = self.get_all_nodes(node.right)

        return left_nodes + [node.sentence] + right_nodes

    def get_all_edges(self, node):
        if node is None:
            return []
        left_edges = self.get_all_edges(node.left)
        right_edges = self.get_all_edges(node.right)

        all_edges = []
        if node.left is not None:
            all_edges.append((node.left.sentence, node.sentence, node.operation))
        if node.right is not None:
            all_edges.append((node.right.sentence, node.sentence, node.operation))

        all_edges = left_edges + all_edges + right_edges

        return all_edges

    def construct_execution_string(self, node, with_intermediates=True):
        if node is None:
            return "", ""
        left_string, left_intermediate_string = self.construct_execution_string(node.left, with_intermediates)
        right_string, right_intermediate_string = self.construct_execution_string(node.right, with_intermediates)

        if node.operation == "fusion":
            if with_intermediates:
                temp_intermediate_string = left_intermediate_string + right_intermediate_string + " | " + "<I" + str(
                    node.index) + "> --> " + node.sentence.strip()
                return node.operation + " ( " + left_string + " " + right_string + " ) --> " + "<I" + str(
                    node.index) + ">", temp_intermediate_string
            else:
                return node.operation + " ( " + left_string + " " + right_string + " )", ""
        elif node.operation == "compression" or node.operation == "paraphrase":
            if with_intermediates:
                temp_intermediate_string = left_intermediate_string + " | " + "<I" + str(
                    node.index) + "> --> " + node.sentence.strip()
                return node.operation + " ( " + left_string + " ) --> " + "<I" + str(
                    node.index) + ">", temp_intermediate_string
            else:
                return node.operation + " ( " + left_string + " )", ""
        else:
            return "<S" + str(node.index) + ">", ""
