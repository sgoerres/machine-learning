#source: https://stackoverflow.com/a/39772170
# modified to return a string instead of printing
from sklearn.tree import _tree
from sklearn import tree
import pydot
import matplotlib.pyplot as plt
import numpy as np

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    outputstring = ("def tree({}):\n".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        recursestring = ""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recursestring = recursestring + ("{}if {} <= {}:\n".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            recursestring = recursestring+("{}else:  # if {} > {}\n".format(indent, name, threshold))
            recursestring = recursestring+recurse(tree_.children_right[node], depth + 1)
        else:
            recursestring = recursestring + ("{}return {}\n".format(indent, tree_.value[node]))
        return recursestring

    outputstring = outputstring + recurse(0, 1)

    return outputstring

def output_all_regressors(adaboostregressor, feature_names,  output_dir):
    counter = 0
    for e in adaboostregressor.estimators_:
        outfile = output_dir + "f{counter}_out.dot"
        tree.export_graphviz(e, out_file=outfile,
                             feature_names=feature_names)  # , feature_names=cleared_train.columns.values
        outfile_png = output_dir + f"{counter}_out.png"
        (graph,) = pydot.graph_from_dot_file(outfile)
        graph.write_png(outfile_png)
        outText = output_dir + f"{counter}_graph.txt"
        with open(outText, "w") as text_file:
            text_file.write(O.tree_to_code(e, feature_names))  #
        counter = counter + 1

def output_plot(y_test, y_pred, output_dir):
    x_idx = np.arange(y_test.shape[0])
    plt.figure()
    # plt.scatter(X_train, y_train, c="k", label="training samples")
    plt.scatter(x_idx, y_test, c="b", label="test samples")
    plt.plot(x_idx, y_pred, c="g", label="adaboost predictions")
    plt.legend()
    plt.show()
    plt.savefig(output_dir + "overall_curve.png")
    print(f"Runtime total [s]: {S._rt()}")