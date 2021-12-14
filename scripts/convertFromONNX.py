# ConvertFromONNX.py
# Description: 

import util

def ONNX_Keras(onnxIn, modelOut):
    from onnx2keras import onnx_to_keras
    
    print("Converting ONNX file to Keras...")
    
    # Get input name from loaded model
    input_all = onnxIn.graph.input
    output = onnxIn.graph.output
    input_init = onnxIn.graph.initializer
    #net_feed_input = list(set(input_all) - set(input_initializer))
    
    # Debug: Lets make sure we pass in the right input names
    # Note: apparantly this is creating a set. A set cannnot be indexed to get single value.
    input_names = {i.name for i in input_all} 
    output_names = {i.name for i in output}
    print("Input Names:\n", input_names, "\n")
    print("Output Names:\n", output_names, "\n")
    
    #print(type(input_names))
    input_names = list(input_names)
    input_name = input_names[0]
    
    # Call converter (input: main model input name)
    modelKeras = onnx_to_keras(onnxIn, input_names)
    
    #util.ModelSavedDialogue(modelOut)

def ONNX_PyTorch(onnxIn, modelOut):
    print("Currently PyTorch does not support conversion from ONNX.\n")
          
    # Convert to PyTorch
    print("Currently PyTorch does not suuport ONNX model convertion.")
    # https://github.com/pytorch/pytorch/issues/21683
    
def ONNX_ScikitLearn(onnxIn, modelOut):
    print("Converting ONNX file to Scikit-Learn")
    
    util.ModelSavedDialogue(modelOut)

def ONNX_TensorFlow(onnxIn, modelOut):
    import os
    import onnx_tf
    from onnx_tf.backend import prepare
    import warnings
    
    warnings.filterwarnings('ignore')
    
    #print("onnx_tf:\t", onnx_tf.__version__)
    
    # Import ONNX model to TensorFlow
    print("Preparing ONNX file...")
    tf_rep = prepare(onnxIn)
    tf_rep.export_graph(modelOut)
    
    # Display results
    print("\nDisplaying model...", "\n"
          "Input nodes: ", tf_rep.inputs, "\n"
          "Model output nodes: ", tf_rep.outputs, "\n\n"
          "All Model nodes: ", tf_rep.tensor_dict)
    
    util.ModelSavedDialogue(modelOut)
   