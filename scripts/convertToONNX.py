
# Description: This script contains all import defs for a framework into ONNX file format.

import numpy
import tensorflow as tf
import util
print("Package Version:\n"
      "numpy:\t\t", numpy.__version__,"\n"
      "tensorflow:\t", tf.__version__, "\n")

# Keras
def Keras_ONNX(modelIn, modelONNX):
    from tensorflow import keras
    from tensorflow.keras import layers
    import keras2onnx
    
    # Load Keras model
    modelKeras = tf.keras.models.load_model(modelIn)
    print("Keras model loaded: ", modelIn)

    print("Displaying Keras model summary...\n")
    modelKeras.summary()
    print("-"*65, "\n", "-"*65)
    
    # Convert to ONNX
    debugMode = 0 
    print("Displaying ONNX model summary...")
    
    _, base = util.ChangeExtension(modelIn, '') 
    model = keras2onnx.convert_keras(modelKeras, base, debug_mode=0)

    # Save the model in ONNX format
    keras2onnx.save_model(model, modelONNX)
    
    util.ModelSavedDialogue(modelONNX)
# ----------------------------------------------------------
# PyTorch
def PyTorch_ONNX(modelIn, modelONNX):
    import torch
    import torchvision
    from torchvision import models

    # Load PyTorch model {.pt | .pth}
    model = torch.load(modelIn)
    print("PyTorch model loaded: ", modelIn)
        
    print("Displaying shape...\n", model)
    
    # Convert to ONNX format   
    torch.onnx.export(model, 
                      torch.randn(1, 3, 224, 224), 
                      modelONNX)
    
    util.ModelSavedDialogue(modelONNX)
# ---------------------------------------------------------- 
# Scikit-Learn    
def ScikitLearn_ONNX(modelIn, modelONNX):
    import pandas
    import pickle
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    # Load SciKit-Learn Model (as .pkl)
    with open(modelIn, 'rb') as file:
        model = pickle.load(file)
        
    print("Scikit_Learn model loaded: ", modelIn)
    
    # Convert to ONNX
    initial_type = [('float_input', FloatTensorType([None, 4]))]
    onnx = convert_sklearn(model, initial_types=initial_type)
    with open(modelONNX, "wb") as file:
        file.write(onnx.SerializeToString())
    
    util.ModelSavedDialogue(modelONNX)
# ----------------------------------------------------------
# TensorFlow
def TensorFlow_ONNX(modelIn, modelONNX):
    import tensorflow as tf
    import tf2onnx
    
    print("TensorFlow model loaded: ", modelIn)
    
    ext = util.GetExtension(modelIn)
    _, base = util.ChangeExtension(modelIn, "")
    modelDescription = base.join(" model")
    
    # Load TensorFlow model via one of two methods
    if ext == "pb":
    # Reference: https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/Exporting.html
        graph_def = tf.compat.v1.GraphDef()
        with open(modelIn, 'rb') as f:
            graph_def.ParseFromString(f.read())
        
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
        
        #inputs[:] = [i + ":0" for i in inputs]
        #outputs[:] = [o + ":0" for o in outputs]
    
        with tf.compat.v1.Session() as sess:
            g = tf2onnx.tfonnx.process_tf_graph(sess.graph, 
                                                input_names=inputs, 
                                                output_names=outputs)
            model_proto = g.make_model(modelDescription)
            checker = onnx.checker.check_model(model_proto)

            tf2onnx.utils.save_onnx_model(modelONNX, 
                                          feed_dict={}, 
                                          model_proto=model_proto)
    elif ext == "h5":
        tf.keras.models.load_model(modelIn) # expecting h5 format
    
        # Convert to ONNX
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, [2, 3], name="input")
            x_ = tf.compat.v1.add(x, x)
            _ = tf.compat.v1.identity(x_, name="output")

            # Convert Protobuf format and map to ONNX model
            onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, 
                                                         input_names=["input:0"], 
                                                         output_names=["output:0"])
            model_proto = onnx_graph.make_model(modelDescription)
            with open(modelONNX, "wb") as f:
                f.write(model_proto.SerializeToString())
    
    util.ModelSavedDialogue(modelONNX)
# ----------------------------------------------------------
def ONNX_Inference(modelONNX):
    import onnxruntime
    
    print("Performing ONNX Runtime session...\n")
    sess = onnxruntime.InferenceSession(modelONNX)