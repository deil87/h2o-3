package hex.genmodel.algos.deeplearning;

import java.util.Arrays;
import java.util.List;

public class NeuralNetwork {  // represent one layer of neural network
  public String _activation;    // string that describe the activation function
  double _drop_out_ratio;              // drop_out_ratio for that layer
  public DeeplearningMojoModel.StoreWeightsBias _weights;  // store layer weight
  public DeeplearningMojoModel.StoreWeightsBias _bias;      // store layer bias
  public float[] _inputs;     // store input to layer
  public float[] _outputs;    // layer output
  public int _outSize;         // number of nodes in this layer
  public int _inSize;         // number of inputs to this layer
  List<String> _validActivation = Arrays.asList("Linear", "Softmax", "ExpRectifierDropout", "ExpRectifier",
          "Rectifier", "RectifierDropout", "MaxoutDropout", "Maxout", "TanhDropout", "Tanh");

  public NeuralNetwork(String activation, double drop_out_ratio, DeeplearningMojoModel.StoreWeightsBias weights,
                       DeeplearningMojoModel.StoreWeightsBias bias, float[] inputs, int outSize) {
    validateInputs(activation, drop_out_ratio, weights._wOrBValues.length, bias._wOrBValues.length, inputs.length,
            outSize);
    _activation=activation;
    _drop_out_ratio=drop_out_ratio;
    _weights=weights;
    _bias=bias;
    _inputs=inputs;
    _outSize=outSize;
    _inSize=_inputs.length;
    _outputs = new float[_outSize];
  }

  public float[] fprop1Layer() {
    float[] input2ActFun = formNNInputs();

    // apply activation function to form NN outputs
    return _outputs;
  }

  public float[] formNNInputs() {
    float[] input2ActFun = new float[_inSize];

    for (int row=0; row < _outSize; row++) {
      input2ActFun[row] = _bias._wOrBValues[row];  //
      for (int col=0; col < _inSize; col++) {
        input2ActFun[row] += _weights._wOrBValues[row * _outSize+col];
      }
    }
    return input2ActFun;
  }

  public void validateInputs(String activation, double drop_out_ratio, int weightLen, int biasLen, int inSize,
                             int outSize) {
    assert inSize*outSize == weightLen : "Your neural network layer number of input * number " +
            "of outputs should equal length of your weight vector";
    assert outSize == biasLen : "Number of bias should equal number of nodes in your nerual network" +
            " layer.";
    assert (drop_out_ratio>=0 && drop_out_ratio<1) : "drop_out_ratio must be >=0 and < 1.";
    assert(outSize > 0) : "number of nodes in neural network must exceed 0.";
    assert (_validActivation.contains(activation)) : "activation must be one of \"Linear\", \"Softmax\", " +
            "\"ExpRectifierDropout\", \"ExpRectifier\", \"Rectifier\", \"RectifierDropout\", \"MaxoutDropout\", " +
            "\"Maxout\", \"TanhDropout\", \"Tanh\"";
  }
}
