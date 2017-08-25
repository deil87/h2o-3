package hex.genmodel.algos.deeplearning;

import hex.genmodel.MojoModel;

public class DeeplearningMojoModel extends MojoModel {
  public int _mini_batch_size;
  public int _nums; // number of numerical columns
  public int _cats; // number of categorical columns
  public int[] _catoffsets;
  public double[] _normmul;
  public double[] _normsub;
  public double[] _normrespmul;
  public double[] _normrespsub;
  public boolean _use_all_factor_levels;
  public String _activation;
  public String[] _allActivations;  // store activation function of all layers
  public boolean _imputeMeans;
  public int[] _units;  // size of neural network, input, hidden layers and output layer
  public double[] _all_drop_out_ratios; // input layer and hidden layers
  public StoreWeightsBias[] _weights; // stores weights of different layers
  public StoreWeightsBias[] _bias;    // store bias of different layers
  public int[] _catNAFill; // if mean imputation is true, mode imputation for categorical columns
  public int _numLayers;    // number of neural network layers.

  /***
   * Should set up the neuron network frame work here
   * @param columns
   * @param domains
   */
  DeeplearningMojoModel(String[] columns, String[][] domains) {
    super(columns, domains);
  }

  public void init() {
    _numLayers = _units.length-1;
    _allActivations = new String[_numLayers];
    for (int index=0; index < (_numLayers-1); index++)
      _allActivations[index]=_activation;
    _allActivations[-1] = this.isClassifier()?"Softmax":"Linear";

  }

  /***
   * This method will be derived from the scoring/prediction function of deeplearning model itself.  However,
   * we followed closely what is being done in deepwater mojo.  The variable offset is not used.
   * @param dataRow
   * @param offset
   * @param preds
   * @return
   */
  @Override
  public final double[] score0(double[] dataRow, double offset, double[] preds) {
    assert(dataRow != null) : "doubles are null"; // check to make sure data is not null
    float[] input2Neurons = new float[_units[0]]; // store inputs into the neural network
    float[] neuronsOutput;  // save output from a neural network layer

    // transform inputs: standardize if needed, imputeMissings, convert categoricals
    setInput(dataRow, input2Neurons, _nums, _cats, _catoffsets, _normmul, _normsub, _use_all_factor_levels, !_imputeMeans);

    // proprogate inputs through neural network
    for (int layer=0; layer < _numLayers; layer++) {
      NeuralNetwork oneLayer = new NeuralNetwork(_allActivations[layer], _all_drop_out_ratios[layer], _weights[layer],
              _bias[layer], input2Neurons, _units[layer+1]);
      neuronsOutput = oneLayer.fprop1Layer();


    }

    // Correction for classification or standardize outputs

    return preds;
  }

  public double[] fprop(float[] input2Neurons) {

    double[] outputs = new double[_units[-1]];  // initiate neural network outputs


    return outputs;
  }

  @Override
  public double[] score0(double[] row, double[] preds) {
    return score0(row, 0.0, preds);
  }

  // class to store weight or bias for one neuron layer
  public static class StoreWeightsBias {
    float[] _wOrBValues; // store weight or bias arrays

    StoreWeightsBias(float[] values) {
      _wOrBValues = values;
    }
  }
}
