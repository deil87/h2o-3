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
  public boolean _imputeMeans;
  public int[] _units;  // size of neural network, input, hidden layers and output layer
  public double[] _all_drop_out_ratios; // input layer and hidden layers
  public StoreWeightsBias[] _weights; // stores weights of different layers
  public StoreWeightsBias[] _bias;    // store bias of different layers
  public int[] _catNAFill; // if mean imputation is true, mode imputation for categorical columns

  /***
   * Should set up the neuron network frame work here
   * @param columns
   * @param domains
   */
  DeeplearningMojoModel(String[] columns, String[][] domains) {
    super(columns, domains);
  }

  /***
   * This method will be derived from the scoring/prediction function of deeplearning model itself.
   * @param dataRow
   * @param offset
   * @param preds
   * @return
   */
  @Override
  public final double[] score0(double[] dataRow, double offset, double[] preds) {
    assert(dataRow != null) : "doubles are null"; // check to make sure data is not null
    float[] input2Neurons = new float[_units[0]];

    // transform inputs: standardize if needed, imputeMissings, convert categoricals
    setInput(dataRow, input2Neurons, _nums, _cats, _catoffsets, _normmul, _normsub, _use_all_factor_levels, !_imputeMeans);


    float[] floats;

    return preds;
  }

  @Override
  public double[] score0(double[] row, double[] preds) {
    return score0(row, 0.0, preds);
  }

  // class to store weight or bias for one neuron layer
  public static class StoreWeightsBias {
    double[] _wOrBValues; // store weight or bias arrays

    StoreWeightsBias(double[] values) {
      _wOrBValues = values;
    }
  }
}
