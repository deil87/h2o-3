package hex.deeplearning;

import hex.ModelMojoWriter;

import java.io.IOException;

import static water.H2O.technote;

public class DeepLearningMojoWriter extends ModelMojoWriter<DeepLearningModel,
        DeepLearningModel.DeepLearningParameters, DeepLearningModel.DeepLearningModelOutput> {

  @SuppressWarnings("unused")
  public DeepLearningMojoWriter() {}
  private DeepLearningModel.DeepLearningParameters _parms;
  private DeepLearningModelInfo _model_info;
  private DeepLearningModel.DeepLearningModelOutput _output;

  public DeepLearningMojoWriter(DeepLearningModel model) {
    super(model);
    _parms = model.get_params();
    _model_info = model.model_info();
    _output = model._output;
    if (_model_info.isUnstable()) { // do not generate mojo for unstable model
      throw new UnsupportedOperationException(technote(4, "Refusing to create a MOJO for an unstable model."));
    }
  }

  @Override
  public String mojoVersion() {
    return "1.00";
  }

  @Override
  protected void writeModelData() throws IOException {
    writekv("mini_batch_size", _parms._mini_batch_size);
    writekv("nums", _model_info.data_info._nums);
    writekv("cats", _model_info.data_info._cats);
    writekv("cat_offsets", _output.catoffsets);
    writekv("norm_mul", _output.normmul);
    writekv("norm_sub", _output.normsub);
    writekv("norm_resp_mul", _output.normrespmul);
    writekv("norm_resp_sub", _output.normrespsub);
    writekv("use_all_factor_levels", _parms._use_all_factor_levels);
    writekv("standardize", _parms._standardize);  // check if need to standardize input
    writekv("hidden", _parms._hidden);
    writekv("activation", _parms._activation);
    writekv("input_dropout_ratio", _parms._input_dropout_ratio);
    writekv("hidden_dropout_ratio", _parms._hidden_dropout_ratios);
    boolean imputeMeans=_parms._missing_values_handling.equals(DeepLearningModel.DeepLearningParameters.MissingValuesHandling.MeanImputation);
    writekv("mean_imputation", imputeMeans);
    if (imputeMeans && _model_info.data_info._cats>0) { // only add this if there are categorical columns
      writekv("cat_modes", _model_info.data_info.catNAFill());
    }
    writekv("neural_network_sizes", _model_info.units); // layer 0 is input, last layer is output
    // keep track of neuron network sizes, weights and biases. Layer 0 is the output layer.  Last layer is output layer
    int numberOfWeights = 1+_parms._hidden.length;
    double[] all_drop_out_ratios = new double[numberOfWeights];
    String[] weightKey = new String[numberOfWeights];
    String[] biasKey = new String[numberOfWeights];

    for (int index = 0; index < numberOfWeights; index++) {
      if (index==0) { // input layer
        all_drop_out_ratios[index]=_parms._input_dropout_ratio;
      } else {
        if (_parms._hidden_dropout_ratios != null) {
          all_drop_out_ratios[index]=_parms._hidden_dropout_ratios[index-1];
        } else {
          all_drop_out_ratios[index]=0.0;
        }
      }

      //generate hash key to store weights/bias of all layers
      weightKey[index] = "weight_layer"+index;
      writekv(weightKey[index], _model_info.get_weights(index).raw());
      biasKey[index] = "bias_layer"+index;
      writekv(biasKey[index], _model_info.get_biases(index).raw());
    }
  }
}
