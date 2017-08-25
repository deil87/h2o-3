package hex.genmodel.algos.deeplearning;

import java.io.Serializable;

public class ActivationUtils {
  // drived from GLMMojoModel
  public interface ActivationFunctions extends Serializable {
    float[] eval(float[] x, double drop_out_ratio);
  }

  public ActivationFunctions createActFuns(String activation) {
    switch (activation) {
      case "Linear": return new LinearOut();
      case "Softmax": return new SoftmaxOut();
      case "ExpRectifierDropout": return new ExpRectifierDropoutOut();
      case "ExpRectifier": return new ExpRectifierOut();
      case "Rectifier": return new RectifierOut();
      case "RectifierDropout": return new RectifierDropoutOut();
      case "MaxoutDropout": return new MaxoutDropoutOut();
      case "Maxout": return new MaxoutOut();
      case "TanhDropout": return new TanhDropoutOut();
      case "Tanh": return new TanhOut();
      default: throw new UnsupportedOperationException("Unexpected activation function: "+activation);
    }
  }

  public static class LinearOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {  // do nothing
      return input;
    }
  }

  public static class SoftmaxOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];
      float scaling = 0;

      for (int index=0; index < nodeSize; index++) {
        output[index] = (float) Math.exp(input[index]);
        scaling += output[index];
      }

      for (int index=0; index < nodeSize; index++)
        output[index] /= scaling;

      return output;
    }
  }

  public static class ExpRectifierDropoutOut extends ExpRectifierOut {
    public float[] eval(float[] input, double drop_out_ratio) {
      float[] output = super.eval(input, drop_out_ratio);
      applyDropout(output, (float) drop_out_ratio, input.length);
      return output;
    }
  }

  public static float[] applyDropout(float[] input, float drop_out_ratio, int nodeSize) {
    if (drop_out_ratio > 0) {
      double multFact = 1-drop_out_ratio;
      for (int index = 0; index < nodeSize; index++)
        input[index] *= multFact;
    }

    return input;
  }

  public static class ExpRectifierOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];

      for (int index = 0; index < nodeSize; index++) {
        output[index] = input[index]>=0?input[index]:(float)Math.exp(input[index])-1;
      }
      return output;
    }
  }

  public static class RectifierOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];

      for (int index = 0; index < nodeSize; index++)
        output[index] = 0.5f*(input[index]+Math.abs(input[index])); // clever.  Copied from Neurons.java

      return output;
    }
  }

  public static class RectifierDropoutOut extends RectifierOut {
    public float[] eval(float[] input, double drop_out_ratio) {
      float[] output = super.eval(input, drop_out_ratio);
      applyDropout(output, (float) drop_out_ratio, input.length);
      return output;
    }
  }

  public static class MaxoutDropoutOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];

      return output;
    }
  }

  public static class MaxoutOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];

      return output;
    }
  }

  public static class TanhDropoutOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];

      return output;
    }
  }

  public static class TanhOut implements ActivationFunctions {
    public float[] eval(float[] input, double drop_out_ratio) {
      int nodeSize = input.length;
      float[] output = new float[nodeSize];

      return output;
    }
  }

}
