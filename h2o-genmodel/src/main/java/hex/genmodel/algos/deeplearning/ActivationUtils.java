package hex.genmodel.algos.deeplearning;

import java.io.Serializable;

public class ActivationUtils {
  // drived from GLMMojoModel
  public interface ActivationFunctions extends Serializable {
    double[] eval(double[] x, double drop_out_ratio, int maxOutk);  // for MaxoutDropout
  }

  public ActivationFunctions createActFuns(String activation) {
    switch (activation) {
      case "Linear":
        return new LinearOut();
      case "Softmax":
        return new SoftmaxOut();
      case "ExpRectifierDropout":
        return new ExpRectifierDropoutOut();
      case "ExpRectifier":
        return new ExpRectifierOut();
      case "Rectifier":
        return new RectifierOut();
      case "RectifierDropout":
        return new RectifierDropoutOut();
      case "MaxoutDropout":
        return new MaxoutDropoutOut();
      case "Maxout":
        return new MaxoutOut();
      case "TanhDropout":
        return new TanhDropoutOut();
      case "Tanh":
        return new TanhOut();
      default:
        throw new UnsupportedOperationException("Unexpected activation function: " + activation);
    }
  }

  public static class LinearOut implements ActivationFunctions {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {  // do nothing
      return input;
    }
  }

  public static class SoftmaxOut implements ActivationFunctions {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      int nodeSize = input.length;
      double[] output = new double[nodeSize];
      double scaling = 0;

      for (int index = 0; index < nodeSize; index++) {
        output[index] =  Math.exp(input[index]);
        scaling += output[index];
      }

      for (int index = 0; index < nodeSize; index++)
        output[index] /= scaling;

      return output;
    }
  }

  public static class ExpRectifierDropoutOut extends ExpRectifierOut {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      double[] output = super.eval(input, drop_out_ratio, maxOutk);
      applyDropout(output, (float) drop_out_ratio, input.length);
      return output;
    }
  }

  public static double[] applyDropout(double[] input, double drop_out_ratio, int nodeSize) {
    if (drop_out_ratio > 0) {
      double multFact = 1 - drop_out_ratio;
      for (int index = 0; index < nodeSize; index++)
        input[index] *= multFact;
    }

    return input;
  }

  public static class ExpRectifierOut implements ActivationFunctions {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      int nodeSize = input.length;
      double[] output = new double[nodeSize];

      for (int index = 0; index < nodeSize; index++) {
        output[index] = input[index] >= 0 ? input[index] : Math.exp(input[index]) - 1;
      }
      return output;
    }
  }

  public static class RectifierOut implements ActivationFunctions {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      int nodeSize = input.length;
      double[] output = new double[nodeSize];

      for (int index = 0; index < nodeSize; index++)
        output[index] = 0.5f * (input[index] + Math.abs(input[index])); // clever.  Copied from Neurons.java

      return output;
    }
  }

  public static class RectifierDropoutOut extends RectifierOut {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      double[] output = super.eval(input, drop_out_ratio, maxOutk);
      applyDropout(output, drop_out_ratio, input.length);
      return output;
    }
  }

  public static class MaxoutDropoutOut extends MaxoutOut {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      double[] output = super.eval(input, drop_out_ratio, maxOutk);
      applyDropout(output, drop_out_ratio, input.length);
      return output;
    }
  }

  public static class MaxoutOut implements ActivationFunctions {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      int nodeSize = input.length / maxOutk;  // maxOutk times the actual output length
      double[] output = new double[nodeSize];

      for (int k = 0; k < maxOutk; k++) {
        int countInd = k * maxOutk;
        double temp = input[countInd];
        for (int index = 0; index < nodeSize; index++) {
          countInd += index;
          temp = temp > input[countInd] ? temp : input[countInd];
        }
        output[k] = temp;
      }
      return output;
    }
  }

  public static class TanhDropoutOut extends TanhOut {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      int nodeSize = input.length;
      double[] output = super.eval(input, drop_out_ratio, maxOutk);
      applyDropout(output, drop_out_ratio, input.length);
      return output;
    }
  }

  public static class TanhOut implements ActivationFunctions {
    public double[] eval(double[] input, double drop_out_ratio, int maxOutk) {
      int nodeSize = input.length;
      double[] output = new double[nodeSize];

      for (int index=0; index < nodeSize; index++)
        output[index] = 1-2/(1+(float)Math.exp(2*input[index]));

      return output;
    }
  }

}
