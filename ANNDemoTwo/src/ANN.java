import java.util.Random;

public class ANN {
	// initializing the layers
	static int inputLayerSize, hiddenLayerOneSize, hiddenLayerTwoSize, outputLayerSize;
	
	// private classes for changing the layers
	private static void changeInputLayerSize (int newInputLayerSize) {
		if (newInputLayerSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			inputLayerSize = newInputLayerSize;
		}
	}
	private static void changeHiddenLayerOneSize (int newHiddenLayerOneSize) {
		if (newHiddenLayerOneSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			hiddenLayerOneSize = newHiddenLayerOneSize;
		}
	}
	private static void changeHiddenLayerTwoSize (int newHiddenLayerTwoSize) {
		if (newHiddenLayerTwoSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			hiddenLayerTwoSize = newHiddenLayerTwoSize;
		}
	}
	private static void changeOutputLayerSize (int newOutputLayerSize) {
		if (newOutputLayerSize <= 0) {
			throw new ArithmeticException("layers must have more than one node");
		} else {
			outputLayerSize = newOutputLayerSize;
		}
	}
	
	// the instance of the Random class we will be using to initialize the ANN
	static Random rand = new Random();
	
	// to randomize the values in an array between 0 and 1 (used for the biases and weights arrays)
	private static void randomizeArr (double[] arr) {
		for (int i = 0; i < arr.length; i++) {
			// generates a double on the interval [0.0, 1.0]
			arr[i] = rand.nextDouble();
		}
	}
	
	public ANN (int inputLayerSize, int hiddenLayerOneSize, int hiddenLayerTwoSize, int outputLayerSize) {
		// calls all the methods for changing the size of the layers. They will throw exceptions if illegal values are attempted
		changeInputLayerSize(inputLayerSize);
		changeHiddenLayerOneSize(hiddenLayerOneSize);
		changeHiddenLayerTwoSize(hiddenLayerTwoSize);
		changeOutputLayerSize(outputLayerSize);
		
		// initializing and randomizing all the biases
		double[] biasesLayerOne = new double[hiddenLayerOneSize];
		randomizeArr (biasesLayerOne);
		double[] biasesLayerTwo = new double[hiddenLayerTwoSize];
		randomizeArr (biasesLayerTwo);
		double[] biasesOutputLayer = new double[outputLayerSize];
		randomizeArr (biasesOutputLayer);
		
		// initializing and randomizing all the weights
		double[] weightsAfterInputLayer = new double[inputLayerSize * hiddenLayerOneSize];
		randomizeArr (weightsAfterInputLayer);
		double[] weightsAfterLayerOne = new double[hiddenLayerOneSize * hiddenLayerTwoSize];
		randomizeArr (weightsAfterLayerOne);
		double[] weightsAfterLayerTwo = new double[hiddenLayerTwoSize * outputLayerSize];
		randomizeArr (weightsAfterLayerTwo);
	}
	
	public double calculate () {
		double value = 0;
		return value;
	}
}