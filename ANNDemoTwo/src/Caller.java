import java.io.IOException;

public class Caller {
	// the main() method
	public static void main(String args[]) throws IOException {
		// an instance of the env class which is not saved in the git repo and which
		// contains data like the file location for this project on my computer
		env environment = new env();

		// a String variable to hold the folder location for the ANN save files
		String firstANNSaveDataLocation = environment.getLocation() + "\\src\\ANNData\\secondANN\\";

		// initializing values
		int minInput = -10000;
		int minOutput = 0;
		int maxInput = 10000;
		int maxOutput = 1;
		// set to either "Sigmoid", "ReLU", or "Linear"
		String activationFunction = "Linear";
		boolean binaryOutputs = true;
		boolean readSaveData = true;
		boolean learn = true;
		int minLearningInput = -10000;
		int minLearningOutput = 0;
		int maxLearningInput = 10000;
		int maxLearningOutput = 1;
		int dataSetSize = 100;
		int epochs = 50;
		boolean save = true;
		boolean test = true;
		int minTestInputs = -10000;
		int maxTestInputs = 10000;
		int numTestInputs = 10;

		// an instance of the ANN class (from the ANN.java file)
		ANN firstANN = new ANN(1, 5, 5, 2, minInput, maxInput, minOutput, maxOutput, readSaveData,
				firstANNSaveDataLocation, activationFunction, binaryOutputs);

		// the learning
		trainANN(learn, dataSetSize, epochs, minLearningInput, maxLearningInput, firstANN);

		// save the ANN
		saveANN(save, firstANN, firstANNSaveDataLocation);

		// test the ANN
		testANN(test, firstANN, numTestInputs, minTestInputs, maxTestInputs);
	}

	// a method for the function that the ANN is trying to approximate
	private static double functionOne(double x) {
		return x;
	}

	private static void functionTwo(double x, double[] outputNodes) {
		if (x >= 0) {
			outputNodes[0] = 1;
			outputNodes[1] = 0;
		} else {
			outputNodes[0] = 0;
			outputNodes[1] = 1;
		}
	}

	// a method for assigning inputs and outputs
	private static double[][] assignInputsForOneInputFunctions(int numInputs, int minInput, int maxInput) {
		double[][] inputs = new double[numInputs][1];
		double value = minInput + ((maxInput - minInput) / (numInputs + 1.0));
		for (int i = 0; i < numInputs; i++) {
			inputs[i][0] = value;
			value += (maxInput - minInput) / (numInputs + 1.0);
		}
		return inputs;
	}

	// a method for assigning desired outputs based on inputs
	private static double[][] assignOutputsForOneInputOneOutputFunctions(double[][] inputs) {
		double[][] outputs = new double[inputs.length][1];
		for (int i = 0; i < inputs.length; i++) {
			outputs[i][0] = functionOne(inputs[i][0]);
		}
		return outputs;
	}

	private static double[][] assignOutputsForOneInputTwoOutputFunctions(double[][] inputs) {
		double[][] outputs = new double[inputs.length][2];
		for (int i = 0; i < inputs.length; i++) {
			functionTwo(inputs[i][0], outputs[i]);
		}
		return outputs;
	}

	// a method for testing the ANN
	private static void testANN(boolean test, ANN ANN, int numTestInputs, int minInput, int maxInput) {
		if (test) {
			// assign all the input values and the input and output arrays
			double[][] inputs = assignInputsForOneInputFunctions(numTestInputs, minInput, maxInput);
			double[][] outputs = new double[inputs.length][2];

			// do all the calculations and print them
			for (int i = 0; i < inputs.length; i++) {
				ANN.calculate(inputs[i], outputs[i]);
				System.out.println("===================================");
				System.out.println("inputs[" + i + "][0] = " + inputs[i][0]);
				System.out.println("outputs[" + i + "][0] = " + outputs[i][0]);
				System.out.println("outputs[" + i + "][1] = " + outputs[i][1]);
				System.out.println("===================================");
			}
		}
	}

	// a method for training the ANN
	private static void trainANN(boolean learn, int dataSetSize, int epochs, int min, int max, ANN ANN) {
		if (learn) {
			// generate the test data
			double[][] givenInputs = assignInputsForOneInputFunctions(dataSetSize, min, max);
			double[][] desiredOutputs = assignOutputsForOneInputTwoOutputFunctions(givenInputs);

			// train the ANN
			for (int i = 0; i < epochs; i++) {
				ANN.learn(givenInputs, desiredOutputs, dataSetSize);
			}
		}
	}

	// a method for saving the ANN data
	private static void saveANN(boolean save, ANN ANN, String saveDataLocation) throws IOException {
		if (save) {
			ANN.save(saveDataLocation);
		}
	}
}