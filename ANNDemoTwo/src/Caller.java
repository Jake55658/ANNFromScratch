import java.io.IOException;

public class Caller {
	// the main() method
	public static void main(String args[]) throws IOException {
		// an instance of the env class which is not saved in the git repo and which
		// contains data like the file location for this project on my computer
		env environment = new env();

		// a String variable to hold the folder location for the ANN save files
		String firstANNSaveDataLocation = environment.getLocation() + "\\src\\ANNData\\firstANN\\";

		// initializing values
		int min = -10000;
		int max = 10000;
		boolean readSaveData = true;
		boolean learn = true;
		int dataSetSize = 10;
		int epochs = 10;
		boolean save = true;
		boolean test = true;
		int numTestInputs = 10;

		// an instance of the ANN class (from the ANN.java file)
		ANN firstANN = new ANN(1, 20, 20, 1, min, max, readSaveData, firstANNSaveDataLocation);

		// the learning
		trainANN(learn, dataSetSize, epochs, min, max, firstANN);

		// save the ANN
		saveANN(save, firstANN, firstANNSaveDataLocation);

		// test the ANN
		testANN(test, firstANN, numTestInputs, min, max);
	}

	// a method for the function that the ANN is trying to approximate
	private static double function(double x) {
		return x;
	}

	// a method for assigning inputs and outputs
	private static double[][] assignInputsForOneInputOneOutputFunctions(int numInputs, int minInput, int maxInput) {
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
			outputs[i][0] = function(inputs[i][0]);
		}
		return outputs;
	}

	// a method for testing the ANN
	private static void testANN(boolean test, ANN ANN, int numTestInputs, int minInput, int maxInput) {
		if (test) {
			// assign all the input values and the input and output arrays
			double[][] inputs = assignInputsForOneInputOneOutputFunctions(numTestInputs, minInput, maxInput);
			double[][] outputs = assignOutputsForOneInputOneOutputFunctions(inputs);

			// do all the calculations and print them
			for (int i = 0; i < inputs.length; i++) {
				ANN.calculate(inputs[i], outputs[i]);
				System.out.println("===================================");
				System.out.println("inputs[" + i + "][0] = " + inputs[i][0]);
				System.out.println("outputs[" + i + "][0] = " + outputs[i][0]);
				System.out.println("===================================");
			}
		}
	}

	// a method for training the ANN
	private static void trainANN(boolean learn, int dataSetSize, int epochs, int min, int max, ANN ANN) {
		if (learn) {
			// generate the test data
			double[][] givenInputs = assignInputsForOneInputOneOutputFunctions(dataSetSize, min, max);
			double[][] desiredOutputs = assignOutputsForOneInputOneOutputFunctions(givenInputs);
			
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