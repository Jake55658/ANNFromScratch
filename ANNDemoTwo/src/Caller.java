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
		boolean save = true;
		boolean test = true;

		// an instance of the ANN class (from the ANN.java file)
		ANN firstANN = new ANN(1, 3, 3, 1, min, max, readSaveData, firstANNSaveDataLocation);

		/*
		 * the learning:
		 */
		if (learn) {
			// generate the test data
			int dataSetSize = 100;
			// the first index is each data point, the second index is the data in each data
			// point
			double[][] givenInputs = new double[dataSetSize][1];
			double[][] desiredOutputs = new double[dataSetSize][1];
			// train the ANN
			int epochs = 10;
			for (int i = 0; i < epochs; i++) {
				// pass the testing data in

				// since we are using the "i" in the for() loop to keep track of the value to
				// pass into the inputs and outputs, a separate counter variable is used to keep
				// track of the index of the input and output arrays
				double value = min + ((max - min) / (dataSetSize + 1.0));
				for (int j = 0; j < dataSetSize; j++) {
					givenInputs[j][0] = value;
					desiredOutputs[j][0] = value;
					value += (max - min) / (dataSetSize + 1.0);
				}

				// do the actual training
				firstANN.learn(givenInputs, desiredOutputs, dataSetSize);
			}
		}

		// save the ANN
		if (save) {
			firstANN.save(firstANNSaveDataLocation);
		}

		// test the ANN
		if (test) {
			// assign all the input values and the input and output arrays
			int numInputs = 10;
			double[][] inputs = new double[numInputs][1];
			double[][] outputs = new double[numInputs][1];
			double value = min + ((max - min) / (numInputs + 1.0));
			for (int i = 0; i < numInputs; i++) {
				inputs[i][0] = value;
				value += (max - min) / (numInputs + 1.0);
			}
			// do all the calculations and print them
			for (int i = 0; i < inputs.length; i++) {
				firstANN.calculate(inputs[i], outputs[i]);
				System.out.println("=================================");
				System.out.println("inputs[" + i + "][0] = " + inputs[i][0]);
				System.out.println("outputs[" + i + "][0] = " + outputs[i][0]);
				System.out.println("=================================");
			}
		}
	}
}