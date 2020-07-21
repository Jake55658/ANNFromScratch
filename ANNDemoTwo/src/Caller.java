public class Caller {
	// the main() method
	public static void main(String args[]) {
		// an instance of the ANN class (from the ANN.java file)
		ANN firstANN = new ANN(1, 5, 5, 1, 0, 1000, 0, 1000);

		/*
		 * the input array for the NN
		 * 
		 * we are defining it and assigning it values here since we are assigning it
		 * values here, the size of the array is automatically assigned according to how
		 * many values are passed in
		 */
		double[] inputs = new double[] {87};

		// the output array for the NN
		// we must assign the outputs[] array a size for the NN to work
		double[] outputs = new double[1];

		// using our instance of an ANN to calculate outputs based on the inputs
		firstANN.calculate(inputs, outputs);
	}
}