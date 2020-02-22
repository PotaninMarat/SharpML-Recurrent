using SharpML.Models;
using System;

namespace SharpML.DataStructs
{
    public class DataStep {

	public NNValue Input = null;
	public NNValue TargetOutput = null;
	
	public DataStep() {
		
	}

        public DataStep(double[] input, double[] targetOutput)
        {
            Input = new NNValue(input);
            if (targetOutput != null)
            {
               TargetOutput = new NNValue(targetOutput);
            }
        }

        public DataStep(double[] input)
        {
            Input = new NNValue(input);
        }


        public DataStep(NNValue input, NNValue targetOutput)
        {
            Input = input.Clone();
            TargetOutput = targetOutput.Clone();
        }

        public DataStep(NNValue input)
        {
            Input = input.Clone();
        }

        public override string ToString() {
		String result = "";
		for (int i = 0; i < Input.DataInTensor.Length; i++) {
            result += String.Format("{0:N5}", Input.DataInTensor[i]) + "\t";
		}
		result += "\t->\t";
		if (TargetOutput != null) {
            for (int i = 0; i < TargetOutput.DataInTensor.Length; i++)
            {
                result += String.Format("{0:N5}", TargetOutput.DataInTensor[i]) + "\t";
			}
		}
		else {
			result += "___\t";
		}
		return result;
	}
}
}
