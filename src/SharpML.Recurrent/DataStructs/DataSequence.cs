using System;
using System.Collections.Generic;

namespace SharpML.DataStructs
{
    public class DataSequence
    {
        public List<DataStep> Steps { get; set; }

        public DataSequence()
        {
            Steps = new List<DataStep>();
        }

        public DataSequence(List<DataStep> steps)
        {
            this.Steps = steps;
        }

        public override string ToString()
        {
            String result = "";
           // result += "========================================================\n";
            foreach (DataStep step in Steps)
            {
                result += step.ToString() + "\n";
            }
            //result += "========================================================\n";
            return result;
        }


    }
}
