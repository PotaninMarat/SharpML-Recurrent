using SharpML.Activations;
using SharpML.Loss;
using System.Collections.Generic;

namespace SharpML.DataStructs
{
    public abstract class DataSet 
    {
        public Shape InputShape { get; set; }
        public Shape OutputShape { get; set; }
        public ILoss LossFunction { get; set; }
        public List<DataSequence> Training { get; set; }
        public List<DataSequence> Validation { get; set; }
        public List<DataSequence> Testing { get; set; }

        //public virtual void DisplayReport(INetwork network, Random rng)
        //{

        //}

        public virtual INonlinearity GetModelOutputUnitToUse()
        {
            return null;
        }
}
}
