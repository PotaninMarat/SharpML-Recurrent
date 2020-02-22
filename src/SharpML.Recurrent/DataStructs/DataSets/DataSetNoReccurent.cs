using SharpML.Loss;
using SharpML.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpML.DataStructs.DataSets
{
    public class DataSetNoReccurent : DataSet
    {
        public DataSetNoReccurent(NNValue[] inputs, NNValue[] outputs, ILoss los)
        {
            InputDimension = inputs[0].Len;
            OutputDimension = outputs[0].Len;
            LossTraining = los;

            var data = GetSequences(inputs, outputs);

            Training = data;
            Validation = data;
            Testing = data;
        }

        static List<DataSequence> GetSequences(NNValue[] inputs, NNValue[] outputs)
        {
            List<DataSequence> dataSequences = new List<DataSequence>();

            for (int i = 0; i < inputs.Length; i++)
            {
                DataStep dataStep = new DataStep(inputs[i], outputs[i]);
                DataSequence dataSequence = new DataSequence();
                dataSequence.Steps.Add(dataStep);
                dataSequences.Add(dataSequence);
            }

            return dataSequences;
        }
    }
}
