using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpML.Recurrent.Models;

namespace SharpML.Recurrent.Loss
{
    public interface ILoss
    {
        void Backward(NNValue actualOutput, NNValue targetOutput);
        double Measure(NNValue actualOutput, NNValue targetOutput);
    }
}
