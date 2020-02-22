using System;
using System.Collections.Generic;
using System.Linq;
using SharpML.Models;

namespace SharpML.Util
{
    public class Util
    {

        public static int PickIndexFromRandomVector(NNValue probs, Random r)
        {
            double mass = 1.0;
            for (int i = 0; i < probs.DataInTensor.Length; i++)
            {
                double prob = probs.DataInTensor[i] / mass;
                if (r.NextDouble() < prob)
                {
                    return i;
                }
                mass -= probs.DataInTensor[i];
            }
            throw new Exception("no target index selected");
        }

        public static double Median(List<Double> vals)
        {
            vals = vals.OrderBy(a => a).ToList();
            int mid = vals.Count / 2;
            if (vals.Count % 2 == 1)
            {
                return vals[mid];
            }
            else
            {
                return (vals[mid - 1] + vals[mid]) / 2;
            }
        }
    }
}
