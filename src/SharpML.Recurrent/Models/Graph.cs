using System;
using System.Collections.Generic;
using SharpML.Recurrent.Activations;

namespace SharpML.Recurrent.Models
{
    public class Graph
    {

        public bool ApplyBackprop { get; set; }


        public List<IRunnable> Backprop { get; set; }

        public Graph()
            : this(true)
        {

        }

        public Graph(bool applyBackprop)
        {
            this.ApplyBackprop = applyBackprop;
            this.Backprop = new List<IRunnable>();
        }

        public void Backward()
        {
            for (int i = Backprop.Count - 1; i >= 0; i--)
            {
                Backprop[i].Run();
            }
        }

        public NNValue ConcatVectors(NNValue m1, NNValue m2)
        {
            if (m1.W > 1 || m2.W > 1)
            {
                throw new Exception("Expected column vectors");
            }
            NNValue returnObj = new NNValue(m1.H + m2.H);

            int loc = 0;
            for (int i = 0; i < m1.DataInTensor.Length; i++)
            {
                returnObj.DataInTensor[loc] = m1.DataInTensor[i];
                returnObj.DifData[loc] = m1.DifData[i];
                returnObj.StepCache[loc] = m1.StepCache[i];
                loc++;
            }
            for (int i = 0; i < m2.DataInTensor.Length; i++)
            {
                returnObj.DataInTensor[loc] = m2.DataInTensor[i];
                returnObj.DifData[loc] = m2.DifData[i];
                returnObj.StepCache[loc] = m2.StepCache[i];
                loc++;
            }
            if (this.ApplyBackprop)
            {
                Runnable bp = new Runnable();
                bp.Run = delegate()
                {
                    int index0 = 0;
                    for (int i = 0; i < m1.DataInTensor.Length; i++)
                    {
                        m1.DataInTensor[i] = returnObj.DataInTensor[index0];
                        m1.DifData[i] = returnObj.DifData[index0];
                        m1.StepCache[i] = returnObj.StepCache[index0];
                        index0++;
                    }
                    for (int i = 0; i < m2.DataInTensor.Length; i++)
                    {
                        m2.DataInTensor[i] = returnObj.DataInTensor[index0];
                        m2.DifData[i] = returnObj.DifData[index0];
                        m2.StepCache[i] = returnObj.StepCache[index0];
                        index0++;
                    }
                };

                Backprop.Add(bp);
            }
            return returnObj;
        }

        public NNValue Nonlin(INonlinearity neuron, NNValue m)
        {
            NNValue returnObj = new NNValue(m.H, m.W);
            int n = m.DataInTensor.Length;
            returnObj = neuron.Forward(m);

            if (this.ApplyBackprop)
            {
                Runnable bp = new Runnable();
                bp.Run = delegate()
                {
                    var data = neuron.Backward(m);

                    for (int i = 0; i < n; i++)
                    {
                        m.DifData[i] += data.DataInTensor[i]* returnObj.DifData[i];
                    }

                };
                Backprop.Add(bp);
            }
            return returnObj;
        }

        public NNValue Mul(NNValue m1, NNValue m2)
        {
            if (m1.W != m2.H)
            {
                throw new Exception("matrix dimension mismatch");
            }

            int m1Rows = m1.H;
            int m1Cols = m1.W;
            int m2Cols = m2.W;
            NNValue returnObj = new NNValue(m1Rows, m2Cols);
            int outcols = m2Cols;
            for (int i = 0; i < m1Rows; i++)
            {
                int m1Col = m1Cols * i;
                for (int j = 0; j < m2Cols; j++)
                {
                    double dot = 0;
                    for (int k = 0; k < m1Cols; k++)
                    {
                        dot += m1.DataInTensor[m1Col + k] * m2.DataInTensor[m2Cols * k + j];
                    }
                    returnObj.DataInTensor[outcols * i + j] = dot;
                }
            }
            if (this.ApplyBackprop)
            {
                Runnable bp = new Runnable();
                bp.Run = delegate()
                {
                    for (int i = 0; i < m1.H; i++)
                    {
                        int outcol = outcols * i;
                        for (int j = 0; j < m2.W; j++)
                        {
                            double b = returnObj.DifData[outcol + j];
                            for (int k = 0; k < m1.W; k++)
                            {
                                m1.DifData[m1Cols * i + k] += m2.DataInTensor[m2Cols * k + j] * b;
                                m2.DifData[m2Cols * k + j] += m1.DataInTensor[m1Cols * i + k] * b;
                            }
                        }
                    }

                };
                Backprop.Add(bp);
            }
            return returnObj;
        }

        public NNValue Add(NNValue m1, NNValue m2)
        {
            if (m1.H != m2.H || m1.W != m2.W)
            {
                throw new Exception("matrix dimension mismatch");
            }
            NNValue returnObj = new NNValue(m1.H, m1.W);
            for (int i = 0; i < m1.DataInTensor.Length; i++)
            {
                returnObj.DataInTensor[i] = m1.DataInTensor[i] + m2.DataInTensor[i];
            }
            if (this.ApplyBackprop)
            {
                Runnable bp = new Runnable();
                bp.Run = delegate()
                {
                    for (int i = 0; i < m1.DataInTensor.Length; i++)
                    {
                        m1.DifData[i] += returnObj.DifData[i];
                        m2.DifData[i] += returnObj.DifData[i];
                    }
                };
                Backprop.Add(bp);
            }
            return returnObj;
        }

        public NNValue OneMinus(NNValue m)
        {
            NNValue ones = NNValue.Ones(m.H, m.W);
            return Subtract(ones, m);
        }

        public NNValue Subtract(NNValue m1, NNValue m2)
        {
            return Add(m1, Neg(m2));
        }

        public NNValue smul(NNValue m, double s)
        {
            NNValue m2 = NNValue.Uniform(m.H, m.W, s);
            return Elmul(m, m2);
        }

        public NNValue smul(double s, NNValue m)
        {
            NNValue returnObj = smul(m, s);
            return returnObj;
        }

        public NNValue Neg(NNValue m)
        {
            NNValue negones = NNValue.NegativeOnes(m.H, m.W);
            NNValue returnObj = Elmul(negones, m);
            return returnObj;
        }

        public NNValue Elmul(NNValue m1, NNValue m2)
        {
            if (m1.H != m2.H || m1.W != m2.W)
            {
                throw new Exception("matrix dimension mismatch");
            }
            NNValue returnObj = new NNValue(m1.H, m1.W);
            for (int i = 0; i < m1.DataInTensor.Length; i++)
            {
                returnObj.DataInTensor[i] = m1.DataInTensor[i] * m2.DataInTensor[i];
            }
            if (this.ApplyBackprop)
            {
                Runnable bp = new Runnable();
                bp.Run = delegate()
                {
                    for (int i = 0; i < m1.DataInTensor.Length; i++)
                    {
                        m1.DifData[i] += m2.DataInTensor[i] * returnObj.DifData[i];
                        m2.DifData[i] += m1.DataInTensor[i] * returnObj.DifData[i];
                    }
                };
                Backprop.Add(bp);
            }
            return returnObj;
        }
    }
}
