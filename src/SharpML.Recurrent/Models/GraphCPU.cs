using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using SharpML.Activations;
using SharpML.DataStructs;

namespace SharpML.Models
{
    public class GraphCPU : IGraph
    {

        public bool ApplyBackprop { get; set; }

        public List<IRunnable> Backprop { get; set; }

        public GraphCPU()
            : this(true)
        {

        }

        public GraphCPU(bool applyBackprop)
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
                returnObj[loc] = m1[i];
                returnObj.DifData[loc] = m1.DifData[i];
                returnObj.StepCache[loc] = m1.StepCache[i];
                loc++;
            }
            for (int i = 0; i < m2.DataInTensor.Length; i++)
            {
                returnObj[loc] = m2[i];
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
                        m.DifData[i] += data.DataInTensor[i]*returnObj.DifData[i];
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

        /// <summary>
        /// Свертка
        /// </summary>
        /// <param name="input">Тензор входа</param>
        /// <param name="filters">Фильтры</param>
        public NNValue Convolution(NNValue input, NNValue[] filters, int padX, int padY)
        {

            int outpH, outpW, outpD = filters.Length;
                
         
           outpH = input.H - filters[0].H + 1+padY;
           outpW = input.W - filters[0].W + 1+padX;

            if ( (outpW < 1) || (outpH < 1) )
            {
                throw new Exception("Недостаточная размерность выхода");
            }


            NNValue returnObj = new NNValue(outpH, outpW, outpD);

            //  Parallel.For(0, outpD, new ParallelOptions() { MaxDegreeOfParallelism = 2}, s =>
            for (int s = 0; s < outpD; s++)
            {

                

                for (int y = -padY; y < outpH; y++)
                {

                    for (int x = -padX; x < outpW; x++)
                    {
                        for (int z = 0; z < input.D; z++)
                        {
                            for (int dy = 0; dy < filters[0].H; dy++)
                            {
                                int y1 = y + dy;

                                for (int dx = 0; dx < filters[0].W; dx++)
                                {
                                    int x1 = x + dx;
                                    if ((x1 > -1) && (y1 > -1)&& (x1 < outpW) && (y1 < outpH))
                                    returnObj[y1, x1, s] += input[y + dy, x + dx, z] * filters[s][dy, dx, z];
                                }
                            }
                        }
                    }
                }
            }

            //------------------------------------------------------------------------------------------
            
                
                    

                    // Обратный проход
            if (this.ApplyBackprop)
            {
                Runnable bp = new Runnable();
                bp.Run = delegate ()
                {

                    Parallel.For(0, outpD, new ParallelOptions() { MaxDegreeOfParallelism = 2 }, d =>
                    {
                        for (int z = 0; z < input.D; z++)
                        {
                            for (int y = 0; y < filters[0].H; y++)
                            {
                                for (int x = 0; x < filters[0].W; x++)
                                {
                                    for (int a = -padY; a < outpH; a++)
                                    {
                                        
                                        for (int b = -padX; b < outpW; b++)
                                        {
                                            int x1 = b + x;
                                            int y1 = a + y;

                                            if ((y1 > -1) && (x1 > -1) && (y1 < outpH) && (x1 < outpW))
                                            {
                                                double delt = returnObj.DifData[outpW * y1 + x1 + outpW * outpH * d];
                                                filters[d].DifData[filters[0].W * y + x + filters[0].W * filters[0].H * z] +=
                                                delt * input[a + y, b + x, z];
                                            }
                                            // --------------- ------------- ------------ ---------------//
                                        }
                                    }
                                }
                            }
                        }
                    });

                    Parallel.For(0, input.D, new ParallelOptions() { MaxDegreeOfParallelism = 2 }, n =>
                    {
                        for (int y = -padY; y < input.H; y++)
                        {
                            for (int x = -padX; x < input.W; x++)
                            {
                                for (int i = 0; i < outpD; i++)
                                {
                                    for (int dy = 0; dy < filters[0].H; dy++)
                                    {
                                        var dyy = y - dy;
                                        for (int dx = 0; dx < filters[0].W; dx++)
                                        {
                                            var dxx = x - dx;
                                            if ((dyy > -1) && (dxx > -1) && (dyy < outpH) && (dxx < outpW))
                                            {
                                                if ((dxx > -1) && (dyy > -1) && (dyy < outpH) && (dxx < outpW))
                                                {
                                                    double delt = returnObj.DifData[outpW * (y - dy) + (x - dx) + outpW * outpH * i];
                                                    input.DifData[input.W * y + x + input.W * input.H * n] += delt * filters[i][dy, dx, n];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    });
                };
                Backprop.Add(bp);
            }
            return returnObj;
        }

        /// <summary>
        /// Макс пул
        /// </summary>
        /// <param name="inp"></param>
        /// <param name="h"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        public NNValue MaxPooling(NNValue input, int h, int w)
        {
            //Func<double[], double> funcMax = (dataArray) =>
            //{
            //    return dataArray.Max();
            //};

            int outpH = input.H/h, outpW = input.W / w, outpD = input.D;
            double[] data = new double[h*w];
            bool[,,] map = new bool[input.H,input.W,input.D];


            if ((outpW < 1) || (outpH < 1))
            {
                throw new Exception("Недостаточная размерность выхода");
            }

            int modW = input.W % w;
            int modH = input.H % h;

            int maxX = input.W - modW;
            int maxY = input.H - modH;

            NNValue returnObj = new NNValue(outpH, outpW, outpD);

            for(int s = 0; s < outpD; s++)
            {
                for (int y = 0, y1 = 0; y < maxY; y += h, y1++)
                {
                    for (int x = 0, x1 = 0; x < maxX; x += w, x1++)
                    {

                        int i = y, j = x;

                        for (int dy = 0; dy < h; dy++)
                        {

                            for (int dx = 0; dx < w; dx++)
                            {
                                if (input[i, j, s] < input[y + dy, x + dx, s])
                                {
                                    i = y + dy;
                                    j = x + dx;
                                }
                            }
                        }

                        var max = input[i, j, s];
                        returnObj[y1, x1, s] = max;

                        map[i, j, s] = true;
                    }
                }
            };
            //------------------------------------------------------------------------------------------

            // Обратный проход
            if (this.ApplyBackprop)
            {
                int i = 0;

                Runnable bp = new Runnable();
                bp.Run = delegate ()
                {
                    for(int n = 0; n<input.D; n++)
                    {
                        for (int y = 0; y < input.H; y++)
                        {
                            for (int x = 0; x < input.W; x++)
                            {
                                 input.DifData[input.W * y + x + input.W * input.H * n] = map[y,x,n]? returnObj.DifData[i++]:0; 
                            }
                        }
                    }
                };
                Backprop.Add(bp);
            }
            return returnObj;

        }

        /// <summary>
        /// Преобразование размерности тензора
        /// </summary>
        /// <param name="input">Тензор</param>
        /// <param name="newShape">Новая размерность</param>
        /// <param name="gain">Усиление градиента</param>
        /// <returns></returns>
        public NNValue ReShape(NNValue input, Shape newShape, float gain)
        {
            NNValue outp = input.Clone();

            outp.H = newShape.H;
            outp.W = newShape.W;
            outp.D = newShape.D;

            // Обратный проход
            if (this.ApplyBackprop)
            {

                Runnable bp = new Runnable();
                bp.Run = delegate ()
                {
                    if (gain == 1.0)
                    {
                        for (int i = 0; i < input.Len; i++)
                        {
                            input.DifData[i] = outp.DifData[i];
                        }
                    }
                    else
                    {
                        for (int i = 0; i < input.Len; i++)
                        {
                            input.DifData[i] = gain*outp.DifData[i];
                        }
                    }
                };
                Backprop.Add(bp);
            }


            return outp;
        }

        public NNValue UnPooling(NNValue inp, int h, int w)
        {
            int outpH = inp.H * h, outpW = inp.W * w, outpD = inp.D;
            //double[] data = new double[h * w];
            //bool[,,] map = new bool[inp.H, inp.W, inp.D];


            if ((outpW < 1) || (outpH < 1))
            {
                throw new Exception("Недостаточная размерность выхода");
            }

            
            NNValue returnObj = new NNValue(outpH, outpW, outpD);

            for (int s = 0; s < outpD; s++)
            {
                for (int y = 0, y1 = 0; y < outpH; y += h, y1++)
                {
                    for (int x = 0, x1 = 0; x < outpW; x += w, x1++)
                    {
                        returnObj[y, x, s] = inp[y1,x1,s];
                    }
                }
            };
            //------------------------------------------------------------------------------------------

            // Обратный проход
            if (this.ApplyBackprop)
            {

                Runnable bp = new Runnable();
                bp.Run = delegate ()
                {
                    for (int n = 0; n < inp.D; n++)
                    {
                        for (int y = 0, y1 = 0; y < inp.H; y++, y1+=h)
                        {
                            for (int x = 0, x1 =0; x < inp.W; x++, x1+=w)
                            {
                                inp.DifData[inp.W * y + x + inp.W * inp.H * n] = returnObj.DifData[outpW * y1 + x1 + outpW * outpW * n];
                            }
                        }
                    }
                };
                Backprop.Add(bp);
            }
            return returnObj;
        }
    }
}
