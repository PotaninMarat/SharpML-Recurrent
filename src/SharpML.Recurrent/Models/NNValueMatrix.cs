using System;
using System.IO;
using System.Text;

namespace SharpML.Recurrent.Models
{
    [Serializable]
    public class NNValue
    {
        /// <summary>
        /// Высота
        /// </summary>
        public int H { get; set; }
        /// <summary>
        /// Ширина
        /// </summary>
        public int W { get; set; }
        /// <summary>
        /// Глубина
        /// </summary>
        public int D { get; set; }
        /// <summary>
        /// Число элементов тензора
        /// </summary>
        public int Len { get; set; }
        /// <summary>
        /// Данные в тензоре
        /// </summary>
        public double[] DataInTensor;
        /// <summary>
        /// Данные диференцирования
        /// </summary>
        public double[] DifData;
        /// <summary>
        /// Кэш
        /// </summary>
        public double[] StepCache;

        int _s;

        public NNValue(int dim)
        {
            Len = H = dim;
            W = 1;
            D = 1;
            DataInTensor = new double[H];
            DifData = new double[H];
            StepCache = new double[H];
        }

        public NNValue(int h, int w)
        {
            H = h;
            W = w;
            D = 1;
            _s = Len = h * w;
            DataInTensor = new double[Len];
            DifData = new double[Len];
            StepCache = new double[Len];
        }

        public NNValue(int h, int w, int d)
        {
            H = h;
            W = w;
            D = d;
            _s = h * w;
            Len = _s* d;
            DataInTensor = new double[Len];
            DifData = new double[Len];
            StepCache = new double[Len];
        }

        public NNValue(double[] vector)
        {
            _s = Len = H = vector.Length;
            W = 1;
            D = 1;
            DataInTensor = vector;
            DifData = new double[Len];
            StepCache = new double[Len];
        }

        /// <summary>
        /// Перевод матрицы в строку
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            StringBuilder result = new StringBuilder();
            for (int r = 0; r < H; r++)
            {
                for (int c = 0; c < W; c++)
                {
                    result.Append(String.Format("{0}", GetW(r, c)) + " ");
                }
                result.Append("\n");
            }
            return result.ToString();
        }

        /// <summary>
        /// Перевод матрицы в массив строк
        /// </summary>
        /// <returns></returns>
        public string[] ToStrings()
        {
            string[] result = new string[H];

            for (int r = 0; r < H; r++)
            {
                for (int c = 0; c < W; c++)
                {
                    result[r] += String.Format("{0:F5}", GetW(r, c)) + " ";
                }
                result[r] = result[r].Trim();
            }
            return result;
        }

        public NNValue Clone()
        {
            NNValue result = new NNValue(H, W);
            for (int i = 0; i < DataInTensor.Length; i++)
            {
                result.DataInTensor[i] = DataInTensor[i];
                result.DifData[i] = DifData[i];
                result.StepCache[i] = StepCache[i];
            }
            return result;
        }

        public void ResetDw()
        {
            for (int i = 0; i < DifData.Length; i++)
            {
                DifData[i] = 0;
            }
        }

        public void ResetStepCache()
        {
            for (int i = 0; i < StepCache.Length; i++)
            {
                StepCache[i] = 0;
            }
        }

        public static NNValue Transpose(NNValue m)
        {
            NNValue result = new NNValue(m.W, m.H);
            for (int r = 0; r < m.H; r++)
            {
                for (int c = 0; c < m.W; c++)
                {
                    result.SetW(c, r, m.GetW(r, c));
                }
            }
            return result;
        }


        /// <summary>
        /// Заполнение тензора случайными числами
        /// </summary>
        /// <param name="h">Ширина</param>
        /// <param name="w">Высота</param>
        /// <param name="initParamsStdDev">ско</param>
        /// <param name="rng">Генератор псевдослуч. чисел</param>
        public static NNValue Random(int h, int w, double initParamsStdDev, Random rng)
        {
            NNValue result = new NNValue(h, w);
            for (int i = 0; i < result.Len; i++)
            {
                result.DataInTensor[i] = 2 * (rng.NextDouble() - 0.5) * initParamsStdDev;
            }
            return result;
        }

        /// <summary>
        /// Заполнение тензора случайными числами
        /// </summary>
        /// <param name="h">Ширина</param>
        /// <param name="w">Высота</param>
        /// <param name="d">Глубина</param>
        /// <param name="initParamsStdDev">ско</param>
        /// <param name="rng">Генератор псевдослуч. чисел</param>
        public static NNValue Random(int h, int w, int d, double initParamsStdDev, Random rng)
        {
            NNValue result = new NNValue(h, w, d);
            for (int i = 0; i < result.DataInTensor.Length; i++)
            {
                result.DataInTensor[i] = 2 * (rng.NextDouble() - 0.5) * initParamsStdDev;
            }
            return result;
        }

        public static NNValue Ident(int dim)
        {
            NNValue result = new NNValue(dim, dim);
            for (int i = 0; i < dim; i++)
            {
                result.SetW(i, i, 1.0);
            }
            return result;
        }

        public static NNValue Uniform(int rows, int cols, double s)
        {
            NNValue result = new NNValue(rows, cols);
            for (int i = 0; i < result.DataInTensor.Length; i++)
            {
                result.DataInTensor[i] = s;
            }
            return result;
        }

        public static NNValue Ones(int rows, int cols)
        {
            return Uniform(rows, cols, 1.0);
        }

        public static NNValue NegativeOnes(int rows, int cols)
        {
            return Uniform(rows, cols, -1.0);
        }

        public void SaveAsText(string path)
        {
            string[] conent = ToStrings();
            File.WriteAllLines(path, conent); 
        }

        private int GetByIndex(int h, int w)
        {
            int ix = W * h + w;
            return ix;
        }

        private int GetByIndex(int h, int w, int d)
        {
            int ix = W * h + w+_s*d;
            return ix;
        }

        private double GetW(int h, int w)
        {
            return DataInTensor[GetByIndex(h, w)];
        }

        private double GetW(int h, int w, int d)
        {
            return DataInTensor[GetByIndex(h, w, d)];
        }

        private void SetW(int h, int w, double val)
        {
            DataInTensor[GetByIndex(h, w)] = val;
        }

        private void SetW(int h, int w, int d, double val)
        {
            DataInTensor[GetByIndex(h, w, d)] = val;
        }
    }
}
