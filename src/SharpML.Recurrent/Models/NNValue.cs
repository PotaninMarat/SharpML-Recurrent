using SharpML.DataStructs;
using System;
using System.IO;
using System.Text;

namespace SharpML.Models
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
        /// <summary>
        /// Кэш для более сложных моделей обучения
        /// </summary>
        public double[] StepCache2;

        public double this[int i]
        {
            get
            {
                return DataInTensor[i];
            }
            set
            {
                DataInTensor[i] = value;
            }
        }

        public double this[int h, int w]
        {
            get
            {
                return GetW(h, w);
            }
            set
            {
                SetW(h, w, value);
            }
        }

        public double this[int h, int w, int d]
        {
            get
            {
                return GetW(h, w, d);
            }
            set
            {
                SetW(h, w, d, value);
            }
        }

        int _s;

        public NNValue(int dim)
        {
            Len = H = dim;
            W = 1;
            D = 1;
            DataInTensor = new double[H];
            DifData = new double[H];
            StepCache = new double[H];
            StepCache2 = new double[H];
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
            StepCache2 = new double[Len];
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
            StepCache2 = new double[Len];
        }

        public NNValue(double[] vector)
        {
            _s = Len = H = vector.Length;
            W = 1;
            D = 1;
            DataInTensor = vector;
            DifData = new double[Len];
            StepCache = new double[Len];
            StepCache2 = new double[Len];
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

        /// <summary>
        /// Перевод матрицы в массив строк
        /// </summary>
        /// <returns></returns>
        public string[] ToTxts()
        {
            string[] result = new string[Len + 4];
            result[0] = "h:" + H;
            result[1] = "w:" + W;
            result[2] = "d:" + D;
            result[3] = "data:";


            for (int i = 0; i < Len; i++)
            {
                result[i + 4] = "" + this[i];
            }
            return result;
        }

        /// <summary>
        /// Перевод матрицы в массив строк
        /// </summary>
        /// <returns></returns>
        public string[] ToTxtsNoInfo()
        {
            string[] result = new string[Len];
            
            for (int i = 0; i < Len; i++)
            {
                result[i] = "" + this[i];
            }
            return result;
        }

        /// <summary>
		/// Гауссовское распределение
		/// </summary>
		/// <returns>Возвращает норм. распред величину СКО = 1, M = 0</returns>
		static public double Gauss(Random A)
        {
            double a = 2 * A.NextDouble() - 1,
            b = 2 * A.NextDouble() - 1,
            s = a * a + b * b;

            if (a == 0 && b == 0)
            {
                a = 0.000001;
                s = a * a + b * b;
            }

            return b * Math.Sqrt(Math.Abs(-2 * Math.Log(s) / s));
        }

        public NNValue Clone()
        {
            NNValue result = new NNValue(H, W, D);
            for (int i = 0; i < DataInTensor.Length; i++)
            {
                result.DataInTensor[i] = DataInTensor[i];
                result.DifData[i] = DifData[i];
                result.StepCache[i] = StepCache[i];
                result.StepCache2[i] = StepCache2[i];
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
                StepCache2[i] = 0;
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
        /// <param name="rnd">Генератор псевдослуч. чисел</param>
        public static NNValue Random(int h, int w, double initParamsStdDev, Random rnd)
        {
            NNValue result = new NNValue(h, w);
            for (int i = 0; i < result.Len; i++)
            {
                result.DataInTensor[i] = Gauss(rnd)*initParamsStdDev;
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
        public static NNValue Random(int h, int w, int d, double initParamsStdDev, Random rnd)
        {
            NNValue result = new NNValue(h, w, d);
            for (int i = 0; i < result.DataInTensor.Length; i++)
            {
                result.DataInTensor[i] = Gauss(rnd) * initParamsStdDev;
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
            string[] conent = ToTxts();
            File.WriteAllLines(path, conent);
        }
        public void SaveAsTextNoInfo(string path)
        {
            string[] conent = ToTxtsNoInfo();
            File.WriteAllLines(path, conent);
        }

        public Shape GetShape()
        {
            return new Shape(H, W, D);
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
