using SharpML.Networks;
using SharpML.Networks.Base;
using System.IO;

namespace SharpML.Util
{
    public static class Saver
    {
        public static void FFNNW_Save(INetwork network, string pathFolder)
        {
            string setting = "\\set.ffnnw";
            string bias = "\\bias.txtvector";
            string w = "\\w.txtmatrix";

            if (!Directory.Exists(pathFolder))
            {
                Directory.CreateDirectory(pathFolder);
            }


            string[] layers = new string[network.Layers.Count];
            int i = 0;

            foreach (var item in network.Layers)
            {
                var lay = (item as FeedForwardLayer);
                layers[i] = lay._w.W + " " + lay._w.H + " " + Compress(lay._f.ToString());

                string layPath = pathFolder + "\\" + i;

                if (!Directory.Exists(layPath))
                {
                    Directory.CreateDirectory(layPath);
                }

                lay._w.SaveAsText(layPath + w);
                lay._b.SaveAsText(layPath + bias);
                i++;
            }

            File.WriteAllLines(pathFolder + setting, layers);
        }

        static string Compress(string nonLin)
        {
            if (nonLin.Contains("TanhUnit"))
            {
                return "tanh";
            }
            else if (nonLin.Contains("SigmoidUnit"))
            {
                return "sigmoid";
            }
            else if (nonLin.Contains("RectifiedLinearUnit"))
            {
                return "relu";
            }
            else if (nonLin.Contains("LinearUnit"))
            {
                return "linear";
            }
            else if (nonLin.Contains("SoftmaxUnit"))
            {
                return "softmax";
            }
            

            return nonLin;
        }
    }
}
