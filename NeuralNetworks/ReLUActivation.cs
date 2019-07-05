// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using HEWrapper;

namespace NeuralNetworks
{
    public class ReLUActivation : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            return (ProcessInEnv(env => m.ElementWiseMultiply(m, env)));
        }

        public override double GetOutputScale()
        {
            var s = Source.GetOutputScale();
            if (s < 0)
            {
                return 0;
            }
            else
            {
                return s;
            }
        }
    }
}