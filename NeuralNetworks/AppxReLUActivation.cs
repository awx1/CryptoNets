// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

using HEWrapper;
using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetworks
{
    public class AppxReLUActivation : BaseLayer
    {
        public override IMatrix Apply(IMatrix m)
        {
            // Generate {Scale, Format, # of Rows, # of Columns}
            var m_scale = Source.GetOutputScale();
            var m_format = m.Format;
            var m_row = m.RowCount;
            var m_column = m.ColumnCount;

            // Polynomial Constants
            double[] constants = {0.125, 0.25, 0.5};

            // Generates the matrices of polynomial constants
            var M = Matrix<double>.Build;
            var cons_0 = M.Dense((int)m_row, (int)m_column, constants[0]);
            var cons_1 = M.Dense((int)m_row, (int)m_column, constants[1]);
            var cons_2 = M.Dense((int)m_row, (int)m_column, constants[2]);

            // Encrypts the matricies of polynomial constants
            var sec_deg = Factory.GetPlainMatrix(cons_0, m_format, m_scale);
            var fir_deg = Factory.GetPlainMatrix(cons_1, m_format, m_scale);
            var zero_deg = Factory.GetPlainMatrix(cons_2, m_format, m_scale);

            // Builds the 2nd deg polynomial of ReLU
            var m_mult_mult = ProcessInEnv(env => m.ElementWiseMultiply(m, env));
            m_mult_mult.RegisterScale(m_scale);
            var sec_poly = ProcessInEnv(env => m_mult_mult.ElementWiseMultiply(sec_deg, env));
            // Builds the 1st deg polynomial of ReLU
            var fir_poly = ProcessInEnv(env => m.ElementWiseMultiply(fir_deg, env));
            // Sums the two previous pieces
            var store = ProcessInEnv(env => sec_poly.Add(fir_poly, env));
            // Scales the 0th deg polynomial to s^2 & builds ReLU
            zero_deg.RegisterScale(m_scale * m_scale);
            var poly = ProcessInEnv(env => store.Add(zero_deg, env));
            // Disposal
            m_mult_mult.Dispose();
            sec_poly.Dispose();
            fir_poly.Dispose();
            store.Dispose();
            zero_deg.Dispose();
            fir_deg.Dispose();
            sec_deg.Dispose();

            return poly;
        }

        public override double GetOutputScale()
        {
            var s = Source.GetOutputScale();
            return s * s;
        }
    }
}
