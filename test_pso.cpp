//
// Author: Changjie Guan (changjieguan@gmail.com)
//
//
//BSD 2-Clause License
//
//Copyright (c) 2018, Changjie
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//  list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#include<iostream>
#include <chrono>

#include "matplotlibcpp.h"
#include "pso.hpp"

using namespace std;
namespace plt = matplotlibcpp;


int main()
{
	MPC_PSO_Solver solver;
	Solver_Result result;

	solver.m_pso_setting.MAX_STEP = 3e3;
	solver.m_mpc_setting.dt = 0.05;

	Solver_ARG arg;
	dtype rst[8];

	dtype x = -1;
	dtype y = 10;
	dtype psi = 0;
	dtype v = 10;
	dtype cte = -11;//polyEval(coeffs, x) - y;
	dtype epsi = 0.;//psi - atan(coeffs[1]);

	arg.initState.x = x;
	arg.initState.y = y;
	arg.initState.psi = psi;
	arg.initState.v = v;
	arg.initState.cte = cte;
	arg.initState.epsi = epsi;

	arg.coeffs[0] = -1.;
	arg.coeffs[1] = 0.;
	arg.coeffs[2] = 0.;

	std::vector<double> x_vals = {x};
	std::vector<double> y_vals = {y};
	std::vector<double> psi_vals = {psi};
	std::vector<double> v_vals = {v};
	std::vector<double> cte_vals = {cte};
	std::vector<double> epsi_vals = {epsi};
	std::vector<double> delta_vals = {};
	std::vector<double> a_vals = {};

	int iters = 50;

	std::chrono::steady_clock::time_point begin =
			std::chrono::steady_clock::now();

	for (size_t i = 0; i < iters; i++)
	{
		solver.solve(arg, result);
		//solver.m_swarm.printFitness();
		//cout<<"best fitness = "<<solver.m_swarm.fit_b<<endl;

		x_vals.push_back(result.nextState.x);
		y_vals.push_back(result.nextState.y);
		psi_vals.push_back(result.nextState.psi);
		v_vals.push_back(result.nextState.v);
		cte_vals.push_back(result.nextState.cte);
		epsi_vals.push_back(result.nextState.epsi);

		delta_vals.push_back(result.input.delta);
		a_vals.push_back(result.input.a);

		arg.initState.x = result.nextState.x;
		arg.initState.y = result.nextState.y;
		arg.initState.psi = result.nextState.psi;
		arg.initState.v = result.nextState.v;
		arg.initState.cte = result.nextState.cte;
		arg.initState.epsi = result.nextState.epsi;

		arg.init_guess = result.guess;
		arg.useGuess = true;
		arg.N = 20;
	}
	std::chrono::steady_clock::time_point end =
			std::chrono::steady_clock::now();
	std::cout << "Average Time difference = "
			<< std::chrono::duration_cast<std::chrono::microseconds>(end -
					begin).count() / 1000./(float)iters
					<< "ms \n";

	plt::subplot(5, 1, 1);
	plt::title("CTE");
	plt::plot(cte_vals);

	plt::subplot(5, 1, 2);
	plt::title("Delta (Radians)");
	plt::plot(delta_vals);

	plt::subplot(5, 1, 3);
	plt::title("Acc");
	plt::plot(a_vals);

	plt::subplot(5, 1, 4);
	plt::title("Velocity");
	plt::plot(v_vals);

	plt::subplot(5, 1, 5);
	plt::title("Trajectory");
	plt::plot(x_vals, y_vals);

	plt::show();
	return 0;
}
