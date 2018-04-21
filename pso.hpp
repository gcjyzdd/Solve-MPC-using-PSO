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
/*Particle Swarm Optimization
 * specially designed for model predictive control
 * */
#ifndef __PSO_HPP__
#define __PSO_HPP__

#include <random>
#include <iostream>
#include <iomanip>

#define PI 3.1415926
#define PSO_INERTIA 0.7298

#define SWARM_SIZE 64
#define HORIZON 25
#define D_STATE 6
#define D_INPUT 2

typedef float dtype;


struct State
{
	dtype x,y,psi,v,cte,epsi;
	State(){x=0;y=0;psi=0;v=0;cte=0;epsi=0;}
};

struct Input
{
	dtype delta;
	dtype a;
	Input(){a=0;delta=0;}
};

struct States
{
	//x,y,psi,v,cte,epsi;
	dtype x[HORIZON];
	dtype y[HORIZON];
	dtype psi[HORIZON];
	dtype v[HORIZON];
	dtype cte[HORIZON];
	dtype epsi[HORIZON];
};

struct Inputs
{
	dtype delta[HORIZON-1];
	dtype a[HORIZON-1];
};

struct Solver_ARG
{
	State initState;
	dtype coeffs[3];

	Inputs init_guess;
	bool useGuess;
	size_t N;

	Solver_ARG()
	{
		N = 0;
		useGuess = false;
	}
};

struct Solver_Result
{
	Input input;
	State nextState;

	Inputs guess;
};

struct Particle
{
	States state;
	Inputs input;
	Inputs vel;
	Inputs input_b;

	dtype coeffs[3];	// [0 1 2]
};

struct Swarm
{
	Particle particle[SWARM_SIZE];	// a swarm of particles

	dtype fit[SWARM_SIZE];		// personal cost at each step
	dtype fit_bp[SWARM_SIZE];	// personal best cost over all steps

	Particle p_best;			// global best particle
	dtype fit_b;				// global best cost

	size_t index_best;			// index of global best particle

	dtype w;					// weight of previous velocity
	size_t step;				// current step number

	void printInput()
	{
		using std::cout;
		cout<<"input.delta = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			for(int j=0;j<(HORIZON-1);j++)
			{
				cout<<std::setprecision(3)<<particle[i].input.delta[j]<<"\t";
			}
			cout<<"\n";
		}
		cout<<"input.a = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			for(int j=0;j<(HORIZON-1);j++)
			{
				cout<<std::setprecision(3)<<particle[i].input.a[j]<<"\t";
			}
			cout<<"\n";
		}
	}
	void printVelocity()
	{
		using std::cout;
		cout<<"vel.delta = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			for(int j=0;j<(HORIZON-1);j++)
			{
				cout<<std::setprecision(3)<<particle[i].vel.delta[j]<<"\t";
			}
			cout<<"\n";
		}
		cout<<"vel.a = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			for(int j=0;j<(HORIZON-1);j++)
			{
				cout<<std::setprecision(3)<<particle[i].vel.a[j]<<"\t";
			}
			cout<<"\n";
		}
	}
	void printFitness()
	{
		using std::cout;
		cout<<"fitness = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			cout<<std::setprecision(5)<<fit[i]<<"\t";
		}
		cout<<"\n";
	}
};


struct PSO_Setting
{
	dtype tol;
	size_t MAX_STEP;
	size_t dec_stage;

	size_t step;
	size_t stop_size;

	// input limits
	dtype lb_d;
	dtype lb_a;
	dtype ub_d;
	dtype ub_a;

	dtype c1,c2;
	dtype w_max, w_min;
};

struct MPC_Setting
{
	dtype dt;
	dtype Lf;
	dtype ref_v;
	dtype w_cte;
};

struct MPC_PSO_Solver
{
	Swarm m_swarm;

	PSO_Setting m_pso_setting;
	MPC_Setting m_mpc_setting;

	Solver_ARG m_arg;

	Input m_result;

	dtype m_rst[8];

	// uniformed random numbers
	dtype m_rho1[SWARM_SIZE][HORIZON-1];
	dtype m_rho2[SWARM_SIZE][HORIZON-1];

	std::random_device m_rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 m_gen;

	MPC_PSO_Solver();

	void updateRNG()
	{
		std::uniform_real_distribution<> dis(0.0, 1.0);
		for (int n = 0; n < SWARM_SIZE; ++n) {
			for(int j=0;j<(HORIZON-1);j++)
			{
				//Use dis to transform the random unsigned int generated by gen into a double in [0, 1)
				m_rho1[n][j] = dis(m_gen);
				m_rho2[n][j] = dis(m_gen);
			}
		}
	}

	void printRNG()
	{
		using std::cout;
		cout<<"rho_1 = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			for(int j=0;j<HORIZON;j++)
			{
				cout<<std::setprecision(3)<<m_rho1[i][j]<<"\t";
			}
			cout<<"\n";
		}
		cout<<"rho_2 = \n";
		for(int i=0;i<SWARM_SIZE;i++)
		{
			for(int j=0;j<HORIZON;j++)
			{
				cout<<std::setprecision(3)<<m_rho2[i][j]<<"\t";
			}
			cout<<"\n";
		}
	}

	void calc_inertia_lin_dec();
	void initSettings();
	void setMPC(MPC_Setting &h_set);
	void setPSO(PSO_Setting &h_set);
	void initParticles();
	void evalFitness();
	void updateFitnessPersonal();
	void updateFitnessBest();
	void updateInput();
	void solve(Solver_ARG &h_arg, Solver_Result & rst);
	void printInfo()
	{
		std::cout<<"*****Solve Model Predictive Control With Particle Swarm Optimization*****\n\n";
		std::cout<<"#\n# Copyright (c) 2018, Changjie Guan.\n#\n# All rights reserved.\n#\n# Author: Changjie Guan (changjieguan@gmail.com)\n#\n\n\n";

		std::cout<<"Swarm Size = "<<SWARM_SIZE<<", Horizon = "<<HORIZON <<std::endl;

		std::cout<<"MAX_STEP = "<<m_pso_setting.MAX_STEP<<", tol = "<<m_pso_setting.tol<<std::endl;
		std::cout<<"c1 = "<<m_pso_setting.c1<<", c2 = "<<m_pso_setting.c2<<std::endl;
		std::cout<<"\n*********************************END*************************************\n\n\n";
	}
};

#endif
