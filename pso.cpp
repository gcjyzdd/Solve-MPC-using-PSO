//
// Copyright 2018 TASS International.
//
// All rights reserved.
//
// Author: Changjie Guan (changjie.guan@tassinternational.com)
//
#include <stdio.h>
#include <math.h>

#include "pso.hpp"

inline dtype pow2(dtype x)
{
	return x*x;
}

inline dtype polyval(dtype *coeffs, dtype x)
{
	return coeffs[0] + coeffs[1] * x + coeffs[2] * x * x;
}

void MPC_PSO_Solver::calc_inertia_lin_dec()
{
	m_swarm.step = m_swarm.step + 1;
	if(m_swarm.step <= m_pso_setting.dec_stage)
	{
		m_swarm.w = m_pso_setting.w_min + (m_pso_setting.w_max - m_pso_setting.w_min) * (m_pso_setting.dec_stage - m_swarm.step)/m_pso_setting.dec_stage;
	}
	else
	{
		m_swarm.w = m_pso_setting.w_min;
	}

}

MPC_PSO_Solver::MPC_PSO_Solver()
{
	m_gen = std::mt19937(m_rd()); //Standard mersenne_twister_engine seeded with rd()

	initSettings();

	printInfo();
}

void MPC_PSO_Solver::initSettings()
{
	m_pso_setting.MAX_STEP = 1e3;
	m_pso_setting.dec_stage = 3 * m_pso_setting.MAX_STEP / 4;
	m_pso_setting.step = 0;
	m_pso_setting.stop_size = 8;
	m_pso_setting.tol = 1e-5;
	m_pso_setting.lb_a = -1.;
	m_pso_setting.lb_d = -25/180.*PI;
	m_pso_setting.ub_a = 1.;
	m_pso_setting.ub_d = 25/180.*PI;
	m_pso_setting.c1 = 1.496;
	m_pso_setting.c2 = 1.496;
	m_pso_setting.w_max = PSO_INERTIA;
	m_pso_setting.w_min = 0.3;

	// mpc
	m_mpc_setting.Lf = 1.17;
	m_mpc_setting.dt = 1/100.;
	m_mpc_setting.ref_v = 50.;
	m_mpc_setting.w_cte = 1.0;

	m_swarm.w = PSO_INERTIA;
}

void MPC_PSO_Solver::setMPC(MPC_Setting &h_set)
{
	m_mpc_setting = h_set;
}

void MPC_PSO_Solver::setPSO(PSO_Setting &h_set)
{
	m_pso_setting = h_set;
}

void MPC_PSO_Solver::initParticles()
{
	updateRNG();
	for(int i=0; i<SWARM_SIZE;i++)
	{
		m_swarm.fit[i] = 1e9;
		m_swarm.fit_bp[i] = 1e9;
		// init coeffs
		for(int j=0;j<3;j++){m_swarm.particle[i].coeffs[j] = m_arg.coeffs[j];}

		// init state of each particle
		m_swarm.particle[i].state.x[0] = m_arg.initState.x;
		m_swarm.particle[i].state.y[0] = m_arg.initState.y;
		m_swarm.particle[i].state.psi[0] = m_arg.initState.psi;
		m_swarm.particle[i].state.v[0] = m_arg.initState.v;
		m_swarm.particle[i].state.cte[0] = m_arg.initState.cte;
		m_swarm.particle[i].state.epsi[0] = m_arg.initState.epsi;

		// init input of each particle
		for(int j=0;j<(HORIZON-1);j++)
		{
			// input
			m_swarm.particle[i].input.delta[j] = m_pso_setting.lb_d + m_rho1[i][j] * (m_pso_setting.ub_d - m_pso_setting.lb_d);
			m_swarm.particle[i].input.a[j] = m_pso_setting.lb_a + m_rho2[i][j] * (m_pso_setting.ub_a - m_pso_setting.lb_a);

			// input
			m_swarm.particle[i].input_b.delta[j] = m_swarm.particle[i].input.delta[j];
			m_swarm.particle[i].input_b.a[j] = m_swarm.particle[i].input.a[j];
		}
	}

	updateRNG();
	for(int i=0; i<SWARM_SIZE;i++)
	{
		for(int j=0;j<(HORIZON-1);j++)
		{
			// vel
			m_swarm.particle[i].vel.delta[j] = (m_pso_setting.lb_d + m_rho1[i][j] * (m_pso_setting.ub_d - m_pso_setting.lb_d) - m_swarm.particle[i].input.delta[j])/2.;
			m_swarm.particle[i].vel.a[j] = (m_pso_setting.lb_a + m_rho2[i][j] * (m_pso_setting.ub_a - m_pso_setting.lb_a) - m_swarm.particle[i].input.a[j])/2.;

		}
	}

	m_swarm.fit_b = 1e9;
	for(int j=0;j<(HORIZON-1);j++)
	{
		m_swarm.p_best.input.delta[j] = m_swarm.particle[0].input.delta[j];
		m_swarm.p_best.input.a[j] = m_swarm.particle[0].input.a[j];
	}

	// apply initial guess
	for(int i=0; i<m_arg.N;i++)
	{
		for(int j=0;j<(HORIZON-1);j++)
		{
			m_swarm.particle[i].input.delta[j] = m_arg.init_guess.delta[j];
			m_swarm.particle[i].input.a[j] = m_arg.init_guess.a[j];
		}
	}
}

void MPC_PSO_Solver::evalFitness()
{
	m_swarm.step++;

	#pragma omp parallel for
	for(int i=0; i<SWARM_SIZE;i++)
	{
		m_swarm.fit[i] = 0.;
		for(int j=0;j<(HORIZON-1);j++)
		{
			// update state
			m_swarm.particle[i].state.x[j+1]=m_swarm.particle[i].state.x[j]+
					m_swarm.particle[i].state.v[j]*cos(m_swarm.particle[i].state.psi[j])*m_mpc_setting.dt;

			m_swarm.particle[i].state.y[j+1]=m_swarm.particle[i].state.y[j]+
					m_swarm.particle[i].state.v[j]*sin(m_swarm.particle[i].state.psi[j])*m_mpc_setting.dt;

			m_swarm.particle[i].state.psi[j+1]=m_swarm.particle[i].state.psi[j] +
					m_swarm.particle[i].state.v[j]*m_swarm.particle[i].input.delta[j]/m_mpc_setting.Lf*m_mpc_setting.dt;

			m_swarm.particle[i].state.v[j+1]=m_swarm.particle[i].state.v[j]+m_swarm.particle[i].input.a[j]*m_mpc_setting.dt;

			m_swarm.particle[i].state.cte[j+1]=polyval(m_swarm.particle[i].coeffs, m_swarm.particle[i].state.x[j])-
					m_swarm.particle[i].state.y[j]+m_swarm.particle[i].state.v[j]*sin(m_swarm.particle[i].state.epsi[j])*m_mpc_setting.dt;

			m_swarm.particle[i].state.epsi[j+1]=m_swarm.particle[i].state.psi[j]-
					atan(m_swarm.particle[i].coeffs[1]+2*m_swarm.particle[i].coeffs[2]*m_swarm.particle[i].state.x[j])+
					m_swarm.particle[i].state.v[j]*m_swarm.particle[i].input.delta[j]/m_mpc_setting.Lf*m_mpc_setting.dt;

			m_swarm.fit[i]=m_swarm.fit[i]+m_mpc_setting.w_cte*pow2(m_swarm.particle[i].state.cte[j+1])+
					pow2(m_swarm.particle[i].state.epsi[j+1])+
					pow2(m_swarm.particle[i].state.v[j+1]-m_mpc_setting.ref_v);
		}
		for(size_t j=0;j<(HORIZON-1);j++)
		{
			m_swarm.fit[i]=m_swarm.fit[i]+pow2(m_swarm.particle[i].input.a[j])+pow2(m_swarm.particle[i].input.delta[j]);
		}
		for(size_t j=0;j<(HORIZON-2);j++)
		{
			m_swarm.fit[i]=m_swarm.fit[i]+pow2(m_swarm.particle[i].input.a[j+1]-m_swarm.particle[i].input.a[j])+
					pow2(m_swarm.particle[i].input.delta[j+1]-m_swarm.particle[i].input.delta[j]);
		}
	}
}

void MPC_PSO_Solver::updateFitnessPersonal()
{
	for(int i=0; i<SWARM_SIZE;i++)
	{
		if(m_swarm.fit[i] < m_swarm.fit_bp[i])
		{
			m_swarm.fit_bp[i] = m_swarm.fit[i];
			for(int j=0;j<(HORIZON-1);j++)
			{
				m_swarm.particle[i].input_b.delta[j] = m_swarm.particle[i].input.delta[j];
				m_swarm.particle[i].input_b.a[j] = m_swarm.particle[i].input.a[j];
			}
		}
	}
}

void MPC_PSO_Solver::updateFitnessBest()
{
	for(int i=0; i<SWARM_SIZE;i++)
	{
		if(m_swarm.fit_bp[i] < m_swarm.fit_b )
		{
			m_swarm.fit_b = m_swarm.fit_bp[i];
			m_swarm.index_best = i;
		}
	}
	for(int j=0;j<(HORIZON-1);j++)
	{
		m_swarm.p_best.input.delta[j] = m_swarm.particle[m_swarm.index_best].input_b.delta[j];
		m_swarm.p_best.input.a[j] = m_swarm.particle[m_swarm.index_best].input_b.a[j];

		m_swarm.p_best.state.x[j] = m_swarm.particle[m_swarm.index_best].state.x[j];
		m_swarm.p_best.state.y[j] = m_swarm.particle[m_swarm.index_best].state.y[j];
		m_swarm.p_best.state.psi[j] = m_swarm.particle[m_swarm.index_best].state.psi[j];
		m_swarm.p_best.state.v[j] = m_swarm.particle[m_swarm.index_best].state.v[j];
		m_swarm.p_best.state.cte[j] = m_swarm.particle[m_swarm.index_best].state.cte[j];
		m_swarm.p_best.state.epsi[j] = m_swarm.particle[m_swarm.index_best].state.epsi[j];
	}
}

void MPC_PSO_Solver::updateInput()
{
	updateRNG();

	#pragma omp parallel for
	for(size_t i = 0; i<SWARM_SIZE; i++)
	{
		for(size_t j=0;j<(HORIZON-1);j++)
		{
			m_swarm.particle[i].vel.a[j] = m_pso_setting.c1*m_rho2[i][j]*(m_swarm.particle[i].input_b.a[j]-m_swarm.particle[i].input.a[j]) +m_swarm.w*m_swarm.particle[i].vel.a[j];
			m_swarm.particle[i].vel.a[j] = m_pso_setting.c2*m_rho2[i][j]*(m_swarm.p_best.input.a[j]-m_swarm.particle[i].input.a[j]) +m_swarm.particle[i].vel.a[j];

			m_swarm.particle[i].vel.delta[j] = m_pso_setting.c1*m_rho1[i][j]*(m_swarm.particle[i].input_b.delta[j]-m_swarm.particle[i].input.delta[j]) + m_swarm.w*m_swarm.particle[i].vel.delta[j];
			m_swarm.particle[i].vel.delta[j] = m_pso_setting.c2*m_rho1[i][j]*(m_swarm.p_best.input.delta[j]-m_swarm.particle[i].input.delta[j]) + m_swarm.particle[i].vel.delta[j];

			m_swarm.particle[i].input.a[j] = m_swarm.particle[i].input.a[j] + m_swarm.particle[i].vel.a[j];
			m_swarm.particle[i].input.delta[j] = m_swarm.particle[i].input.delta[j] + m_swarm.particle[i].vel.delta[j];

			// TODO mirror velocity
			if(m_swarm.particle[i].input.a[j] > m_pso_setting.ub_a)
			{
				m_swarm.particle[i].input.a[j] = m_pso_setting.ub_a;
				m_swarm.particle[i].vel.a[j] = 0;
			}
			if(m_swarm.particle[i].input.a[j] < m_pso_setting.lb_a)
			{
				m_swarm.particle[i].input.a[j] = m_pso_setting.lb_a;
				m_swarm.particle[i].vel.a[j] = 0;
			}

			if(m_swarm.particle[i].input.delta[j] > m_pso_setting.ub_d)
			{
				m_swarm.particle[i].input.delta[j] = m_pso_setting.ub_d;
				m_swarm.particle[i].vel.delta[j] = 0;
			}
			if(m_swarm.particle[i].input.delta[j] < m_pso_setting.lb_d)
			{
				m_swarm.particle[i].input.delta[j] = m_pso_setting.lb_d;
				m_swarm.particle[i].vel.delta[j] = 0;
			}
		}
	}
}

void MPC_PSO_Solver::solve(Solver_ARG &h_arg, Solver_Result & rst)
{
	m_arg = h_arg;
	initParticles();

	//m_swarm.printInput();
	evalFitness();
	//m_swarm.printFitness();

	for(int i=0;i<m_pso_setting.MAX_STEP;i++)
	{
		evalFitness();
		calc_inertia_lin_dec();
		updateFitnessPersonal();
		updateFitnessBest();
		updateInput();
	}

	rst.nextState.x = m_swarm.p_best.state.x[1];
	rst.nextState.y = m_swarm.p_best.state.y[1];
	rst.nextState.psi = m_swarm.p_best.state.psi[1];
	rst.nextState.v = m_swarm.p_best.state.v[1];
	rst.nextState.cte = m_swarm.p_best.state.cte[1];
	rst.nextState.epsi = m_swarm.p_best.state.epsi[1];

	rst.input.delta = m_swarm.p_best.input.delta[0];
	rst.input.a = m_swarm.p_best.input.a[0];

	rst.guess = m_swarm.p_best.input;
}
