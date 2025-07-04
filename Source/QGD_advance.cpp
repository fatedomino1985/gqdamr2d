#include "AmrQGD.H"
#include <iostream>

using namespace amrex;

Real AmrQGD::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
    // At the beginning of step, we make the new data from previous step the
    // old data of this step.
    for (int k = 0; k < NUM_STATE_TYPE; ++k) {
        state[k].allocOldData();
        state[k].swapTimeLevels(dt);
    }

    //double mu_T = mutGas;
    
    auto dx = Geom().CellSizeArray();
	
	auto hx_i = dx[0]; 
	auto hy_j = dx[1]; 

	MultiFab& S_new = state[0].newData();// Reference to existing MultiFab
	auto const& VNew2 = S_new.arrays();//our real VNew
	//-auto VNew = VNew2;//fake. because +1 XY,Z
	
	// Create a new MultiFab with the same structure (box array, distribution, components, ghost cells)
	MultiFab S_copy(S_new.boxArray(), S_new.DistributionMap(),
                S_new.nComp(), S_new.nGrow());
	// Initialize S_copy with zeros
	S_copy.setVal(0.0);
	
	auto const& VNew = S_copy.arrays();//fake, zeros
	
	MultiFab& S_old = state[0].oldData();
	FillPatcherFill(S_old, 0, ncomp, nghost, time, State_Type, 0);
	
	auto const& VOld = S_old.arrays();

	double maxCs = 0.;
	
	int ir = 0, ira = 1, irb = 2, 
	iux = 3, iuy = 4,
	ip = 6-1, iE = 7-1, iE_in = 8-1, iE_in0 = 9-1,
	iT = 10-1, 
	iCs = 11-1,  //or auto const& Cs = S_new[0].arrays();,  
	iVa = 12-1, iVb = 13-1;
	
	//amrex::Array4<amrex::Real> Cs;?
	
	double alpha = alphaQgd;
	
	double Sc = ScQgd, Pr = PrQgd;
	double Fx = 0.0, Fy = 0.0;
	double Q = 0.0;
	double gma = gamma_a;
	double gmb = gamma_b;

	double Ra = RGas_a;
	double Rb = RGas_b;
	double cva = Ra/(gma-1);//717.5 Air
	double cvb = Rb/(gmb-1);//1495 Water
 
	double cpa = cva*gma;//Air 1004.5
	double cpb = cvb*gmb;//Kitamura: Water 4186
	//Ra = cpa*(gma - 1)/gma
	//Rb = cpb*(gmb - 1)/gmb; 
	
	double pa_inf = painf, pb_inf = pbinf;
	
		//int nc1_x, nc1_y, nc1_z;
		//amrex::Box box = S_new.boxArray()[0];
		//nc1_x = box.length(0), nc1_y = box.length(1), nc1_z = 1;//-box.length(2);
		//int nc_x = nc1_x-1, nc_y = nc1_y-1, nc_z = nc1_z-1;
		//amrex::Print() << "  nc1_x= " << nc1_x << "  nc1_y= " << nc1_y << "  nc1_z= " << nc1_z << " \n";
		
		
		//nc1_x=140, nc1_y=80,nc1_z=1;
		auto nc = Geom().Domain();
		int nc_x = nc.bigEnd(0), nc_y=nc.bigEnd(1), //nc_z=nc.bigEnd(2);
		nc_z=0;//cause '0::Assertion `i>=0 && i < dim' failed'
		int nc1_x = nc_x+1, nc1_y = nc_y+1, nc1_z = nc_z+1;
		//amrex::Print() << "  nc1_x = " << nc1_x << "  nc1_y = " << nc1_y << "  nc1_z= " << nc1_z << " \n";

	
    //%% Time step
	//% Calculation dt
	if (typeCs == 1) {
		// typeCs = 1 - according to Zlotnik's seminar, it is more correct
		// [new2].(33)
		amrex::ParallelFor(S_old, S_old.nGrowVect(), 
		[=, &maxCs] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double ro = VOld[bi](i,j,k,ir);
			double roa = VOld[bi](i,j,k,ira); double rob = VOld[bi](i,j,k,irb);
			double p = VOld[bi](i,j,k,ip);
	
			double adE_in = VOld[bi](i,j,k,iE_in) - VOld[bi](i,j,k,iE_in0);
			double rm = VOld[bi](i,j,k,ir), rma = VOld[bi](i,j,k,ira), rmb = VOld[bi](i,j,k,irb); // It's rmaf
			double cvm = (rma * cva + rmb * cvb) / rm;
			double cpm = (rma * cpa + rmb * cpb) / rm;
			double gam = cpm / cvm;
		    double sigma_a = Ra * roa / (cvm * ro);//% 2.(18)
			double sigma_b = Rb * rob / (cvm * ro);
			// p is p_+
			double E1 = sigma_a * (ro * adE_in - pa_inf) / pow(p + pa_inf, 2.); 
			E1 = E1 + sigma_b * (ro * adE_in - pb_inf) / pow(p + pb_inf, 2.); 
			E1 = 1.0 / E1; 
			E1 = sqrt(gam / ro * E1); 
			double Cs = E1;
			VNew[bi](i,j,k,iCs) = Cs;
			if (isnan(VNew[bi](i,j,k,iCs)) || isinf(VNew[bi](i,j,k,iCs)) || VNew[bi](i,j,k,iCs) <= 0 )
			{
				VNew[bi](i,j,k,iCs) = 1.0 * pow(10, -8);
				//Cs = VNew[bi](i,j,k,iCs);
				amrex::Print() << " error Cs = " << Cs << "\n";
				//amrex::Print() << "  E1= " << sigma_a * (ro * adE_in - pa_inf) / pow(p + pa_inf, 2.)
				//	<< "  E2= " << sigma_b * (ro * adE_in - pb_inf) / pow(p + pb_inf, 2.) << "\n";
				amrex::Print() << " ro = " << VOld[bi](i,j,k,ir) << "\n";
				exit(EXIT_FAILURE);
			}
			if (Cs > maxCs) 
			{
				maxCs = Cs;
			}
		
		}//++
		);
	}
	else if (typeCs == 100) {
		// H_2D считает, но не долго.Лучше самую первую Cs
		// А вот I_2D на стеках считает
		amrex::ParallelFor(S_old, [=, &maxCs] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double rma = VOld[bi](i,j,k,ira), rmb = VOld[bi](i,j,k,irb);
			double gamCs = gma * rma * Ra + gmb * rmb * Rb;
			double Tm = VOld[bi](i,j,k,iT); double rm = VOld[bi](i,j,k,ir);
			double Cs = sqrt(gamCs * Tm / rm); // H_2D(A_3D-waterCI) не считает, а I_2D(B_3D-airCI) - считает.
			VNew[bi](i,j,k,iCs) = Cs;
				
			if (isnan(VNew[bi](i,j,k,iCs)) || isinf(VNew[bi](i,j,k,iCs)) || VNew[bi](i,j,k,iCs) <= 0 )
			{
				fprintf(stderr, "\niCs100 %d,%d,%d Cs = %le "
					"v1=%le v2=%f v3=%le", i,j,k, Cs, 
					rma, rmb, rm);
			
				VNew[bi](i,j,k,iCs) = 1.0 * pow(10, -8);//test5
				//-VNew[bi](i,j,k,iCs) = VNew[bi](i-1,j-1,k-1,iCs);
				Cs = VNew[bi](i,j,k,iCs);
				exit(EXIT_FAILURE);
			}
			if (Cs > maxCs) {
				maxCs = Cs;
			}
		});
	}
	else if (typeCs == 101) {
		// almost the same as 100.
		// 7 page, after formula(19) % On Regularized Systems of Equations for Gas
		// Mixture Dynamics with New Regularizing Velocities and Diffusion Fluxes
		amrex::ParallelFor(S_old, [=, &maxCs] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			double rm = VOld[bi](i,j,k,ir), rma = VOld[bi](i,j,k,ira), rmb = VOld[bi](i,j,k,irb);
			double cvm = (rma * cva + rmb * cvb) / rm; double cpm = (rma * cpa + rmb * cpb) / rm;
			double gam = cpm / cvm; double R = (rma * Ra + rmb * Rb) / rm;
			double Tm = VOld[bi](i,j,k,iT);
			double Cs = sqrt(gam * R * Tm);
			VNew[bi](i,j,k,iCs) = Cs;
			if (Cs > maxCs) {
				maxCs = Cs;
			}
		});
	}
	
	if (isMMCs == 1) {//maxmaxCs
		amrex::ParallelFor(S_old, S_old.nGrowVect(), [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
		{
			VNew[bi](i,j,k,iCs) = maxCs;
		});
	}
	
	//double h = 1.0 / 2.0 * (hx_i + hy_j);

	//amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    amrex::ParallelFor(S_old, S_old.nGrowVect(), [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k)
    {
		//if(k==0 && j==0) std::cout << "i = " << i << "\n";//140=> 0..139, but bnd -2,-1 and 140,141
		

		double rm, rma, rmb;
		double cvm, cpm, gam;
		//double sigma_a, sigma_b;
		
		double hx4, hy4;
		//double hxy; 
		double hx, hy;

		double pm, Em;//, E_inm;
		double cs, Tm;
		double uxm, uym;
		double tau, visc, cond;
		
		double dpx, dpy, dTx, dTy;
		double duxx, duxy, duyx, duyy;
		double hWx1, hWy1;
		double hwx, hwy;
		double droauxx, droauyy, drobuxx, drobuyy;
		double Wxa2, Wxb2, Wya2, Wyb2;
		double Froax, Frobx, Frox, Froay, Froby, Froy;
		double divu, Ptau;
		double PNSxx, PNSxy, PNSyx, PNSyy;
		double Pxx, Pxy, Pyx, Pyy;
		double Fuxx, Fuxy, Fuyx, Fuyy;
		double E_in0m, droex, droey, drox, droy, qstx, qsty;
		double FEx, FEy;
		
		//double b, c, d;
		//double dts;
		double h;
		//double adE_in;
		
		if ((i > -2 && j > -2 && k > -1) 
				&& (i < nc1_x && j < nc1_y && k < nc1_z))
		{
		//%%  X fluxes
		if (j > -1)
		{
			hy4 = 4*dx[1];//hy[j + 1] + 2.0 * hy_j + hy[j - 1]; //% 4 * hy_j;
		
			rm = (VOld[bi](i,j,k,ir) + VOld[bi](i+1,j,k,ir)) * 0.5;
			// upwind
			rma = (VOld[bi](i,j,k,ira) + VOld[bi](i+1,j,k,ira)) * 0.5; rmb = (VOld[bi](i,j,k,irb) + VOld[bi](i+1,j,k,irb)) * 0.5;
			// end upwind
			uxm = (VOld[bi](i,j,k,iux) + VOld[bi](i+1,j,k,iux)) * 0.5; uym = (VOld[bi](i,j,k,iuy) + VOld[bi](i+1,j,k,iuy)) * 0.5;
			pm = (VOld[bi](i,j,k,ip) + VOld[bi](i+1,j,k,ip)) * 0.5;
			Em = (VOld[bi](i,j,k,iE) + VOld[bi](i+1,j,k,iE)) * 0.5;

			cvm = (rma * cva + rmb * cvb) / rm; cpm = (rma * cpa + rmb * cpb) / rm;
			gam = cpm / cvm;

			//++
			cs = (VNew[bi](i,j,k,iCs) + VNew[bi](i+1,j,k,iCs)) * 0.5;
			Tm = (VOld[bi](i,j,k,iT) + VOld[bi](i+1,j,k,iT)) * 0.5;

			h = sqrt(hx_i * hy_j);
			tau = alpha * h / (cs + i_t * amrex::Math::abs(pow(uxm, 2) + pow(uym, 2)));
			visc = tau * pm * Sc;// nu[2](78) or [1].(20)
			cond = tau * cpm * pm / Pr;// Злотник(78) (у него обратный)
			//cond = visc/(Pr*(gam-1));

			dpx = (VOld[bi](i+1,j,k,ip) - VOld[bi](i,j,k,ip)) / hx_i;
			dpy = (VOld[bi](i+1,j+1,k,ip) + VOld[bi](i,j+1,k,ip) - VOld[bi](i+1,j-1,k,ip) - VOld[bi](i,j-1,k,ip)) / hy4;
			
			//std::cout << "QGD_advance.cpp VOld[bi](i+1,j-1,k,ip) = "<< VOld[bi](i+1,j-1,k,ip) << " [i="<< i << ",j=" << j << ",k=" << k << "]\n";
			
			duxx = (VOld[bi](i+1,j,k,iux) - VOld[bi](i,j,k,iux)) / hx_i;
			duxy = (VOld[bi](i+1,j+1,k,iux) + VOld[bi](i,j+1,k,iux) - VOld[bi](i+1,j-1,k,iux) - VOld[bi](i,j-1,k,iux)) / hy4;

			duyx = (VOld[bi](i+1,j,k,iuy) - VOld[bi](i,j,k,iuy)) / hx_i;
			duyy = (VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i,j+1,k,iuy) - VOld[bi](i+1,j-1,k,iuy) - VOld[bi](i,j-1,k,iuy)) / hy4;

			hWx1 = tau * (rm * (uxm * duxx + uym * duxy) + dpx - rm * Fx);
			hWy1 = tau * (rm * (uxm * duyx + uym * duyy) + dpy - rm * Fy);

			hwx = hWx1 / rm;

			droauxx = (VOld[bi](i+1,j,k,ira) * VOld[bi](i+1,j,k,iux) - VOld[bi](i,j,k,ira) * VOld[bi](i,j,k,iux)) / hx_i;
			droauyy = (VOld[bi](i+1,j+1,k,ira) * VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i,j+1,k,ira) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i+1,j-1,k,ira) * VOld[bi](i+1,j-1,k,iuy) - VOld[bi](i,j-1,k,ira) * VOld[bi](i,j-1,k,iuy)) / hy4;

			drobuxx = (VOld[bi](i+1,j,k,irb) * VOld[bi](i+1,j,k,iux) - VOld[bi](i,j,k,irb) * VOld[bi](i,j,k,iux)) / hx_i;
			drobuyy = (VOld[bi](i+1,j+1,k,irb) * VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i,j+1,k,irb) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i+1,j-1,k,irb) * VOld[bi](i+1,j-1,k,iuy) - VOld[bi](i,j-1,k,irb) * VOld[bi](i,j-1,k,iuy)) / hy4;

			Wxa2 = tau * (droauxx + droauyy) * uxm + rma * hwx;
			Wxb2 = tau * (drobuxx + drobuyy) * uxm + rmb * hwx;

			Froax = rma * uxm - Wxa2; Frobx = rmb * uxm - Wxb2; Frox = Froax + Frobx;

			divu = duxx + duyy;
			Ptau = tau * ((uxm * dpx + uym * dpy) + rm * pow(cs, 2) * divu - pow(cs, 2) / (gam * cvm * Tm) * Q);//[new2].(51)

			PNSxx = 2.0 * visc * duxx - 2.0 / 3.0 * visc * divu;
			PNSyx = visc * (duyx + duxy) + 0.0; 

			Pxx = PNSxx + uxm * hWx1 + Ptau;
			Pxy = PNSyx + uxm * hWy1;

			Fuxx = pm + uxm * Frox - Pxx;
			Fuyx = uym * Frox - Pxy;

			dTx = (VOld[bi](i+1,j,k,iT) - VOld[bi](i,j,k,iT)) / hx_i;

			E_in0m = (VOld[bi](i,j,k,iE_in0) + VOld[bi](i+1,j,k,iE_in0)) * 0.5;
			droex = (VOld[bi](i+1,j,k,ir) * VOld[bi](i+1,j,k,iE_in) - VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iE_in)) / hx_i;
			droey = (VOld[bi](i+1,j+1,k,ir) * VOld[bi](i+1,j+1,k,iE_in) + VOld[bi](i,j+1,k,ir) * VOld[bi](i,j+1,k,iE_in) - VOld[bi](i+1,j-1,k,ir) * VOld[bi](i+1,j-1,k,iE_in) - VOld[bi](i,j-1,k,ir) * VOld[bi](i,j-1,k,iE_in)) / hy4;

			drox = (VOld[bi](i+1,j,k,ir) - VOld[bi](i,j,k,ir)) / hx_i;
			droy = (VOld[bi](i+1,j+1,k,ir) + VOld[bi](i,j+1,k,ir) - VOld[bi](i+1,j-1,k,ir) - VOld[bi](i,j-1,k,ir)) / hy4;

			qstx = tau * uxm * (uxm * (droex - (gam * cvm * Tm + E_in0m) * drox) +
				uym * (droey - (gam * cvm * Tm + E_in0m) * droy) -
				Q);

			FEx = (Em + pm) * Frox / rm - cond * dTx - qstx - (Pxx * uxm + Pxy * uym);

			hy = hy_j;

			VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) - Froax * hy; 
			VNew[bi](i+1,j,k,ira) = VNew[bi](i+1,j,k,ira) + Froax * hy;
			VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) - Frobx * hy;
			VNew[bi](i+1,j,k,irb) = VNew[bi](i+1,j,k,irb) + Frobx * hy;
			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) - Fuxx * hy;
			VNew[bi](i+1,j,k,iux) = VNew[bi](i+1,j,k,iux) + Fuxx * hy;
			VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) - Fuyx * hy;
			VNew[bi](i+1,j,k,iuy) = VNew[bi](i+1,j,k,iuy) + Fuyx * hy;
			VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) - FEx * hy;
			VNew[bi](i+1,j,k,iE) = VNew[bi](i+1,j,k,iE) + FEx * hy;
		}
		//%%  Y fluxes
		if (i > -1)
		{
			hx4 = 4*dx[0];//hx[i + 1] + 2.0 * hx_i + hx[i - 1];
			
			rm = (VOld[bi](i,j,k,ir) + VOld[bi](i,j+1,k,ir)) * 0.5;
			// upwind
			rma = (VOld[bi](i,j,k,ira) + VOld[bi](i,j+1,k,ira)) * 0.5; rmb = (VOld[bi](i,j,k,irb) + VOld[bi](i,j+1,k,irb)) * 0.5;
			// end upwind
			uxm = (VOld[bi](i,j,k,iux) + VOld[bi](i,j+1,k,iux)) * 0.5; uym = (VOld[bi](i,j,k,iuy) + VOld[bi](i,j+1,k,iuy)) * 0.5; 
			pm = (VOld[bi](i,j,k,ip) + VOld[bi](i,j+1,k,ip)) * 0.5;
			Em = (VOld[bi](i,j,k,iE) + VOld[bi](i,j+1,k,iE)) * 0.5;

			cvm = (rma * cva + rmb * cvb) / rm; cpm = (rma * cpa + rmb * cpb) / rm;
			gam = cpm / cvm;

			cs = (VNew[bi](i,j,k,iCs) + VNew[bi](i,j+1,k,iCs)) * 0.5;
			Tm = (VOld[bi](i,j,k,iT) + VOld[bi](i,j+1,k,iT)) * 0.5;

			h = sqrt(hx_i * hy_j);
			tau = alpha * h / (cs + i_t * amrex::Math::abs(pow(uxm, 2) + pow(uym, 2)));
			visc = tau * pm * Sc; // nu[2](78) or [1](20)
			cond = tau * cpm * pm / Pr; // Злотник(78)
			// cond = visc / (Pr * (gam - 1));

			dpx = (VOld[bi](i+1,j+1,k,ip) + VOld[bi](i+1,j,k,ip) - VOld[bi](i-1,j+1,k,ip) - VOld[bi](i-1,j,k,ip)) / hx4;
			dpy = (VOld[bi](i,j+1,k,ip) - VOld[bi](i,j,k,ip)) / hy_j;

			duxx = (VOld[bi](i+1,j+1,k,iux) + VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j+1,k,iux) - VOld[bi](i-1,j,k,iux)) / hx4;
			duxy = (VOld[bi](i,j+1,k,iux) - VOld[bi](i,j,k,iux)) / hy_j;

			duyx = (VOld[bi](i+1,j+1,k,iuy) + VOld[bi](i+1,j,k,iuy) - VOld[bi](i-1,j+1,k,iuy) - VOld[bi](i-1,j,k,iuy)) / hx4;
			duyy = (VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j,k,iuy)) / hy_j;

			hWx1 = tau * (rm * (uxm * duxx + uym * duxy) + dpx - rm * Fx);
			hWy1 = tau * (rm * (uxm * duyx + uym * duyy) + dpy - rm * Fy);

			hwy = hWy1 / rm;

			droauxx = (VOld[bi](i+1,j+1,k,ira) * VOld[bi](i+1,j+1,k,iux) + VOld[bi](i+1,j,k,ira) * VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j+1,k,ira) * VOld[bi](i-1,j+1,k,iux) - VOld[bi](i-1,j,k,ira) * VOld[bi](i-1,j,k,iux)) / hx4;
			droauyy = (VOld[bi](i,j+1,k,ira) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j,k,ira) * VOld[bi](i,j,k,iuy)) / hy_j;

			drobuxx = (VOld[bi](i+1,j+1,k,irb) * VOld[bi](i+1,j+1,k,iux) + VOld[bi](i+1,j,k,irb) * VOld[bi](i+1,j,k,iux) - VOld[bi](i-1,j+1,k,irb) * VOld[bi](i-1,j+1,k,iux) - VOld[bi](i-1,j,k,irb) * VOld[bi](i-1,j,k,iux)) / hx4;
			drobuyy = (VOld[bi](i,j+1,k,irb) * VOld[bi](i,j+1,k,iuy) - VOld[bi](i,j,k,irb) * VOld[bi](i,j,k,iuy)) / hy_j;

			Wya2 = tau * (droauxx + droauyy) * uym + rma * hwy;
			Wyb2 = tau * (drobuxx + drobuyy) * uym + rmb * hwy;

			Froay = rma * uym - Wya2; Froby = rmb * uym - Wyb2; Froy = Froay + Froby;

			divu = duxx + duyy;
			Ptau = tau * ((uxm * dpx + uym * dpy) + rm * pow(cs, 2) * divu - pow(cs, 2) / (gam * cvm * Tm) * Q);

			PNSxy = visc * (duxy + duyx) + 0; 
			PNSyy = 2.0 * visc * duyy - 2.0 / 3.0 * visc * divu;  

			Pyx = PNSxy + uym * hWx1; 
			Pyy = PNSyy + uym * hWy1 + Ptau; 

			Fuxy = uxm * Froy - Pyx; Fuyy = pm + uym * Froy - Pyy;

			dTy = (VOld[bi](i,j+1,k,iT) - VOld[bi](i,j,k,iT)) / hy_j;
			E_in0m = (VOld[bi](i,j,k,iE_in0) + VOld[bi](i,j+1,k,iE_in0)) * 0.5;
			droex = (VOld[bi](i+1,j+1,k,ir) * VOld[bi](i+1,j+1,k,iE_in) + VOld[bi](i+1,j,k,ir) * VOld[bi](i+1,j,k,iE_in) - VOld[bi](i-1,j+1,k,ir) * VOld[bi](i-1,j+1,k,iE_in) - VOld[bi](i-1,j,k,ir) * VOld[bi](i-1,j,k,iE_in)) / hx4;
			droey = (VOld[bi](i,j+1,k,ir) * VOld[bi](i,j+1,k,iE_in) - VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iE_in)) / hy_j;

			drox = (VOld[bi](i+1,j+1,k,ir) + VOld[bi](i+1,j,k,ir) - VOld[bi](i-1,j+1,k,ir) - VOld[bi](i-1,j,k,ir)) / hx4;
			droy = (VOld[bi](i,j+1,k,ir) - VOld[bi](i,j,k,ir)) / hy_j;

			qsty = tau * uym * (uxm * (droex - (gam * cvm * Tm + E_in0m) * drox) +
				uym * (droey - (gam * cvm * Tm + E_in0m) * droy) -
				Q);

			FEy = (Em + pm) * Froy / rm - cond * dTy - qsty - (Pyx * uxm + Pyy * uym);

			hx = hx_i;

			VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) - Froay * hx; 
			
			VNew[bi](i,j+1,k,ira) = VNew[bi](i,j+1,k,ira) + Froay * hx;
			VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) - Froby * hx;
			VNew[bi](i,j+1,k,irb) = VNew[bi](i,j+1,k,irb) + Froby * hx;
			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) - Fuxy * hx;
			VNew[bi](i,j+1,k,iux) = VNew[bi](i,j+1,k,iux) + Fuxy * hx;
			VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) - Fuyy * hx;
			VNew[bi](i,j+1,k,iuy) = VNew[bi](i,j+1,k,iuy) + Fuyy * hx;
			VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) - FEy * hx;
			VNew[bi](i,j+1,k,iE) = VNew[bi](i,j+1,k,iE) + FEy * hx;
		}
		
		}
		
		// New variables. Saved
		VNew[bi](i,j,k,iE_in0) = VOld[bi](i,j,k,iE_in0);
		});
		amrex::ParallelFor(S_old, [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k){
		// New variables
		
		double sigma_a, sigma_b;
		
		double cvm, cpm, gam;
		
		double b, c, d;
		double dts;
		//double h;
		double adE_in;
		
		dts = dt / (hx_i * hy_j);
		//h = 1.0/2. * (hx_i + hy_j);
		
		//if (i > nil && i <= nc_x && j > nil && j <= nc_y && k > nil && k <= nc_z)
		{
		
		
			VNew[bi](i,j,k,ira) = VNew[bi](i,j,k,ira) * dts + VOld[bi](i,j,k,ira); VNew[bi](i,j,k,irb) = VNew[bi](i,j,k,irb) * dts + VOld[bi](i,j,k,irb);
			
			if (VNew[bi](i,j,k,ira) < 0.0)
				VNew[bi](i,j,k,ira) = 1.0 * pow(10, -8.);
			if (VNew[bi](i,j,k,irb) < 0.0)
				VNew[bi](i,j,k,irb) = 1.0 * pow(10, -8.);

			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) * dts + VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iux); // +dt.*RSux;
			VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) * dts + VOld[bi](i,j,k,ir) * VOld[bi](i,j,k,iuy); // +dt.*RSuy;
			VNew[bi](i,j,k,iE) = VNew[bi](i,j,k,iE) * dts + VOld[bi](i,j,k,iE); // +dt * RSE;

			VNew[bi](i,j,k,ir) = VNew[bi](i,j,k,ira) + VNew[bi](i,j,k,irb);
			VNew[bi](i,j,k,iux) = VNew[bi](i,j,k,iux) / VNew[bi](i,j,k,ir); 
			VNew[bi](i,j,k,iuy) = VNew[bi](i,j,k,iuy) / VNew[bi](i,j,k,ir);
			
			VNew[bi](i,j,k,iE_in) = (VNew[bi](i,j,k,iE) - 0.5 * VNew[bi](i,j,k,ir) * (VNew[bi](i,j,k,iux) * VNew[bi](i,j,k,iux) + VNew[bi](i,j,k,iuy) * VNew[bi](i,j,k,iuy))) / VNew[bi](i,j,k,ir);
			adE_in = VNew[bi](i,j,k,iE_in) - VNew[bi](i,j,k,iE_in0);

			cvm = (VNew[bi](i,j,k,ira) * cva + VNew[bi](i,j,k,irb) * cvb) / VNew[bi](i,j,k,ir);
			cpm = (VNew[bi](i,j,k,ira) * cpa + VNew[bi](i,j,k,irb) * cpb) / VNew[bi](i,j,k,ir);
			gam = cpm / cvm;

			sigma_a = Ra * VNew[bi](i,j,k,ira) / (cvm * VNew[bi](i,j,k,ir));//2.(18)
			sigma_b = Rb * VNew[bi](i,j,k,irb) / (cvm * VNew[bi](i,j,k,ir));
			b = sigma_a * (VNew[bi](i,j,k,ir) * adE_in - pa_inf) - pa_inf + sigma_b * (VNew[bi](i,j,k,ir) * adE_in - pb_inf) - pb_inf;
			c = (sigma_a * pb_inf + sigma_b * pa_inf) * VNew[bi](i,j,k,ir) * adE_in - gam * pa_inf * pb_inf;
			d = pow(b, 2.) + 4.0 * c;
			if (d <= 0) {
				fprintf(stderr, "\nError in d: i=%d j=%d k=%d, bi=%d, d=%g, step=%d, time=%.8le"
					, i, j, k, bi, d, int(time/dt), time);
				exit(EXIT_FAILURE);
			}
			VNew[bi](i,j,k,ip) = 0.5 * (b + sqrt(d));
			
			if ((VNew[bi](i,j,k,ip) < 0.0) || isnan(VNew[bi](i,j,k,ip))) {//еще проверку на complex number - Комплексное число
				fprintf(stderr, "\nError in p: i=%d j=%d k=%d, bi=%d, p=%g, "
					"step = %d, time = %.8le", 
					i, j, k, bi, VNew[bi](i,j,k,ip), 
					int(time/dt), time);//--ncycle, time);// --iteration, time);
				fprintf(stderr, "\nOther values: ro=%le, E=%le, E_in=%le, ux=%le, uy=%le \n"
					, VNew[bi](i,j,k,ir), VNew[bi](i,j,k,iE), VNew[bi](i,j,k,iE_in), VNew[bi](i,j,k,iux), VNew[bi](i,j,k,iuy));
				double xbeg = -15*pow(10,-3);//A_3D
				fprintf(stderr, "\nxy in (%le, %le)", xbeg+i*hx_i, j*hy_j);
				exit(EXIT_FAILURE);
			}
			
			VNew[bi](i,j,k,iT) = Ra * VNew[bi](i,j,k,ira) / (VNew[bi](i,j,k,ip) + pa_inf) + Rb * VNew[bi](i,j,k,irb) / (VNew[bi](i,j,k,ip) + pb_inf);//[new2].(42)
			VNew[bi](i,j,k,iT) = 1.0 / VNew[bi](i,j,k,iT);
			VNew[bi](i,j,k,iVa) = Ra * VNew[bi](i,j,k,ira) * VNew[bi](i,j,k,iT) / (VNew[bi](i,j,k,ip) + pa_inf); //[2].(48) or [new2].(41)
			VNew[bi](i,j,k,iVb) = Rb * VNew[bi](i,j,k,irb) * VNew[bi](i,j,k,iT) / (VNew[bi](i,j,k,ip) + pb_inf);
			
			
			
			for (int ia=0; ia<ncomp; ia++)
					VNew2[bi](i,j,k,ia) = VNew[bi](i,j,k,ia);
		}
		
    }//++
	);
	
	
	
	//exit(EXIT_FAILURE);
	/*
    Real maxval = S_new.max(0);
    Real minval = S_new.min(0);
    amrex::Print() << "min/max rho = " << minval << "/" << maxval;
    maxval = S_new.max(4);
    minval = S_new.min(4);
    amrex::Print() << "  min/max Sc number = " << minval << "/" << maxval << "\n";
	*/
	
    return dt;
}
