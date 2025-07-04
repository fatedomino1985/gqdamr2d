#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

//
#include "AmrQGD.H" 

using namespace amrex;

struct QGDBCFill 
{
        AMREX_GPU_DEVICE
        void operator() (const IntVect& iv, Array4<Real> const& dest,
                         const int /*dcomp*/, const int /*numcomp*/,
                         GeometryData const& geom, const Real /*time*/,
                         const BCRec* /*bcr*/, const int /*bcomp*/,
                         const int /*orig_comp*/) const
        {
            const int ilo = geom.Domain().smallEnd(0);
            const int ihi = geom.Domain().bigEnd(0);
            const int jlo = geom.Domain().smallEnd(1);
            const int jhi = geom.Domain().bigEnd(1);
            //const auto problo = geom.ProbLo();//data();
            const auto [i,j,k] = iv.dim3();

            // const auto problo = data1.ProbLo();
            //const auto dx = geom.CellSize();
			
			
			int ir = 0, ira = 1, irb = 2, 
			iux = 3, iuy = 4,
			ip = 5, //iE = 6, 
			iE_in = 7, iE_in0 = 8,
			iT = 9, 
			iCs = 10,
			iVa = 11, iVb = 12;
			//{
				//%%3.5 Shock/Water-Column interaction [3]Китамура
				//double Runiv = AmrQGD::Runiv;
				double gma = AmrQGD::gamma_a, gmb = AmrQGD::gamma_b;
			
				double Ra = AmrQGD::RGas_a;
				double Rb = AmrQGD::RGas_b;
				double cva = Ra / (gma - 1); // 717.5 Air
				double cvb = Rb / (gmb - 1); // 1495 Water
				//double pL = AmrQGD::pL;
				double pR = AmrQGD::pR;
				double pa_inf = AmrQGD::painf,
				pb_inf = AmrQGD::pbinf;
				double T1 = AmrQGD::T1;
				//double T2 = AmrQGD::T2;
				double u1x = AmrQGD::u1x,
				//u2x = AmrQGD::u2x,
				u1y = AmrQGD::u1y;
				//double u2y = AmrQGD::u2y;
				//double E_in01 = AmrQGD::E_in01, E_in02 = AmrQGD::E_in02;//0.0; //y_1 = 0; *y_2 = 0.0;
				double esp = AmrQGD::esp;
				//double x_bub = AmrQGD::x_bub;
				//double r_bub = AmrQGD::r_bub;
			//}
			
            //surfaces
            if (i < ilo) {// Left bound
                dest(i,j,k,8) = dest(ilo,j,k,iE_in0); 			 										//E_in0
                dest(i,j,k,10) = dest(ilo,j,k,iCs); 			 										//Cs
                dest(i,j,k,11) = 1 - esp; 						 										//Va
                dest(i,j,k,12) = 1 - dest(i,j,k,iVa); 			 										//Vb
				dest(i,j,k,3) = u1x; 							 										//ux
                dest(i,j,k,4) = u1y; 							 										//uy
				dest(i,j,k,9) = T1; 							 										//T
				dest(i,j,k,5) = dest(ilo,j,k,ip); 				 										//p
				dest(i,j,k,1) = dest(i,j,k,iVa) * (dest(i,j,k,ip) + pa_inf) / (Ra * dest(i,j,k,iT));	//rho_a
                dest(i,j,k,2) = dest(i,j,k,iVb) * (dest(i,j,k,ip) + pb_inf) / (Rb * dest(i,j,k,iT));	//rho_b
				dest(i,j,k,0) = dest(i,j,k,ira)+dest(i,j,k,irb); 										//rho
				double E_in = cva*dest(i,j,k,iT) + dest(i,j,k,iT) * Ra * pa_inf / (dest(i,j,k,ip) + pa_inf) + dest(i,j,k,iE_in0);// [2](9)
                E_in = dest(i,j,k,ira)*E_in + dest(i,j,k,irb)*(cvb*dest(i,j,k,iT) + dest(i,j,k,iT) * Rb * pb_inf / (dest(i,j,k,ip) + pb_inf) + dest(i,j,k,iE_in0));
				dest(i,j,k,7) = E_in/dest(i,j,k,ir); 			 										//E_in
				double uu = (dest(i,j,k,iux)*dest(i,j,k,iux)+dest(i,j,k,iuy)*dest(i,j,k,iuy));
                dest(i,j,k,6) = dest(i,j,k,iE_in)*dest(i,j,k,ir)+0.5*dest(i,j,k,ir)*uu; 				//E
                
            }
            if (i > ihi) {// Right bound
                dest(i,j,k,0) = dest(ihi,j,k,ir);
                dest(i,j,k,1) = dest(ihi,j,k,1);
                dest(i,j,k,2) = dest(ihi,j,k,2);
                dest(i,j,k,3) = dest(ihi,j,k,iux);
                dest(i,j,k,4) = dest(ihi,j,k,iuy);
                dest(i,j,k,5) = pR;
				dest(i,j,k,7) = dest(ihi,j,k,iE_in);
				double uu = (dest(i,j,k,iux)*dest(i,j,k,iux)+dest(i,j,k,iuy)*dest(i,j,k,iuy));
                dest(i,j,k,6) = dest(i,j,k,ir)*(uu)*0.5+dest(i,j,k,ir)*dest(i,j,k,iE_in);
                dest(i,j,k,8) = dest(ihi,j,k,8);
                dest(i,j,k,9) = dest(ihi,j,k,9);
                dest(i,j,k,10) = dest(ihi,j,k,10);
                dest(i,j,k,11) = dest(ihi,j,k,11);
                dest(i,j,k,12) = dest(ihi,j,k,12);
            }
            if (j < jlo) {// Front bound
                dest(i,j,k,0) = dest(i,jlo,k,0);
                dest(i,j,k,1) = dest(i,jlo,k,1);
                dest(i,j,k,2) = dest(i,jlo,k,2);
                dest(i,j,k,3) = dest(i,jlo,k,3);
                dest(i,j,k,4) = dest(i,jlo,k,iuy);//0;//dest(i,jlo,k,iuy);//-dest(i,jlo,k,iuy); //- if y from 0, not -15e-03
                dest(i,j,k,5) = dest(i,jlo,k,5);
                dest(i,j,k,6) = dest(i,jlo,k,6);
                dest(i,j,k,7) = dest(i,jlo,k,7);
                dest(i,j,k,8) = dest(i,jlo,k,8);
                dest(i,j,k,9) = dest(i,jlo,k,9);
                dest(i,j,k,10) = dest(i,jlo,k,10);
                dest(i,j,k,11) = dest(i,jlo,k,11);
                dest(i,j,k,12) = dest(i,jlo,k,12);
            }
            if (j > jhi) {// Rear bound
                dest(i,j,k,0) = dest(i,jhi,k,0);
                dest(i,j,k,1) = dest(i,jhi,k,1);
                dest(i,j,k,2) = dest(i,jhi,k,2);
                dest(i,j,k,3) = dest(i,jhi,k,3);
                dest(i,j,k,4) = dest(i,jhi,k,4);
                dest(i,j,k,5) = dest(i,jhi,k,5);
                dest(i,j,k,6) = dest(i,jhi,k,6);
                dest(i,j,k,7) = dest(i,jhi,k,7);
                dest(i,j,k,8) = dest(i,jhi,k,8);
                dest(i,j,k,9) = dest(i,jhi,k,9);
                dest(i,j,k,10) = dest(i,jhi,k,10);
                dest(i,j,k,11) = dest(i,jhi,k,11);
                dest(i,j,k,12) = dest(i,jhi,k,12);
            }
			//-------------edges--------------//
			
            if(i < ilo && j < jlo) {//left mirror into, without j
                dest(i,j,k,0) = dest(ilo,jlo,k,0);
                dest(i,j,k,1) = dest(ilo,jlo,k,1);
                dest(i,j,k,2) = dest(ilo,jlo,k,2);
                dest(i,j,k,3) = dest(ilo,jlo,k,3);
                dest(i,j,k,4) = dest(ilo,jlo,k,4);//0;//dest(ilo,jlo,k,4);
                dest(i,j,k,5) = dest(ilo,jlo,k,5);
                dest(i,j,k,6) = dest(ilo,jlo,k,6);
                dest(i,j,k,7) = dest(ilo,jlo,k,7);
                dest(i,j,k,8) = dest(ilo,jlo,k,8);
                dest(i,j,k,9) = dest(ilo,jlo,k,9);   
                dest(i,j,k,10) = dest(ilo,jlo,k,10);
                dest(i,j,k,11) = dest(ilo,jlo,k,11);
                dest(i,j,k,12) = dest(ilo,jlo,k,12);    
            }
            if(i < ilo && j > jhi) {
                dest(i,j,k,0) = dest(ilo,jhi,k,0);
                dest(i,j,k,1) = dest(ilo,jhi,k,1);
                dest(i,j,k,2) = dest(ilo,jhi,k,2);
                dest(i,j,k,3) = dest(ilo,jhi,k,3);
                dest(i,j,k,4) = dest(ilo,jhi,k,4);//0;//dest(ilo,jhi,k,4);
                dest(i,j,k,5) = dest(ilo,jhi,k,5);
                dest(i,j,k,6) = dest(ilo,jhi,k,6);
                dest(i,j,k,7) = dest(ilo,jhi,k,7);
                dest(i,j,k,8) = dest(ilo,jhi,k,8);
                dest(i,j,k,9) = dest(ilo,jhi,k,9);  
                dest(i,j,k,10) = dest(ilo,jhi,k,10);
                dest(i,j,k,11) = dest(ilo,jhi,k,11);
                dest(i,j,k,12) = dest(ilo,jhi,k,12);         
            }                        
                  
            if(i > ihi && j < jlo) {
                dest(i,j,k,0) = dest(ihi,jlo,k,0);
                dest(i,j,k,1) = dest(ihi,jlo,k,1);
                dest(i,j,k,2) = dest(ihi,jlo,k,2);
                dest(i,j,k,3) = dest(ihi,jlo,k,3);
                dest(i,j,k,4) = dest(ihi,jlo,k,4);//0;//dest(ihi,jlo,k,4);
                dest(i,j,k,5) = dest(ihi,jlo,k,5);
                dest(i,j,k,6) = dest(ihi,jlo,k,6);
                dest(i,j,k,7) = dest(ihi,jlo,k,7);
                dest(i,j,k,8) = dest(ihi,jlo,k,8);
                dest(i,j,k,9) = dest(ihi,jlo,k,9);  
                dest(i,j,k,10) = dest(ihi,jlo,k,10);
                dest(i,j,k,11) = dest(ihi,jlo,k,11);
                dest(i,j,k,12) = dest(ihi,jlo,k,12);      
            }
            if(i > ihi && j > jhi) {
                dest(i,j,k,0) = dest(ihi,jhi,k,0);
                dest(i,j,k,1) = dest(ihi,jhi,k,1);
                dest(i,j,k,2) = dest(ihi,jhi,k,2);
                dest(i,j,k,3) = dest(ihi,jhi,k,3);
                dest(i,j,k,4) = dest(ihi,jhi,k,4);//0;//dest(ihi,jhi,k,4);
                dest(i,j,k,5) = dest(ihi,jhi,k,5);
                dest(i,j,k,6) = dest(ihi,jhi,k,6);
                dest(i,j,k,7) = dest(ihi,jhi,k,7);
                dest(i,j,k,8) = dest(ihi,jhi,k,8);
                dest(i,j,k,9) = dest(ihi,jhi,k,9);  
                dest(i,j,k,10) = dest(ihi,jhi,k,10);
                dest(i,j,k,11) = dest(ihi,jhi,k,11);
                dest(i,j,k,12) = dest(ihi,jhi,k,12);         
            }
        }
};

void bcfill (Box const& bx, FArrayBox& data,
             int dcomp, int numcomp,
             Geometry const& geom, Real time,
             const Vector<BCRec>& bcr, int bcomp,int scomp)
{
    GpuBndryFuncFab<QGDBCFill> gpu_bndry_func(QGDBCFill{});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}