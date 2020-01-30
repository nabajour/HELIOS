// =================================================================================
// This file contains all the device functions and CUDA kernels.
// Copyright (C) 2018 Matej Malik
// =================================================================================
// This file is part of HELIOS.
//
//     HELIOS is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     HELIOS is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     HELIOS directory under <license.txt>. If not, see
//     <http://www.gnu.org/licenses/>.
// =================================================================================

#include<stdio.h>

// switch between double and single precision (currently, single prec. provides no speed-up and thus appears to be useless)
/***
#define USE_SINGLE
***/
#ifdef USE_SINGLE
typedef float utype;
#else
typedef double utype;
#endif


// physical constants
const utype PI = 3.141592653589793;
const utype HCONST = 6.62607004e-27;
const utype CSPEED = 29979245800.0;
const utype KBOLTZMANN = 1.38064852e-16;
const utype STEFANBOLTZMANN = 5.6703669999999995e-5; // yes, it needs to have this exact value to be consistent with astropy


// calculates the normal distribution
__device__ utype norm_pdf(
        utype x, 
        utype mu, 
        utype s
        ){

    return 1.0 / (s * sqrt(2.0 * PI)) * exp(-((x-mu)*(x-mu))/(2.0*(s*s)));
}


// computes the blackbody function for given wavelength & temperature
__device__ utype planck_func(utype lamda, utype T){
    
    utype num = 2.0 * HCONST * CSPEED * CSPEED;
    utype denom = pow(1.0*lamda, 5.0) * (exp(HCONST*CSPEED/(lamda*KBOLTZMANN*T)) - 1.0);
    
    return num / denom;
}


// atomicAdd for single precision
__device__ float atomicAdd_single(float* address, float value){

    float old = value;  
    float ret=atomicExch(address, 0.0f);
    float new_old=ret+old;

    while ((old = atomicExch(address, new_old))!=0.0f)
    {
        new_old = atomicExch(address, 0.0f);
        new_old += old;
    }
    return ret;
}


// atomicAdd for double precision
__device__ double atomicAdd_double(double* address, double val) {
    
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


// calculates analytically the integral of the planck function
__device__ utype analyt_planck(
        int 	n, 
        utype 	y1, 
        utype 	y2
){

    utype dn=n;

    return exp(-dn*y2) * ((y2*y2*y2)/dn + 3.0*(y2*y2)/(dn*dn) + 6.0*y2/(dn*dn*dn) + 6.0/(dn*dn*dn*dn))
            - exp(-dn*y1) * ((y1*y1*y1)/dn + 3.0*(y1*y1)/(dn*dn) + 6.0*y1/(dn*dn*dn) + 6.0/(dn*dn*dn*dn));
}


//  calculates the transmission function
__device__ utype trans_func(
        utype epsi, 
        utype delta_tau, 
        utype w0, 
        utype g0
){

    return exp(-1.0/epsi*sqrt((1.0-w0*g0)*(1.0-w0))*delta_tau);
}


// calculates the G+ function
__device__ utype G_plus_func(
        utype w0, 
        utype g0, 
        utype epsi, 
        utype mu_star){

    utype num = (1.0 - w0) * (1.0 - w0 * g0) - 1.0;

    utype denom = pow(mu_star,-2.0) - pow(epsi,-2.0) * (1.0 - w0) * (1.0 - w0 * g0);

    utype second_term = 1.0/epsi + 1.0/(mu_star * (1.0 - w0 * g0));
    
    utype third_term = w0 * g0 * mu_star / (1.0 - w0 * g0);
            
    utype bracket = num/denom * second_term + third_term;

    utype result =  0.5 * bracket;

    return result;
}


// calculates the G- function
__device__ utype G_minus_func(
        utype w0, 
        utype g0, 
        utype epsi, 
        utype mu_star){

    utype num = (1.0 - w0) * (1.0 - w0 * g0) - 1.0;

    utype denom = pow(mu_star,-2.0) - pow(epsi,-2.0) * (1.0 - w0) * (1.0 - w0 * g0);

    utype second_term = 1.0/epsi - 1.0/(mu_star * (1.0 - w0 * g0));
    
    utype third_term = w0 * g0 * mu_star / (1.0 - w0 * g0);
            
    utype bracket = num/denom * second_term - third_term;

    utype result =  0.5 * bracket;

    return result;
}


// limiting the values of the G_plus and G_minus coefficients to less than 1e8. 
// This value is somewhat ad hoc from visual analysis. To justify, results are quite insensitive to this value.
__device__ utype G_limiter(utype G){
    
    if(abs(G) < 1e8){
        return G;	
    }
    else{
        return 0;
    }
}


// calculates the power operation with a foor loop -- is allegedly faster than the implemented pow() function
__device__ utype power_int(utype x, int i){

    utype result = 1.0;
    int j = 1;
    
    while(j<=i){
        result *= x;
        j++;
    }
    return result;
}


// calculates the single scattering albedo w0
__device__ utype single_scat_alb(
        utype scat_cross, 
        utype opac_abs, 
        utype meanmolmass
){

    return scat_cross / (scat_cross + opac_abs*meanmolmass);
}


// fitting function for the E parameter according to "Heng, Malik & Kitzmann 2018
__device__ utype E_parameter(
        utype w0, 
        utype g0
){
    utype E;
    
    if (w0 > 0 && g0 >= 0){
        
        E = max(1.0, 1.225 - 0.1582*g0 - 0.1777*w0 - 0.07465*pow(1.0*g0, 2.0) + 0.2351*w0*g0 - 0.05582*pow(w0, 2.0));
    }
    else{
        E = 1.0;
    }
    return E;
}


// calculates the two-stream coupling coefficient Zeta_minus with the scattering coefficient E
__device__ utype zeta_minus(
        utype w0, 
        utype g0,
        int scat_corr
){
    utype E;

    if(scat_corr==1){
        E = E_parameter(w0, g0);
    }
    else{
        E = 1.0;
    }
    return 0.5 * (1.0 - sqrt((E - w0)/(E*(1.0 - w0*g0))) );
}


// calculates the two-stream coupling coefficient Zeta_plus with the scattering coefficient E
__device__ utype zeta_plus(
        utype w0, 
        utype g0,
        int scat_corr
){
    utype E;

    if(scat_corr==1){
        E = E_parameter(w0, g0);
    }
    else{
        E = 1.0;
    }
    return 0.5 * (1.0 + sqrt((E - w0)/(E*(1.0 - w0*g0))) );
}


// calculates the derivative of the Planck function regarding temperature
__device__ utype dB_dT(
        utype lambda, 
        utype T
){

    utype D = 2.0 * HCONST * power_int(CSPEED, 3) * HCONST / (power_int(lambda, 6) * KBOLTZMANN * (T*T));

    utype num =  exp(HCONST * CSPEED / (lambda * KBOLTZMANN * T));

    utype denom = (exp( HCONST * CSPEED / (lambda * KBOLTZMANN * T)) - 1.0) * (exp( HCONST * CSPEED / (lambda * KBOLTZMANN * T)) - 1.0);

    utype result = D * num / denom ;

    return result;
}


// calculates the integral of the Planck derivative over a wavelength interval
__device__ utype integrated_dB_dT(
        utype* kw, 
        utype* ky, 
        int 	ny, 
        utype 	lambda_bot, 
        utype 	lambda_top, 
        utype 	T
){

    utype result = 0;

    for (int y=0;y<ny;y++){
        utype x = (ky[y]-0.5)*2.0;
        utype arg = (lambda_top-lambda_bot)/2.0 * x + (lambda_top+lambda_bot)/2.0;
        result += (lambda_top-lambda_bot)/2.0 * kw[y]* dB_dT(arg,T);
    }
    return result;
}


// calculates the exponential integral of 1st kind
__device__ utype expint1(
        utype x
){

    utype a[] = {-0.57721566,0.99999193,-0.24991055,0.05519968,-0.00976004,0.00107857};
    utype b[] = {1,8.5733287401,18.059016973,8.6347608925,0.2677737343};
    utype c[] = {1,9.5733223454,25.6329561486,21.0996530827,3.9584969228};

    utype result;

    if(x < 1){
        result = -log(x);
        for(int j=0;j<6;j++){
            result += a[j] * power_int(x,j);
        }
    }
    else{
        utype num=0;
        utype denom=0;
        for(int j=0;j<5;j++){
            num += b[j] * power_int(x,4 - j);
            denom += c[j] * power_int(x,4 - j);
            result = 1/x*exp(-x)*num/denom;
        }
    }
    return result;
}

// constructing a table with Planck function values for given wavelengths and in a suitable temperature range
__global__ void plancktable(
        utype* planck_grid, 
        utype* lambda_edge, 
        utype* deltalambda,
        int 	nwave, 
        utype 	Tstar, 
        int 	p_iter,
        int     dim,
        int     step
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int t = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nwave && t < (dim/10+1)) {

        utype T;
        utype shifty;
        utype D;
        utype y_bot;
        utype y_top;

        // building flexible temperature grid from '1 K' to 'dim * 2 - 1 K' at 'step K' resolution
        // and Tstar
        if(t < (dim/10)){
                T = (t + p_iter * (dim/10)) * step + 1;
        }
        if(p_iter == 9){
            if(t == dim/10){
                T = Tstar;
            }
        }

        planck_grid[x + (t + p_iter * (dim/10)) * nwave] = 0.0;

        // analytical calculation, only for T > 0
        if(T > 0.01){
            D = 2.0 * (power_int(KBOLTZMANN / HCONST, 3) * KBOLTZMANN * power_int(T, 4)) / (CSPEED*CSPEED);
            y_top = HCONST * CSPEED / (lambda_edge[x+1] * KBOLTZMANN * T);
            y_bot = HCONST * CSPEED / (lambda_edge[x] * KBOLTZMANN * T);

            // rearranging so that y_top < y_bot (i.e. wavelengths are always increasing)
            if(y_bot < y_top){
                shifty = y_top;
                y_top = y_bot;
                y_bot = shifty;
            }

            for(int n=1;n<200;n++){
                planck_grid[x + (t + p_iter * (dim/10)) * nwave] += D * analyt_planck(n, y_bot, y_top);
            }
        }
        planck_grid[x + (t + p_iter * (dim/10)) * nwave] /= deltalambda[x];
    }
}

// OUTDATED -- plancktable was found to provide exactly the same results
// // constructing a table with Planck function values at precise wavelength values for the sampling case
// __global__ void plancktable_sampling(
//         utype* planck_grid, 
//         utype* lambda,
//         int 	nwave, 
//         utype 	Tstar,
//         int 	p_iter,
//         int     dim,
//         int     step
// ){
// 
//     int x = threadIdx.x + blockIdx.x * blockDim.x;
//     int t = threadIdx.y + blockIdx.y * blockDim.y;
// 
//     if (x < nwave && t < (dim/10+1)) {
// 
//         utype T;
// 
//         // building flexible temperature grid from '1 K' to 'dim * 2 - 1 K' at 'step K' resolution
//         // and Tstar
//         if(t < (dim/10)){
//                 T = (t + p_iter * (dim/10)) * step + 1;
//         }
//         if(p_iter == 9){
//             if(t == (dim/10)){
//                 T = Tstar;
//             }
//         }
//         planck_grid[x + (t + p_iter * (dim/10)) * nwave] = 0.0;
//                 
//         if(T > 0.1){
//             planck_grid[x + (t + p_iter * (dim/10)) * nwave] = planck_func(lambda[x], T);
//         }
//     }
// }

// calculates the planck function for given surface temperature
__global__ void calc_surface_planck(
    utype* planckband_lay, 
    utype* lambda_edge, 
    utype* deltalambda,
    int 	nwave,
    int     numlayers,
    utype 	T_surf
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (x < nwave) {
        
        utype shifty;
        utype D;
        utype y_bot;
        utype y_top;

        planckband_lay[numlayers+1 + x * (numlayers + 2)] = 0.0;

        // analytical calculation, only for T_surf > 0
        if(T_surf > 0.01){
            D = 2.0 * (power_int(KBOLTZMANN / HCONST, 3) * KBOLTZMANN * power_int(T_surf, 4)) / (CSPEED*CSPEED);
            y_top = HCONST * CSPEED / (lambda_edge[x+1] * KBOLTZMANN * T_surf);
            y_bot = HCONST * CSPEED / (lambda_edge[x] * KBOLTZMANN * T_surf);

            // rearranging so that y_top < y_bot (i.e. wavelengths are always increasing)
            if(y_bot < y_top){
                shifty = y_top;
                y_top = y_bot;
                y_bot = shifty;
            }

            for(int n=1;n<200;n++){
                planckband_lay[numlayers+1 + x * (numlayers + 2)] += D * analyt_planck(n, y_bot, y_top);
            }
        }
        planckband_lay[numlayers+1 + x * (numlayers + 2)] /= deltalambda[x];
    }
}

// calculates the planck function for given surface temperature
// Note: US: note used
__global__ void calc_surface_planck_sampling(
    utype* planckband_lay, 
    utype* lambda,
    int 	nwave,
    int     numlayers,
    utype 	T_surf
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (x < nwave) {
        
        planckband_lay[numlayers+1 + x * (numlayers + 2)] = 0.0;

        if(T_surf > 0.1){
            planckband_lay[numlayers+1 + x * (numlayers + 2)] = planck_func(lambda[x], T_surf);
        }
    }
}

// adjust the incident flux to correspond to the correct brightness temperature
__global__ void corr_inc_energy(
        utype* 	planck_grid,
        utype* 	starflux,
        utype* 	deltalambda,
        int 	realstar,
        int 	nwave, 
        utype 	Tstar,
        int     dim
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x < nwave){

        utype num_flux = 0;

        if(realstar == 1){

            for (int xl = 0; xl < nwave; xl++){

                num_flux += deltalambda[xl] * starflux[xl];
            }
        }
        else{
            for (int xl = 0; xl < nwave; xl++){
                
                num_flux += deltalambda[xl] * PI * planck_grid[xl + dim * nwave];
                
            }
        }
        
        utype theo_flux = STEFANBOLTZMANN * pow(Tstar, 4.0);
        
        utype corr_factor = theo_flux / num_flux;
        if(x==0){
            if(corr_factor > 1) printf("\nEnergy budget corrected (increased) by %.2f percent.\n", 100.0 * (corr_factor - 1.0));
            if(corr_factor < 1) printf("\nEnergy budget corrected (decreased) by %.2f percent.\n", 100.0 * (1.0 - corr_factor));
        }
        if(realstar == 1){
            
            starflux[x] *= corr_factor;
        }
        else{
            
            planck_grid[x + dim * nwave] *= corr_factor;
            
        }
    }
}


// adjust the surface emission to satisfy the Stefan-Boltzmann law for the surface temperature
__global__ void corr_surface_emission(
    utype*  F_down_tot,
    utype*  delta_lambda,
    utype* 	planckband_lay,
    utype   surf_albedo,
    utype   T_surf,
    int     nwave,
    int     numlayers,
    int     itervalue
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (x < nwave){
        
        utype uncorr_emission = 0;
        
        for (int xl = 0; xl < nwave; xl++){
            
            uncorr_emission += delta_lambda[xl] * PI * planckband_lay[(numlayers+1) + xl * (numlayers + 2)];
        }
        
        utype corr_factor = 1.0;
        
        if(uncorr_emission > 0){
            corr_factor = STEFANBOLTZMANN * pow(T_surf, 4.0) / uncorr_emission;
        }
        // correction info -- for debugging purposes
        //if(x == 0 and itervalue % 100 == 0) printf("Surface emission corrected by %.2f ppm.\n", 1e6 * (corr_factor - 1.0));
        
        planckband_lay[(numlayers+1) + x * (numlayers + 2)] *= corr_factor;
    }
}


// constructing the atmospheric grid with pressure & column mass
__global__ void gridkernel(
        utype* p_lay, 
        utype* p_int,
        utype* delta_colmass,
        utype* delta_col_upper, 
        utype* delta_col_lower, 
        utype 	p_boa, 
        utype 	p_toa, 
        int 	nlayer, 
        utype 	g
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < nlayer){

        p_lay[i] = p_boa * exp(log(p_toa / p_boa) * i / (nlayer - 1.0));
         p_int[i] = p_lay[i] * exp(log(p_boa / p_toa)   *  1.0 / (2.0 * (nlayer - 1.0)));
        if (i == nlayer - 1){
            p_int[i+1] = p_lay[i] * exp(log(p_toa / p_boa)   *  1.0 / (2.0 * (nlayer - 1.0)));
        }
    }
    __syncthreads();

    if (i < nlayer) {
        delta_colmass[i] = (p_int[i] - p_int[i+1]) / g;
        delta_col_upper[i] = (p_lay[i] - p_int[i+1]) / g;
        delta_col_lower[i] = (p_int[i] - p_lay[i]) / g;
    }
}


// EXPERIMENTAL - normalizing cloud scattering to match absorption coefficients
__global__ void cloud_normalization(
        utype* press,
        utype* abs_cross_cloud,
        utype* cloud_opac,
        utype* scat_cross_cloud,
        utype* cloud_scat_cross,
        utype* scat_cross,
        utype* meanmolmass,
        utype* g_0_tot,
        utype* g_0_cloud,
        utype  g_0,
        utype cloud_opac_tot,
        utype cloud_press,
        utype cloud_width,
        int 	nbin, 
        int 	nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && i < nlay_or_nint) {

        // grey cloud opacity
        cloud_opac[i] = cloud_opac_tot * norm_pdf(log10(press[i]), log10(cloud_press), cloud_width);

        // normalization factor
        utype f_norm_factor =  cloud_opac[i] / abs_cross_cloud[x] * meanmolmass[i];

        // normalized cloud scattering
        cloud_scat_cross[x + nbin * i] = f_norm_factor * scat_cross_cloud[x];

        // calculating the total (weighted average) scattering asymmetry factor
        utype num = g_0 * scat_cross[x + nbin * i] + g_0_cloud[x] * cloud_scat_cross[x + nbin * i];
        utype denom = scat_cross[x + nbin * i] + cloud_scat_cross[x + nbin * i];
        g_0_tot[x + nbin * i] = num / denom;
    }
}


// temperature interpolation for the non-isothermal layers
__global__ void temp_inter(
        utype* tlay, 
        utype* tint, 
        int numinterfaces
){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (0 < i && i < numinterfaces - 1) {
        tint[i] = tlay[i - 1] + 0.5 * (tlay[i] - tlay[i - 1]);
    }
    if (i == 0) {
        tint[i] = tlay[i] - 0.5 * (tlay[i + 1] - tlay[i]);
    }
    if (i == numinterfaces - 1) {
        tint[i] = tlay[i - 1] + 0.5 * (tlay[i - 1] - tlay[i - 2]);
    }
}


// interpolate layer and interface opacities from opacity table
__global__ void opac_interpol(
        utype*  temp, 
        utype*  opactemp, 
        utype*  press, 
        utype*  opacpress,
        utype*  ktable, 
        utype*  opac,
        utype*  crosstable,
        utype*  scat_cross,
        int 	npress, 
        int 	ntemp, 
        int 	ny,
        int 	nbin, 
        utype 	opaclimit, 
        int 	nlay_or_nint
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && i < nlay_or_nint) {

        int x_1micron = lrint(nbin * 2.0 / 3.0);

        utype deltaopactemp = (opactemp[ntemp-1] - opactemp[0])/(ntemp-1.0);
        utype deltaopacpress = (log10(opacpress[npress -1]) - log10(opacpress[0])) / (npress-1.0);
        utype t = (temp[i] - opactemp[0]) / deltaopactemp;

        t = min(ntemp-1.001, max(0.001, t));

        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

        // do the cloud deck
        utype k_cloud = 0.0; //1e-1 * norm_pdf(log10(press[i]),0,1);

        p = min(npress-1.001, max(0.001, p));
                
        int pdown = floor(p);
        int pup = ceil(p);

        if(pdown != pup && tdown != tup){
            for(int y=0;y<ny;y++){
                utype interpolated_opac =
                        ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (pup - p) * (tup - t)
                        + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown] * (p - pdown) * (tup - t)
                        + ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup] * (pup - p) * (t -  tdown)
                        + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tup] * (p - pdown) * (t - tdown);

                if (x < x_1micron) {
                    opac[y+ny*x + ny*nbin*i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y+ny*x + ny*nbin*i] = interpolated_opac;
                }

                opac[y+ny*x + ny*nbin*i] += k_cloud;
            }

            scat_cross[x + nbin * i] =
                    crosstable[x + nbin* pdown + nbin*npress * tdown] * (pup - p) * (tup - t)
                    + crosstable[x + nbin* pup + nbin*npress * tdown] * (p - pdown) * (tup - t)
                    + crosstable[x + nbin* pdown + nbin*npress * tup] * (pup - p) * (t -  tdown)
                    + crosstable[x + nbin* pup + nbin*npress * tup] * (p - pdown) * (t - tdown);
        }

        if(tdown == tup && pdown != pup){
            for(int y=0;y<ny;y++){
                utype interpolated_opac = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (pup - p)
                                            + ktable[y + ny*x + ny*nbin* pup + ny*nbin*npress * tdown] * (p - pdown);
                if (x < x_1micron) {
                    opac[y+ny*x + ny*nbin*i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y+ny*x + ny*nbin*i] = interpolated_opac;
                }
                
                opac[y+ny*x + ny*nbin*i] += k_cloud;
                
            }
            
            scat_cross[x + nbin * i] =
                    crosstable[x + nbin* pdown + nbin*npress * tdown] * (pup - p)
                    + crosstable[x + nbin* pup + nbin*npress * tdown] * (p - pdown);
        }

        if(pdown == pup && tdown != tup){
            for(int y=0;y<ny;y++){
                utype interpolated_opac = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown] * (tup - t)
                                            + ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tup] * (t -  tdown);
                if (x < x_1micron) {
                    opac[y+ny*x + ny*nbin*i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y+ny*x + ny*nbin*i] = interpolated_opac;
                }
                
                opac[y+ny*x + ny*nbin*i] += k_cloud;
            }
            
            scat_cross[x + nbin * i] = 
                    crosstable[x + nbin* pdown + nbin*npress * tdown] * (tup - t) 
                    + crosstable[x + nbin* pdown + nbin*npress * tup] * (t -  tdown);
        }

        if(tdown == tup && pdown == pup){
            for(int y=0;y<ny;y++){

                utype interpolated_opac = ktable[y + ny*x + ny*nbin* pdown + ny*nbin*npress * tdown];

                if (x < x_1micron) {
                    opac[y+ny*x + ny*nbin*i] = max(interpolated_opac, opaclimit);
                }
                else {
                    opac[y+ny*x + ny*nbin*i] = interpolated_opac;
                }
                
                opac[y+ny*x + ny*nbin*i] += k_cloud;
                
            }
            
            scat_cross[x + nbin * i] = crosstable[x + nbin* pdown + nbin*npress * tdown];
        }
    }
}


__device__ utype bilin_interpol_func(
        utype pdowntdown,
        utype puptdown,
        utype pdowntup,
        utype puptup,
        utype p,
        utype t,
        int pdown,
        int pup,
        int tdown,
        int tup
){
    utype interpolated;
    
    if(pdown != pup && tdown != tup){
        interpolated = pdowntdown * (pup - p) * (tup - t)
                    + puptdown * (p - pdown) * (tup - t)
                    + pdowntup * (pup - p) * (t - tdown)
                    + puptup * (p - pdown) * (t - tdown);
    }
    if(tdown == tup && pdown != pup){
        interpolated = pdowntdown * (pup - p) 
                    + puptdown * (p - pdown);
    }
    if(pdown == pup && tdown != tup){
        interpolated = pdowntup * (t - tdown) 
                    + pdowntdown * (tup - t);
    }
    if(tdown == tup && pdown == pup){
        interpolated = pdowntdown;
    }	
    return interpolated;
}


// interpolate the mean molecular mass for each layer
__global__ void meanmolmass_interpol(
        utype* temp, 
        utype* opactemp, 
        utype* meanmolmass, 
        utype* opac_meanmass,
        utype* press, 
        utype* opacpress,
        int 	npress, 
        int 	ntemp, 
        int 	ninterface
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < ninterface){

        utype deltaopactemp = (opactemp[ntemp-1] - opactemp[0])/(ntemp-1.0);
        utype deltaopacpress = (log10(opacpress[npress -1]) - log10(opacpress[0])) / (npress-1.0);
        utype t = (temp[i] - opactemp[0]) / deltaopactemp;

        t = min(ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(opacpress[0])) / deltaopacpress;

        p = min(npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);

        if(tdown != tup && pdown != pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (pup - p) * (tup - t)
                            + opac_meanmass[pup + npress * tdown] * (p - pdown) * (tup - t)
                            + opac_meanmass[pdown + npress * tup] * (pup - p) * (t -  tdown)
                            + opac_meanmass[pup + npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (tup - t)
                            + opac_meanmass[pdown + npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown] * (pup - p)
                            + opac_meanmass[pup + npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            meanmolmass[i] = opac_meanmass[pdown + npress * tdown];
        }
    }
}




// interpolate kappa for each layer
__global__ void kappa_interpol(
    utype*  temp, 
    utype*  entr_temp, 
    utype*  press, 
    utype*  entr_press,
    utype*  kappa, 
    utype*  opac_kappa,
    int     entr_npress,
    int 	entr_ntemp,
    int     nlay_or_nint,
    utype   kappa_kernel_value
){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i < nlay_or_nint){
        
        if(kappa_kernel_value < 0.001){
            
            utype delta_temp = (log10(entr_temp[entr_ntemp-1]) - log10(entr_temp[0])) / (entr_ntemp-1.0);
            utype delta_press = (log10(entr_press[entr_npress-1]) - log10(entr_press[0])) / (entr_npress-1.0);
            
            utype t = (log10(temp[i]) - log10(entr_temp[0])) / delta_temp;
            
            t = min(entr_ntemp-1.001, max(0.001, t));
            
            int tdown = floor(t);
            int tup = ceil(t);
            
            utype p = (log10(press[i]) - log10(entr_press[0])) / delta_press;
            
            p = min(entr_npress-1.001, max(0.001, p));
            
            int pdown = floor(p);
            int pup = ceil(p);
            
            if(tdown != tup && pdown != pup){
                kappa[i] = opac_kappa[pdown + entr_npress * tdown] * (pup - p) * (tup - t)
                + opac_kappa[pup + entr_npress * tdown] * (p - pdown) * (tup - t)
                + opac_kappa[pdown + entr_npress * tup] * (pup - p) * (t -  tdown)
                + opac_kappa[pup + entr_npress * tup] * (p - pdown) * (t - tdown);
            }
            if(tdown != tup && pdown == pup){
                kappa[i] = opac_kappa[pdown + entr_npress * tdown] * (tup - t)
                + opac_kappa[pdown + entr_npress * tup] * (t -  tdown);
            }
            if(tdown == tup && pdown != pup){
                kappa[i] = opac_kappa[pdown + entr_npress * tdown] * (pup - p)
                + opac_kappa[pup + entr_npress * tdown] * (p - pdown);
            }
            if(tdown == tup && pdown == pup){
                kappa[i] = opac_kappa[pdown + entr_npress * tdown];
            }
        }
        else{
            kappa[i] =  kappa_kernel_value;
        }
    }
}


// interpolate entropy for each layer
__global__ void entropy_interpol(
        utype* temp, 
        utype* entr_temp, 
        utype* press, 
        utype* entr_press,
        utype* entropy, 
        utype* opac_entropy,
        int 	entr_npress, 
        int 	entr_ntemp, 
        int 	nlayer
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){

        utype delta_temp = (log10(entr_temp[entr_ntemp-1]) - log10(entr_temp[0])) / (entr_ntemp-1.0);
        utype delta_press = (log10(entr_press[entr_npress-1]) - log10(entr_press[0])) / (entr_npress-1.0);
        utype t = (log10(temp[i]) - log10(entr_temp[0])) / delta_temp;

        t = min(entr_ntemp-1.001, max(0.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);

        utype p = (log10(press[i]) - log10(entr_press[0])) / delta_press;

        p = min(entr_npress-1.001, max(0.001, p));
        
        int pdown = floor(p);
        int pup = ceil(p);

        if(tdown != tup && pdown != pup){
            entropy[i] = opac_entropy[pdown + entr_npress * tdown] * (pup - p) * (tup - t)
                            + opac_entropy[pup + entr_npress * tdown] * (p - pdown) * (tup - t)
                            + opac_entropy[pdown + entr_npress * tup] * (pup - p) * (t -  tdown)
                            + opac_entropy[pup + entr_npress * tup] * (p - pdown) * (t - tdown);
        }
        if(tdown != tup && pdown == pup){
            entropy[i] = opac_entropy[pdown + entr_npress * tdown] * (tup - t)
                            + opac_entropy[pdown + entr_npress * tup] * (t -  tdown);
        }
        if(tdown == tup && pdown != pup){
            entropy[i] = opac_entropy[pdown + entr_npress * tdown] * (pup - p)
                            + opac_entropy[pup + entr_npress * tdown] * (p - pdown);
        }
        if(tdown == tup && pdown == pup){
            entropy[i] = opac_entropy[pdown + entr_npress * tdown];
        }
    }
}


// calculate the heat capacity from kappa and meanmolmass
__global__ void calculate_cp(
        utype* kappa,
        utype* meanmolmass_lay,
        utype* c_p_lay,
        int nlayer
){
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){
        
        c_p_lay[i] = KBOLTZMANN / (kappa[i] * meanmolmass_lay[i]);
    }
}


// interpolates the Planck function for the layer temperatures from the pre-tabulated values
__global__ void planck_interpol_layer(
    utype* 	temp, 
    utype* 	planckband_lay,
    utype* 	planck_grid, 
    utype* 	starflux, 
    int 	realstar, 
    int 	numlayers, 
    int 	nwave,
    utype   T_surf,
    int     dim,
    int     step
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nwave){
        
        if(i < numlayers){
            
            planckband_lay[i + x * (numlayers + 2)] = 0.0;
            
            utype t = (temp[i] - 1.0) / step;
            
            t = max(0.001, min(dim - 1.001, t));
            
            int tdown = floor(t);
            int tup = ceil(t);
            
            if(tdown != tup){
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave] * (tup - t)
                + planck_grid[x + tup * nwave] * (t-tdown);
            }
            if(tdown == tup){
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + tdown * nwave];
            }
            
        }
        // taking stellar and internal temperatures
        if (i == numlayers){
            if(realstar==1){
                planckband_lay[i + x * (numlayers + 2)] = starflux[x]/PI;
            }
            else{
                planckband_lay[i + x * (numlayers + 2)] = planck_grid[x + dim * nwave];
            }
        }
    }
}


// interpolates the Planck function for the interface temperatures from the pre-tabulated values
__global__ void planck_interpol_interface(
    utype* temp, 
    utype* planckband_int, 
    utype* planck_grid, 
    int 	numinterfaces, 
    int 	nwave,
    int     dim,
    int     step
){
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < nwave && i < numinterfaces){
        
        planckband_int[i + x * numinterfaces] = 0.0;
        
        utype t = (temp[i] - 1.0) / step;
        
        t = max(0.001, min(dim - 1.001, t));
        
        int tdown = floor(t);
        int tup = ceil(t);
        
        if(tdown != tup){
            planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave] * (tup - t)
            + planck_grid[x + tup * nwave] * (t - tdown);
        }
        if(tdown == tup){
            planckband_int[i + x * numinterfaces] = planck_grid[x + tdown * nwave];
        }
    }
}


// calculation of transmission, w0, zeta-functions, and capital letters for the layer centers in the isothermal case
// TODO: check ny meaning
// kernel runs per wavelength bin, per wavelength sampling (?) and per layer
__global__ void calc_trans_iso(
        utype* 	trans_wg,
        utype* 	delta_tau_wg,
        utype* 	M_term,
        utype* 	N_term,
        utype* 	P_term,
        utype* 	G_plus,
        utype* 	G_minus,
        utype* 	delta_colmass,
        utype* 	opac_wg_lay,
        utype* cloud_opac_lay,
        utype* 	meanmolmass_lay,
        utype* 	scat_cross_lay,
        utype* 	cloud_scat_cross_lay,
        utype*  w_0,
        utype* 	g_0_tot_lay,
        utype   g_0,
        utype 	epsi,
        utype 	mu_star,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr
){
    // indices
    // wavelength bin
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    // sampling point (?)
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // layer
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer) {

        utype ray_cross;
        utype cloud_cross;
        utype g0 = g_0;

        if(clouds == 1){
            g0 = g_0_tot_lay[x + nbin*i];
        }

        if (scat == 1){
            ray_cross = scat_cross_lay[x + nbin*i];
            cloud_cross = cloud_scat_cross_lay[x + nbin*i];
        }
        else{
            ray_cross = 0;
            cloud_cross = 0;
        }

        w_0[y+ny*x + ny*nbin*i] = single_scat_alb(ray_cross + cloud_cross, opac_wg_lay[y+ny*x + ny*nbin*i] + cloud_opac_lay[i], meanmolmass_lay[i]);
        utype w0 = w_0[y+ny*x + ny*nbin*i];

        delta_tau_wg[y+ny*x + ny*nbin*i] = delta_colmass[i] * (opac_wg_lay[y+ny*x + ny*nbin*i] + cloud_opac_lay[i] + (ray_cross + cloud_cross)/meanmolmass_lay[i]);
        utype del_tau = delta_tau_wg[y+ny*x + ny*nbin*i];
        trans_wg[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau, w0, g0);
        utype trans = trans_wg[y+ny*x + ny*nbin*i];

        utype zeta_min = zeta_minus(w0, g0, scat_corr);
        utype zeta_pl = zeta_plus(w0, g0, scat_corr);

        M_term[y+ny*x + ny*nbin*i] = (zeta_min*zeta_min) * (trans*trans) - (zeta_pl*zeta_pl);
        N_term[y+ny*x + ny*nbin*i] = zeta_pl * zeta_min * (1.0 - (trans*trans));
        P_term[y+ny*x + ny*nbin*i] = ((zeta_min*zeta_min) - (zeta_pl*zeta_pl)) * trans;
                
        G_plus[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w0, g0, epsi, mu_star));
        G_minus[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w0, g0, epsi, mu_star));
    }
}

// calculation of transmission, w0, zeta-functions, and capital letters for the non-isothermal case
__global__ void calc_trans_noniso(
        utype* trans_wg_upper,
        utype* trans_wg_lower,
        utype* delta_tau_wg_upper,
        utype* delta_tau_wg_lower,
        utype* M_upper,
        utype* M_lower,
        utype* N_upper,
        utype* N_lower,
        utype* P_upper,
        utype* P_lower,
        utype* G_plus_upper,
        utype* G_plus_lower,
        utype* G_minus_upper,
        utype* G_minus_lower,
        utype* delta_col_upper,
        utype* delta_col_lower,
        utype* opac_wg_lay,
        utype* opac_wg_int,
        utype* cloud_opac_lay,
        utype* cloud_opac_int,		
        utype* meanmolmass_lay,
        utype* meanmolmass_int,
        utype* scat_cross_lay,
        utype* scat_cross_int,
        utype* cloud_scat_cross_lay,
        utype* cloud_scat_cross_int,		
        utype* w_0_upper,
        utype* w_0_lower,
        utype* 	g_0_tot_lay,
        utype* 	g_0_tot_int,
        utype	g_0,
        utype 	epsi,
        utype 	mu_star,
        int 	scat,
        int 	nbin,
        int 	ny,
        int 	nlayer,
        int 	clouds,
        int 	scat_corr
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.z + blockIdx.z * blockDim.z;

    if (x < nbin && y < ny && i < nlayer){

        utype ray_cross_up;
        utype ray_cross_low;
        utype cloud_cross_up;
        utype cloud_cross_low;
        utype g0_up = g_0;
        utype g0_low = g_0;
        
        if(clouds == 1){
            g0_up = (g_0_tot_lay[x + nbin*i] + g_0_tot_int[x + nbin*(i+1)]) / 2.0;
            g0_low = (g_0_tot_int[x + nbin*i] + g_0_tot_lay[x + nbin*i]) / 2.0;
        }

        if (scat == 1){
            ray_cross_up = (scat_cross_lay[x + nbin*i] + scat_cross_int[x + nbin*(i+1)]) / 2.0;
            ray_cross_low = (scat_cross_int[x + nbin*i] + scat_cross_lay[x + nbin*i]) / 2.0;
            cloud_cross_up = (cloud_scat_cross_lay[x + nbin*i] + cloud_scat_cross_int[x + nbin*(i+1)]) / 2.0;
            cloud_cross_low = (cloud_scat_cross_int[x + nbin*i] + cloud_scat_cross_lay[x + nbin*i]) / 2.0;
        }
        else{
            ray_cross_up = 0;
            ray_cross_low = 0;
            cloud_cross_up = 0;
            cloud_cross_low = 0;
        }
        
        utype opac_up = (opac_wg_lay[y+ny*x + ny*nbin*i]+opac_wg_int[y+ny*x + ny*nbin*(i+1)]) / 2.0;
        utype opac_low = (opac_wg_int[y+ny*x + ny*nbin*i]+opac_wg_lay[y+ny*x + ny*nbin*i]) / 2.0;
        utype cloud_opac_up = (cloud_opac_lay[i] + cloud_opac_int[i+1]) / 2.0;
        utype cloud_opac_low = (cloud_opac_int[i] + cloud_opac_lay[i]) / 2.0;

        utype meanmolmass_up = (meanmolmass_lay[i] + meanmolmass_int[i+1]) / 2.0;
        utype meanmolmass_low = (meanmolmass_int[i] + meanmolmass_lay[i]) / 2.0;
        
        w_0_upper[y+ny*x + ny*nbin*i] = single_scat_alb(ray_cross_up + cloud_cross_up, opac_up + cloud_opac_up, meanmolmass_up);
        utype w_0_up = w_0_upper[y+ny*x + ny*nbin*i];
        w_0_lower[y+ny*x + ny*nbin*i] = single_scat_alb(ray_cross_low + cloud_cross_low, opac_low + cloud_opac_low, meanmolmass_low);
        utype w_0_low = w_0_lower[y+ny*x + ny*nbin*i];

        delta_tau_wg_upper[y+ny*x + ny*nbin*i] = delta_col_upper[i] * (opac_up + cloud_opac_up + (ray_cross_up + cloud_cross_up)/meanmolmass_up);
        utype del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i];
        delta_tau_wg_lower[y+ny*x + ny*nbin*i] = delta_col_lower[i] * (opac_low + cloud_opac_low + (ray_cross_low + cloud_cross_low)/meanmolmass_low);
        utype del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i];

        trans_wg_upper[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau_up, w_0_up, g0_up);
        utype trans_up = trans_wg_upper[y+ny*x + ny*nbin*i];
        trans_wg_lower[y+ny*x + ny*nbin*i] = trans_func(epsi, del_tau_low, w_0_low, g0_low);
        utype trans_low = trans_wg_lower[y+ny*x + ny*nbin*i];
        
        utype zeta_min_up = zeta_minus(w_0_up, g0_up, scat_corr);
        utype zeta_min_low = zeta_minus(w_0_low, g0_low, scat_corr);
        utype zeta_pl_up = zeta_plus(w_0_up, g0_up, scat_corr);		
        utype zeta_pl_low = zeta_plus(w_0_low, g0_low, scat_corr);

        M_upper[y+ny*x + ny*nbin*i] = (zeta_min_up*zeta_min_up) * (trans_up*trans_up) - (zeta_pl_up*zeta_pl_up);
        M_lower[y+ny*x + ny*nbin*i] = (zeta_min_low*zeta_min_low) * (trans_low*trans_low) - (zeta_pl_low*zeta_pl_low);
        N_upper[y+ny*x + ny*nbin*i] = zeta_pl_up * zeta_min_up * (1.0 - (trans_up*trans_up));
        N_lower[y+ny*x + ny*nbin*i] = zeta_pl_low * zeta_min_low * (1.0 - (trans_low*trans_low));
        P_upper[y+ny*x + ny*nbin*i] = ((zeta_min_up*zeta_min_up) - (zeta_pl_up*zeta_pl_up)) * trans_up;
        P_lower[y+ny*x + ny*nbin*i] = ((zeta_min_low*zeta_min_low) - (zeta_pl_low*zeta_pl_low)) * trans_low;

        G_plus_upper[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w_0_up, g0_up, epsi, mu_star));
        G_plus_lower[y+ny*x + ny*nbin*i] = G_limiter(G_plus_func(w_0_low, g0_low, epsi, mu_star));
        G_minus_upper[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w_0_up, g0_up, epsi, mu_star));
        G_minus_lower[y+ny*x + ny*nbin*i] = G_limiter(G_minus_func(w_0_low, g0_low, epsi, mu_star));

    }
}


// calculates the height of a layer
__global__ void calc_delta_z(
        utype* 	tlay,
        utype* 	pint,
        utype* 	play,
        utype* 	meanmolmass_lay,
        utype* 	delta_z_lay,
        utype 	g,
        int		nlayer
){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < nlayer){
        delta_z_lay[i] = KBOLTZMANN * tlay[i] / (meanmolmass_lay[i] * g) * log(pint[i] / pint[i+1]);
    }
}


// calculates the direct beam flux with geometric zenith angle correction, isothermal version
__global__ void fdir_iso(
        utype* 	F_dir_wg,
        utype* 	planckband_lay,
        utype* 	delta_tau_wg,
        utype* 	z_lay,
        utype 	mu_star,
        utype	R_planet,
        utype 	R_star, 
        utype 	a,
        int		dir_beam,
        int		geom_zenith_corr,
        int 	ninterface,
        int 	nbin,
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        utype I_dir = ((R_star / a)*(R_star / a)) * PI * planckband_lay[(ninterface - 1) + x * (ninterface-1+2)];

        // initialize each flux value
        F_dir_wg[y + ny * x + ny * nbin * i]  = -dir_beam * mu_star * I_dir;

        utype mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for(int j = ninterface - 2; j >= i; j--){
            
            if(geom_zenith_corr == 1){
            mu_star_layer_j  = - sqrt(1.0 - pow((R_planet + z_lay[i])/(R_planet+z_lay[j]), 2.0) * (1.0 - pow(mu_star, 2.0)));
            }
            else{
                mu_star_layer_j = mu_star;
            }

            // direct stellar flux	
            F_dir_wg[y+ny*x+ny*nbin*i] *= exp(delta_tau_wg[y+ny*x + ny*nbin*j] / mu_star_layer_j);
        }
    }
}


// calculates the direct beam flux with geometric zenith angle correction, non-isothermal version
__global__ void fdir_noniso(
        utype* 	F_dir_wg,
        utype* 	Fc_dir_wg,
        utype* 	planckband_lay,
        utype* 	delta_tau_wg_upper,
        utype* 	delta_tau_wg_lower,
        utype* 	z_lay,
        utype 	mu_star,
        utype	R_planet,
        utype 	R_star, 
        utype 	a,
        int		dir_beam,
        int		geom_zenith_corr,
        int 	ninterface,
        int 	nbin,
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int y = threadIdx.z + blockIdx.z * blockDim.z;

    if (i < ninterface && x < nbin && y < ny) {

        // the stellar intensity at TOA
        utype I_dir = ((R_star / a)*(R_star / a)) * PI * planckband_lay[(ninterface - 1) + x * (ninterface-1+2)];

        // initialize each flux value
        F_dir_wg[y + ny * x + ny * nbin * i]  = -dir_beam * mu_star * I_dir;

        utype mu_star_layer_j;

        // flux values lower that TOA will now be attenuated depending on their location
        for(int j = ninterface - 2; j >= i; j--){

            if(geom_zenith_corr == 1){
                mu_star_layer_j  = - sqrt(1.0 - pow((R_planet + z_lay[i])/(R_planet+z_lay[j]), 2.0) * (1.0 - pow(mu_star, 2.0)));
            }
            else{
                mu_star_layer_j = mu_star;
            }
            
            utype delta_tau = delta_tau_wg_upper[y+ny*x + ny*nbin*j] + delta_tau_wg_lower[y+ny*x + ny*nbin*j];
            
            // direct stellar flux
            Fc_dir_wg[y+ny*x+ny*nbin*i] = F_dir_wg[y+ny*x+ny*nbin*i] * exp(delta_tau_wg_upper[y+ny*x + ny*nbin*j] / mu_star_layer_j);
            F_dir_wg[y+ny*x+ny*nbin*i] *= exp(delta_tau / mu_star_layer_j);
        }
    }
}


// calculation of the spectral fluxes, isothermal case with emphasis on on-the-fly calculations
__global__ void fband_iso_notabu(
        utype* F_down_wg, 
        utype* F_up_wg, 
        utype* F_dir_wg, 
        utype* planckband_lay,
        utype* w_0,
        utype* delta_tau_wg,
        utype* M_term,
        utype* N_term,
        utype* P_term,
        utype* G_plus,
        utype* G_minus,
        utype* g_0_tot_lay,
        utype 	g_0,
        int 	singlewalk, 
        utype 	Rstar, 
        utype 	a, 
        int 	numinterfaces, 
        int 	nbin, 
        utype 	f_factor, 
        utype 	mu_star,
        int 	ny, 
        utype 	epsi,
        utype 	w_0_limit,
        int 	dir_beam,
        int 	clouds,
        utype   albedo
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && y < ny) {
        
        for (int i = numinterfaces - 1; i >= 0; i--){

            if (i == numinterfaces - 1) {
                
                // flux at TOA (without direct irradiation beam)
                F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
            }
            else {
                
                utype w0 = w_0[y+ny*x + ny*nbin*i];
                utype del_tau = delta_tau_wg[y+ny*x + ny*nbin*i];
                utype M = M_term[y+ny*x + ny*nbin*i];
                utype N = N_term[y+ny*x + ny*nbin*i];
                utype P = P_term[y+ny*x + ny*nbin*i];
                utype G_pl = G_plus[y+ny*x + ny*nbin*i];
                utype G_min = G_minus[y+ny*x + ny*nbin*i];
                utype g0 = g_0;	
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * i];
                }
                
                if(w0 > w_0_limit){
                    // w0 = 1 solution
                    utype first_fraction = (F_up_wg[y+ny*x+ny*nbin*i] - F_down_wg[y+ny*x+ny*nbin*(i+1)]) * 1.0/epsi * (1.0 - g0)*del_tau/(1.0/epsi * (1.0 - g0) * del_tau + 2.0);
                            
                    utype large_bracket = (mu_star/epsi - 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) + (1.0/epsi * (1.0 - g0) * del_tau - mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                    
                    utype direct_terms = mu_star/(1.0/epsi * (1.0 - g0) * del_tau + 2.0) * large_bracket;
                            
                    F_down_wg[y+ny*x+ny*nbin*i] = F_down_wg[y+ny*x+ny*nbin*(i+1)] + first_fraction + direct_terms;

                }
                else{
                    // isothermal solution
                    utype flux_terms = P * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N * F_up_wg[y+ny*x+ny*nbin*i];
                    
                    utype planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] * (N + M - P);
                    
                    utype direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min * M + G_pl * N) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * P * G_min;
                    
                    F_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M * (flux_terms + 2.0 * PI * epsi * planck_terms + direct_terms);
                }
            }
        }

        for (int i = 0; i < numinterfaces; i++){

            if (i == 0){
                
                utype reflected_part = albedo * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                
                // this is the surface emission. it now comes with the emissivity e = (1 - albedo)
                utype internal_part = (1.0 - albedo) * PI * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
                
                F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + internal_part; // internal_part comprises the interior heat plus the surface emission
            }
            else {
                
                utype w0 = w_0[y+ny*x + ny*nbin*(i-1)];
                utype del_tau = delta_tau_wg[y+ny*x + ny*nbin*(i-1)];
                utype M = M_term[y+ny*x + ny*nbin*(i-1)];
                utype N = N_term[y+ny*x + ny*nbin*(i-1)];
                utype P = P_term[y+ny*x + ny*nbin*(i-1)];
                utype G_pl = G_plus[y+ny*x + ny*nbin*(i-1)];
                utype G_min = G_minus[y+ny*x + ny*nbin*(i-1)];
                utype g0 = g_0;
                if(clouds == 1){
                    g0 = g_0_tot_lay[x + nbin * (i-1)];
                }

                if(w0 > w_0_limit){
                    // w0 = 1 solution
                    utype first_fraction = (F_down_wg[y+ny*x+ny*nbin*i] - F_up_wg[y+ny*x+ny*nbin*(i-1)]) * 1.0/epsi * (1.0-g0)*del_tau / (1.0/epsi * (1.0-g0)*del_tau + 2.0);
                    
                    utype large_bracket = (mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) - (1.0/epsi * (1.0 - g0) * del_tau + mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                    
                    utype direct_terms = mu_star/(1.0/epsi * (1.0 - g0) * del_tau + 2.0) * large_bracket;
                    
                    F_up_wg[y+ny*x+ny*nbin*i] = F_up_wg[y+ny*x+ny*nbin*(i-1)] + first_fraction + direct_terms;
                }
                else{
                    // isothermal solution
                    utype flux_terms = P * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N * F_down_wg[y+ny*x+ny*nbin*i];
                    
                    utype planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (N + M - P);
                    
                    utype direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min * N + G_pl * M) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P * G_pl;
                    
                    F_up_wg[y+ny*x+ny*nbin*i] = 1.0 / M * (flux_terms + 2.0 * PI * epsi * planck_terms + direct_terms);
                }
            }
        }
    }
}


// calculation of the spectral fluxes, non-isothermal case with emphasis on on-the-fly calculations
__global__ void fband_noniso_notabu(
        utype* F_down_wg, 
        utype* F_up_wg, 
        utype* Fc_down_wg, 
        utype* Fc_up_wg,
        utype* F_dir_wg,
        utype* Fc_dir_wg,
        utype* planckband_lay, 
        utype* planckband_int,
        utype* w_0_upper,
        utype* w_0_lower,
        utype* delta_tau_wg_upper,
        utype* delta_tau_wg_lower,
        utype* M_upper,
        utype* M_lower,
        utype* N_upper,
        utype* N_lower,
        utype* P_upper,
        utype* P_lower,
        utype* G_plus_upper,
        utype* G_plus_lower,
        utype* G_minus_upper,
        utype* G_minus_lower,
        utype* g_0_tot_lay,
        utype* g_0_tot_int,
        utype 	g_0,
        int 	singlewalk, 
        utype 	Rstar, 
        utype 	a, 
        int 	numinterfaces,
        int 	nbin, 
        utype 	f_factor,
        utype 	mu_star,
        int 	ny,
        utype 	epsi,
        utype 	w_0_limit,
        utype 	delta_tau_limit,
        int 	dir_beam,
        int 	clouds,
        utype   albedo,
        utype*	trans_wg_upper,
        utype* trans_wg_lower
){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < nbin && y < ny) {

        for (int i = numinterfaces - 1; i >= 0; i--){

            if (i == numinterfaces - 1) {
                F_down_wg[y + ny * x + ny * nbin * i] = (1.0 - dir_beam) * f_factor * ((Rstar / a)*(Rstar / a)) * PI * planckband_lay[i + x * (numinterfaces-1+2)];
            }
            else {
                // upper part of layer
                utype w0_up = w_0_upper[y+ny*x + ny*nbin*i];
                utype del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*i];
                utype M_up = M_upper[y+ny*x + ny*nbin*i];
                utype N_up = N_upper[y+ny*x + ny*nbin*i];
                utype P_up = P_upper[y+ny*x + ny*nbin*i];
                utype G_pl_up = G_plus_upper[y+ny*x + ny*nbin*i];
                utype G_min_up = G_minus_upper[y+ny*x + ny*nbin*i];
                utype g0_up = g_0;
                if(clouds == 1){
                    g0_up = (g_0_tot_lay[x + nbin * i] + g_0_tot_int[x + nbin * (i+1)]) / 2.0;
                }
                
                // lower part of layer
                utype w0_low = w_0_lower[y+ny*x + ny*nbin*i];
                utype del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*i];
                utype M_low = M_lower[y+ny*x + ny*nbin*i];
                utype N_low = N_lower[y+ny*x + ny*nbin*i];
                utype P_low = P_lower[y+ny*x + ny*nbin*i];
                utype G_pl_low = G_plus_lower[y+ny*x + ny*nbin*i];
                utype G_min_low = G_minus_lower[y+ny*x + ny*nbin*i];
                utype g0_low = g_0;
                if(clouds == 1){
                    g0_low = (g_0_tot_int[x + nbin * i] + g_0_tot_lay[x + nbin * i]) / 2.0;
                }

                if(w0_up > w_0_limit){
                    // w0 = 1 solution
                    utype first_fraction = (Fc_up_wg[y+ny*x+ny*nbin*i] - F_down_wg[y+ny*x+ny*nbin*(i+1)]) * 1.0/epsi * (1.0-g0_up)*del_tau_up / (1.0/epsi * (1.0-g0_up)*del_tau_up + 2.0);

                    utype large_bracket = (mu_star/epsi - 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) + (1.0/epsi * (1.0 - g0_up) * del_tau_up - mu_star/epsi + 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                                        
                    utype direct_terms = mu_star/(1.0/epsi * (1.0 - g0_up) * del_tau_up + 2.0) * large_bracket;

                    Fc_down_wg[y+ny*x+ny*nbin*i] = F_down_wg[y+ny*x+ny*nbin*(i+1)] + first_fraction + direct_terms;
                }
                else{
                    utype flux_terms;
                    utype planck_terms;
                    utype direct_terms;

                    if(del_tau_up < delta_tau_limit){
                        // the isothermal solution
                        flux_terms = P_up * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N_up * Fc_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = (planckband_int[(i+1)+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_up + M_up - P_up);

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * M_up + G_pl_up * N_up) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * G_min_up * P_up;
                    }
                    else{
                        // the non-isothermal solution
                        utype pgrad_up = (planckband_lay[i + x * (numinterfaces-1+2)] - planckband_int[(i + 1) + x * numinterfaces]) / del_tau_up;

                        flux_terms = P_up * F_down_wg[y+ny*x+ny*nbin*(i+1)] - N_up * Fc_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = planckband_lay[i+x*(numinterfaces-1+2)] * (M_up + N_up) - planckband_int[(i+1)+x*numinterfaces] * P_up + epsi/(1.0-w0_up*g0_up) * pgrad_up * (P_up - M_up + N_up);

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * M_up + G_pl_up * N_up) - F_dir_wg[y+ny*x+ny*nbin*(i+1)]/(-mu_star) * P_up * G_min_up;
                    }
                    Fc_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M_up * (flux_terms + 2.0 * PI * epsi * planck_terms + direct_terms);
                }

                if(w0_low > w_0_limit){
                    // w0 = 1 solution
                    utype first_fraction = (F_up_wg[y+ny*x+ny*nbin*i] - Fc_down_wg[y+ny*x+ny*nbin*i]) * 1.0/epsi * (1.0-g0_low)*del_tau_low / (1.0/epsi * (1.0-g0_low)*del_tau_low + 2.0);

                    utype large_bracket = (mu_star/epsi - 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) + (1.0/epsi * (1.0 - g0_low) * del_tau_low - mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                                                            
                    utype direct_terms = mu_star/(1.0/epsi * (1.0 - g0_low) * del_tau_low + 2.0) * large_bracket;

                    F_down_wg[y+ny*x+ny*nbin*i] = Fc_down_wg[y+ny*x+ny*nbin*i] + first_fraction + direct_terms;
                }
                else{
                    utype flux_terms;
                    utype planck_terms;
                    utype direct_terms;

                    if(del_tau_low < delta_tau_limit){
                        // isothermal solution
                        flux_terms = P_low * Fc_down_wg[y+ny*x+ny*nbin*i] - N_low * F_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[i+x*(numinterfaces-1+2)])/2.0 * (N_low + M_low - P_low);

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_low * M_low + G_pl_low * N_low) - Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * P_low * G_min_low;
                    }
                    else{
                        // non-isothermal solution
                        utype pgrad_low = (planckband_int[i + x * numinterfaces] - planckband_lay[i + x * (numinterfaces-1+2)]) / del_tau_low;

                        flux_terms = P_low * Fc_down_wg[y+ny*x+ny*nbin*i] - N_low * F_up_wg[y+ny*x+ny*nbin*i];

                        planck_terms = planckband_int[i+x*numinterfaces] * (M_low + N_low) - planckband_lay[i+x*(numinterfaces-1+2)] * P_low + epsi/(1.0-w0_low*g0_low) * pgrad_low * (P_low - M_low + N_low) ;

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_low * M_low + G_pl_low * N_low) - Fc_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * P_low * G_min_low;						
                    }
                    F_down_wg[y+ny*x+ny*nbin*i] = 1.0 / M_low * (flux_terms + 2.0*PI*epsi*planck_terms + direct_terms);
                }
            }
        }

        __syncthreads();
        
        for (int i = 0; i < numinterfaces; i++){
            
            if (i == 0){
                
                utype reflected_part = albedo * (F_dir_wg[y+ny*x+ny*nbin*i] + F_down_wg[y+ny*x+ny*nbin* i]);
                
                // this is the surface emission. it now comes with the emissivity e = (1 - albedo)
                utype internal_part = (1.0 - albedo) * PI * planckband_lay[numinterfaces + x * (numinterfaces-1+2)];
                
                F_up_wg[y+ny*x+ny*nbin* i] = reflected_part + internal_part; // internal_part comprises the interior heat plus the surface emission
            }
            else {
                // lower part of layer
                utype w0_low = w_0_lower[y+ny*x + ny*nbin*(i-1)];
                utype del_tau_low = delta_tau_wg_lower[y+ny*x + ny*nbin*(i-1)];
                utype M_low = M_lower[y+ny*x + ny*nbin*(i-1)];
                utype N_low = N_lower[y+ny*x + ny*nbin*(i-1)];
                utype P_low = P_lower[y+ny*x + ny*nbin*(i-1)];
                utype G_pl_low = G_plus_lower[y+ny*x + ny*nbin*(i-1)];
                utype G_min_low = G_minus_lower[y+ny*x + ny*nbin*(i-1)];
                utype g0_low = g_0;
                if(clouds == 1){
                    g0_low = (g_0_tot_int[x + nbin * (i-1)] + g_0_tot_lay[x + nbin * (i-1)]) / 2.0;
                }
                                
                // upper part of layer
                utype w0_up = w_0_upper[y+ny*x + ny*nbin*(i-1)];
                utype del_tau_up = delta_tau_wg_upper[y+ny*x + ny*nbin*(i-1)];
                utype M_up = M_upper[y+ny*x + ny*nbin*(i-1)];
                utype N_up = N_upper[y+ny*x + ny*nbin*(i-1)];
                utype P_up = P_upper[y+ny*x + ny*nbin*(i-1)];
                utype G_pl_up = G_plus_upper[y+ny*x + ny*nbin*(i-1)];
                utype G_min_up = G_minus_upper[y+ny*x + ny*nbin*(i-1)];
                utype g0_up = g_0;
                if(clouds == 1){
                    g0_up = (g_0_tot_lay[x + nbin * (i-1)] + g_0_tot_int[x + nbin * i]) / 2.0;
                }

                if(w0_low > w_0_limit){
                    // w0 = 1 solution
                    utype first_fraction = (Fc_down_wg[y+ny*x+ny*nbin*(i-1)] - F_up_wg[y+ny*x+ny*nbin*(i-1)]) * 1.0/epsi * (1.0-g0_low)*del_tau_low / (1.0/epsi * (1.0-g0_low)*del_tau_low + 2.0);

                    utype large_bracket = (mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) - (1.0/epsi * (1.0 - g0_low) * del_tau_low + mu_star/epsi + 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star);
                    
                    utype direct_terms = mu_star/(1.0/epsi * (1.0 - g0_low) * del_tau_low + 2.0) * large_bracket;

                    Fc_up_wg[y+ny*x+ny*nbin*(i-1)] = F_up_wg[y+ny*x+ny*nbin*(i-1)] + first_fraction + direct_terms;
                }
                else{
                    utype flux_terms;
                    utype planck_terms;
                    utype direct_terms;

                    if(del_tau_low < delta_tau_limit){
                        // isothermal solution
                        flux_terms = P_low * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N_low * Fc_down_wg[y+ny*x+ny*nbin*(i-1)];

                        planck_terms = ( (planckband_int[(i-1)+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_low + M_low - P_low) ) ;

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * (G_min_low * N_low + G_pl_low * M_low) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_low * G_pl_low;

                    }
                    else{
                        // non-isothermal solution
                        utype pgrad_low = (planckband_int[(i-1) + x * numinterfaces] - planckband_lay[(i-1) + x * (numinterfaces-1+2)]) / del_tau_low;

                        flux_terms = P_low * F_up_wg[y+ny*x+ny*nbin*(i-1)] - N_low * Fc_down_wg[y+ny*x+ny*nbin*(i-1)];

                        planck_terms = planckband_lay[(i-1)+x*(numinterfaces-1+2)] * (M_low + N_low) - planckband_int[(i-1)+x*numinterfaces] * P_low + epsi/(1.0-w0_low*g0_low) * pgrad_low * (M_low - P_low - N_low);

                        direct_terms = Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * (G_min_low * N_low + G_pl_low * M_low) - F_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_low * G_pl_low;
                    }
                    Fc_up_wg[y+ny*x+ny*nbin*(i-1)] = 1.0 / M_low * (flux_terms + 2.0*PI*epsi*planck_terms + direct_terms);
                }

                if(w0_up > w_0_limit){
                    // w0 = 1 solution
                    utype first_fraction = (F_down_wg[y+ny*x+ny*nbin*i] - Fc_up_wg[y+ny*x+ny*nbin*(i-1)]) * 1.0/epsi * (1.0-g0_up)*del_tau_up / (1.0/epsi * (1.0-g0_up)*del_tau_up + 2.0);

                    utype large_bracket = (mu_star/epsi + 1.0) * Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) - (1.0/epsi * (1.0 - g0_up) * del_tau_up + mu_star/epsi + 1.0) * F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star);
                                        
                    utype direct_terms = mu_star/(1.0/epsi * (1.0 - g0_up) * del_tau_up + 2.0) * large_bracket;

                    F_up_wg[y+ny*x+ny*nbin*i] = Fc_up_wg[y+ny*x+ny*nbin*(i-1)] + first_fraction + direct_terms;
                }
                else{
                    utype flux_terms;
                    utype planck_terms;
                    utype direct_terms;

                    if(del_tau_up < delta_tau_limit){
                        // isothermal solution
                        flux_terms = P_up * Fc_up_wg[y+ny*x+ny*nbin*(i-1)] - N_up * F_down_wg[y+ny*x+ny*nbin*i];

                        planck_terms = (planckband_int[i+x*numinterfaces] + planckband_lay[(i-1)+x*(numinterfaces-1+2)])/2.0 * (N_up + M_up - P_up);

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * N_up + G_pl_up * M_up) - Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_up * G_pl_up;
                    }
                    else{
                        // non-isothermal solution
                        utype pgrad_up = (planckband_lay[(i-1) + x * (numinterfaces-1+2)] - planckband_int[i + x * numinterfaces]) / del_tau_up;

                        flux_terms = P_up * Fc_up_wg[y+ny*x+ny*nbin*(i-1)] - N_up * F_down_wg[y+ny*x+ny*nbin*i];

                        planck_terms = planckband_int[i+x*numinterfaces] * (M_up + N_up) - planckband_lay[(i-1)+x*(numinterfaces-1+2)] * P_up + epsi/(1.0-w0_up*g0_up) * pgrad_up * (M_up - P_up - N_up);

                        direct_terms = F_dir_wg[y+ny*x+ny*nbin*i]/(-mu_star) * (G_min_up * N_up + G_pl_up * M_up) - Fc_dir_wg[y+ny*x+ny*nbin*(i-1)]/(-mu_star) * P_up * G_pl_up;	
                    }
                    F_up_wg[y+ny*x+ny*nbin*i] = 1.0 / M_up * (flux_terms + 2.0*PI*epsi*planck_terms + direct_terms);
                }
            }
        }
    }
}


// OUTDATED -- calculates the integrated upwards and downwards fluxes
__global__ void integrate_flux(
        utype* deltalambda, 
        utype* F_down_tot, 
        utype* F_up_tot, 
        utype* F_net, 
        utype* F_down_wg, 
        utype* F_up_wg,
        utype* F_dir_wg,
        utype* F_down_band, 
        utype* F_up_band, 
        utype* F_dir_band,
        utype* gauss_weight, 
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < numinterfaces){

        F_up_tot[i] = 0;
        F_down_tot[i] = 0;


        for (int x = 0; x < nbin; x++) {

            F_dir_band[x + nbin * i] = 0;
            
            F_up_band[x + nbin * i] = 0;
            F_down_band[x + nbin * i] = 0;

            // Gauss - Legendre integration over each bin
            for(int y=0;y<ny;y++){
                F_dir_band[x + nbin * i] += 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i];
                F_up_band[x + nbin * i] += 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i];
                F_down_band[x + nbin * i] += 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i];
            }
            // sum the bin contributions to obtain the integrated flux
            F_up_tot[i] += F_up_band[x + nbin * i] * deltalambda[x];
            F_down_tot[i] += (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x];
        }
        // do the total stellar irradiation separately
        F_net[i] = F_up_tot[i] - F_down_tot[i];
    }
}


// calculates the integrated upwards and downwards fluxes
__global__ void integrate_flux_double(
        double* deltalambda, 
        double* F_down_tot, 
        double* F_up_tot, 
        double* F_net, 
        double* F_down_wg, 
        double* F_up_wg,
        double* F_dir_wg,
        double* F_down_band, 
        double* F_up_band, 
        double* F_dir_band,
        double* gauss_weight,
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            F_up_tot[i] = 0;
            F_down_tot[i] = 0;

            F_dir_band[x + nbin * i] = 0;
            F_up_band[x + nbin * i] = 0;
            F_down_band[x + nbin * i] = 0;

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            while(x < nbin && y < ny && i < numinterfaces){
                
                atomicAdd_double(&(F_dir_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_up_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_double(&(F_down_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            y += blockDim.y;
        }
        y = threadIdx.y;
        i += blockDim.z;
    }
    __syncthreads();
    
    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){
        
        while(x < nbin && y == 0 && i < numinterfaces){

            atomicAdd_double(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
            atomicAdd_double(&(F_down_tot[i]), (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x]);

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < 1 && i < numinterfaces){
        F_net[i] = F_up_tot[i] - F_down_tot[i];
        i += blockDim.z;
    }
}


// calculates the integrated upwards and downwards fluxes
__global__ void integrate_flux_single(
        float* deltalambda, 
        float* F_down_tot, 
        float* F_up_tot, 
        float* F_net, 
        float* F_down_wg, 
        float* F_up_wg,
        float* F_dir_wg,
        float* F_down_band, 
        float* F_up_band, 
        float* F_dir_band,
        float* gauss_weight,
        int 	nbin, 
        int 	numinterfaces, 
        int 	ny
){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            F_up_tot[i] = 0;
            F_down_tot[i] = 0;

            F_dir_band[x + nbin * i] = 0;
            F_up_band[x + nbin * i] = 0;
            F_down_band[x + nbin * i] = 0;

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y < ny && i < numinterfaces){

        while(x < nbin && y < ny && i < numinterfaces){

            while(x < nbin && y < ny && i < numinterfaces){

                atomicAdd_single(&(F_dir_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_dir_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_single(&(F_up_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_up_wg[y + ny * x + ny * nbin * i]);
                atomicAdd_single(&(F_down_band[x + nbin * i]), 0.5 * gauss_weight[y] * F_down_wg[y + ny * x + ny * nbin * i]);
                
                x += blockDim.x;
            }
            x = threadIdx.x;
            y += blockDim.y;
        }
        y = threadIdx.y;
        i += blockDim.z;	
    }
    __syncthreads();
    
    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){
        
        while(x < nbin && y == 0 && i < numinterfaces){

            atomicAdd_single(&(F_up_tot[i]), F_up_band[x + nbin * i] * deltalambda[x]);
            atomicAdd_single(&(F_down_tot[i]), (F_dir_band[x + nbin * i] + F_down_band[x + nbin * i]) * deltalambda[x]);

            x += blockDim.x;
        }
        x = threadIdx.x;
        i += blockDim.z;	
    }
    __syncthreads();

    i = threadIdx.z;

    while(x < nbin && y == 0 && i < numinterfaces){

        F_net[i] = F_up_tot[i] - F_down_tot[i];
        
        i += blockDim.z;
    }
}


// calculates the net fluxes and advances the layer temperatures
__global__ void rad_temp_iter(
        utype* F_down_tot, 
        utype* F_up_tot, 
        utype* F_net, 
        utype* F_net_diff, 
        utype* tlay, 
        utype* play,
        utype* tint, 
        utype* pint,
        int* 	abrt, 
        utype* T_store, 
        utype* deltat_prefactor,
        int 	itervalue, 
        utype 	f_factor, 
        int 	foreplay,
        utype 	tstep, 
        int 	numlayers, 
        int 	varydelta, 
        utype 	local_limit,
        int 	adapt_interval,
        int		smooth,
        int     dim,
        int     step
){

    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i < numlayers){

        // obtain constant timestep value
        utype delta_t = tstep;

        // net flux divergence for each layer
        F_net_diff[i] = F_net[i] - F_net[i+1];
        
        // tweaking points
        utype t_mid = tlay[i];
        
        if(smooth ==1){
            if(play[i] < 1e6 && i < numlayers -1){
                t_mid = (tlay[i-1]+tlay[i+1])/2.0;
            }
        }
        // temperature smoothing force -- dependent on the temperature displacement (power of 7 found to be best "middle" between cheating and energy conservation)
        utype F_temp = pow((t_mid - tlay[i]), 7.0);
        
        // net flux gradient -- combination of pure radiative net flux and temperature smoothing term
        utype combined_F_net_diff = F_net_diff[i] + F_temp;
        
        // if using varying timestep
        if(varydelta == 1){
            if (itervalue == foreplay){
                deltat_prefactor[i] = 1e0; // 1e0 found to be most stable. earlier value was 1e2.
            }

            if(combined_F_net_diff != 0){
                delta_t = deltat_prefactor[i] * play[i] / pow(abs(combined_F_net_diff), 0.9); // through tweaking 0.9 found to be most stable
            }
        }
        
        utype delta_T = combined_F_net_diff / (pint[i] - pint[i+1]) * delta_t;
        
        // limits large temperature jumps for increased stability
        if(abs(delta_T) > 500.0){
            delta_T = 500.0 * combined_F_net_diff/abs(combined_F_net_diff);
        }
                
        // adaptive timestepping
        if (varydelta == 1) {
            // store always last 6 or 20 entries of temp. change
            if (itervalue % adapt_interval == 0){
                T_store[i] = tlay[i];
            }

            if (itervalue % adapt_interval == adapt_interval-1) {
                if(abs(tlay[i] - T_store[i]) < adapt_interval / 2.0 * abs(delta_T)){
                    deltat_prefactor[i] /= 1.5; // 1.5 was found to lead to fastest convergence
                }
                else{
                    deltat_prefactor[i] *= 1.1; // 1.1 was found to lead to fastest convergence
                }
            }
        }
        
        // update layer temperatures
        tlay[i] = tlay[i] + delta_T;
        
        // prevent too low temperatures and too high temperatures
        // to decrease the runtime, the temperature are limited to max. 15'000 K or the maximum value of pre-tabulated BB values
        utype max_limit = min(15000.0, dim * step - 1.001);
        tlay[i] = min(max(tlay[i],1.001), max_limit); 

        // abort conditions
        // local radiative equilibrium
        bool condition1 = abs(combined_F_net_diff)/(STEFANBOLTZMANN*(tlay[i]*tlay[i]*tlay[i]*tlay[i])) < local_limit;
        // to prevent too high temperatures in the deep (i.e. radiative equilibrium is not necessary there)
        bool condition2 = tlay[i] > 0.9 * max_limit && play[i] > 1e6;
        
        if (condition1 || condition2){
            abrt[i] = 1;
        }
        else {
            abrt[i] = 0;
        }
    }
}


// advances the layer temperatures in the convection loop
__global__ void conv_temp_iter(
        utype* F_down_tot, 
        utype* F_up_tot, 
        utype* F_net, 
        utype* F_net_diff, 
        utype* tlay, 
        utype* play,
        utype* pint,
        utype* c_p_lay,
        utype* T_store, 
        utype* deltat_prefactor,
        int*   conv_layer,
        utype 	g, 
        int 	numlayers,
        int 	itervalue,
        int 	adapt_interval,
        int 	smooth
){

    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i < numlayers){

        //set constant timestep value
        utype delta_t;
        
        // set initial timestep prefactor
        if (itervalue == 0){
            deltat_prefactor[i] = 1e-3; // 1e-3 found to be most stable. previous value was 1e2.
        }
        if (itervalue == 6000){
            deltat_prefactor[i] = 1e-4; // cheap fix to unstuck calculations after convective stitching
        }
        
        // net flux divergence for each layer
        F_net_diff[i] = F_net[i] - F_net[i+1];
        
        // radiative timescale
        //utype t_rad = c_p_lay[i] * play[i] / (STEFANBOLTZMANN * g * (tlay[i]*tlay[i]*tlay[i])); // OLD approach -- still here for debugging reasons
        utype t_rad = c_p_lay[i] * play[i] / g;

        // tweaking points (completely analogous to the radiative iteration tweaking)
        utype t_mid = tlay[i];

        if(smooth ==1){
            if(play[i] < 1e6 && i < numlayers -1){
                t_mid = (tlay[i-1]+tlay[i+1])/2.0;
            }
        }
        // temperature smoothing force -- dependent on the temperature displacement (power of 7 found to be best "middle" between smoothed temperatures and energy conservation)
        utype F_temp = pow((t_mid - tlay[i]), 7.0);

        // net flux gradient -- combination of pure radiative net flux and temperature smoothing term
        utype combined_F_net_diff =  F_net_diff[i] + F_temp;

        if(combined_F_net_diff != 0){
            delta_t = deltat_prefactor[i] * t_rad / pow(abs(combined_F_net_diff), 0.5); // 0.5 was found to be most stable for the radiative-convective interplay
        }
        
        utype delta_T = g / c_p_lay[i] * combined_F_net_diff / (pint[i] - pint[i+1]) * delta_t;
        
        // limits large temperature jumps for increased stability
        if(abs(delta_T) > 20.0){
            delta_T = 20.0 * combined_F_net_diff/abs(combined_F_net_diff);
        }

        // store last few entries of temp. change
        if (itervalue % adapt_interval == 0){
            T_store[i] = tlay[i];
        }

        if (itervalue % adapt_interval == adapt_interval-1) {
            if(abs(tlay[i] - T_store[i]) < adapt_interval / 2.0 * abs(delta_T)){
                deltat_prefactor[i] /= 1.5; // 1.5 was found to lead to fastest convergence
            }
            else{
                deltat_prefactor[i] *= 1.1; // 1.1 was found to lead to fastest convergence
            }
        }
        
        // update layer temperatures
        tlay[i] = tlay[i] + delta_T;
        
        // prevent too low temperatures
        tlay[i] = max(tlay[i],1.0);
    }
}


// integrates the transmission function for each wavelength bin
__global__ void integrate_optdepth_transmission_iso(
        utype* trans_wg,
        utype* trans_band,
        utype* delta_tau_wg,
        utype* delta_tau_band,
        utype* gauss_weight,  
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < nbin && i < nlayer){

        delta_tau_band[x+nbin*i] = 0;
        trans_band[x+nbin*i] = 0;

        for(int y=0;y<ny;y++){
            delta_tau_band[x+nbin*i] += 0.5 * gauss_weight[y] * delta_tau_wg[y+ny*x + ny*nbin*i];
            trans_band[x+nbin*i] += 0.5 * gauss_weight[y] * trans_wg[y+ny*x + ny*nbin*i];
        }
    }
}


// integrates the transmission function for each wavelength bin
__global__ void integrate_optdepth_transmission_noniso(
        utype* trans_wg_upper,
        utype* trans_wg_lower,
        utype* trans_band,
        utype* delta_tau_wg_upper,
        utype* delta_tau_wg_lower,
        utype* delta_tau_band,
        utype* gauss_weight,  
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;

    if (x < nbin && i < nlayer){

        delta_tau_band[x+nbin*i] = 0;
        trans_band[x+nbin*i] = 0;

        for(int y=0;y<ny;y++){
            delta_tau_band[x+nbin*i] += 0.5 * gauss_weight[y] * (delta_tau_wg_upper[y+ny*x + ny*nbin*i] + delta_tau_wg_lower[y+ny*x + ny*nbin*i]);
            trans_band[x+nbin*i] += 0.5 * gauss_weight[y] * (trans_wg_upper[y+ny*x + ny*nbin*i] * trans_wg_lower[y+ny*x + ny*nbin*i]);
        }
    }
}


// calculates the contribution function
__global__ void calc_contr_func_iso(
        utype* trans_wg,
        utype* trans_weight_band,
        utype* contr_func_band,
        utype* gauss_weight,
        utype* planckband_lay,
        utype 	epsi,
        int     nbin, 
        int     nlayer, 
        int     ny
){

  int x = threadIdx.x + blockIdx.x*blockDim.x; // wavelength bin
  int i = threadIdx.y + blockIdx.y*blockDim.y; // layer
  
  if (x < nbin && i < nlayer){
    
    utype *trans_weight_wg;
    trans_weight_wg = new utype[ny];
    
    for(int y=0;y<ny;y++){  // loop on gaussian point
      
      utype trans_to_top = 1.0;
            
      for (int j = i+1; j < nlayer; j++){
	trans_to_top = trans_to_top * trans_wg[y+ny*x+ny*nbin*j];
      }
      
      trans_weight_wg[y] = (1.0 - trans_wg[y+ny*x+ny*nbin*i]) * trans_to_top;
      
      trans_weight_band[x+nbin*i] += 0.5 * gauss_weight[y] * trans_weight_wg[y];
    }
    contr_func_band[x+nbin*i] = 2.0 * PI * epsi * planckband_lay[i+x*(nlayer+2)] * trans_weight_band[x+nbin*i];
    
    delete [] trans_weight_wg;
  }
}


// calculates the contribution function, non-isothermal version
__global__ void calc_contr_func_noniso(
        utype* trans_wg_upper,
        utype* trans_wg_lower,
        utype* trans_weight_band,
        utype* contr_func_band,
        utype* gauss_weight,
        utype* planckband_lay,
        utype 	epsi,
        int     nbin, 
        int     nlayer, 
        int     ny
){

    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int i = threadIdx.y + blockIdx.y*blockDim.y;
    
    if (x < nbin && i < nlayer){

        utype *trans_weight_wg;
        trans_weight_wg = new utype[ny];
        
        for(int y=0;y<ny;y++){

            utype trans_to_top = 1.0;
            
            for (int j = i+1; j < nlayer; j++){
                trans_to_top = trans_to_top * trans_wg_upper[y+ny*x+ny*nbin*j] * trans_wg_lower[y+ny*x+ny*nbin*j];
            }

            trans_weight_wg[y] = (1.0 - trans_wg_upper[y+ny*x+ny*nbin*i] * trans_wg_lower[y+ny*x+ny*nbin*i]) * trans_to_top;

            trans_weight_band[x+nbin*i] += 0.5 * gauss_weight[y] * trans_weight_wg[y];
        }
        contr_func_band[x+nbin*i] = 2.0 * PI * epsi * planckband_lay[i+x*(nlayer+2)] * trans_weight_band[x+nbin*i];
        
        delete [] trans_weight_wg;
    }
}


// calculates the Planck and Rosseland mean opacities for each layer
__global__ void calc_mean_opacities(
				    utype* planck_opac_T_pl, // out
				    utype* ross_opac_T_pl, // out
				    utype* planck_opac_T_star, // out
				    utype* ross_opac_T_star, // out
				    utype* opac_wg_lay, // in
				    utype* cloud_opac_lay,// in
				    utype* planckband_lay, // in
				    utype* opac_interwave, // in
				    utype* opac_deltawave, // in
				    utype* T_lay, // in
				    utype* gauss_weight, 
				    utype* opac_y,
				    utype* opac_band_lay, // out
				    int 	nlayer, 
				    int 	nbin, 
				    int 	ny, 
				    utype 	T_star
				    ){

    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if(i < nlayer){

        utype num_planck_T_pl = 0;
        utype denom_planck_T_pl = 0;
        utype num_ross_T_pl = 0;
        utype denom_ross_T_pl = 0;
        utype num_planck_T_star = 0;
        utype denom_planck_T_star = 0;
        utype num_ross_T_star = 0;
        utype denom_ross_T_star = 0;

        // integrates opacity over each bin with Gaussian quadrature
        for(int x=0;x<nbin;x++){

            opac_band_lay[x+nbin*i] = 0 + cloud_opac_lay[i];

            for (int y=0;y<ny;y++){
                opac_band_lay[x+nbin*i] += 0.5 * gauss_weight[y] * opac_wg_lay[y+ny*x+ny*nbin*i];
            }
        }

        for (int x = 0; x < nbin; x++) {

            // calculates Planck mean opacity with layer temperatures
            num_planck_T_pl += opac_band_lay[x+nbin*i] * planckband_lay[i+x*(nlayer+2)]*opac_deltawave[x];
            denom_planck_T_pl += planckband_lay[i+x*(nlayer+2)]*opac_deltawave[x];

            // calculates Rosseland mean opacity with layer temperatures
            num_ross_T_pl += integrated_dB_dT(gauss_weight,opac_y,ny,opac_interwave[x],opac_interwave[x+1],T_lay[i]);

            denom_ross_T_pl += integrated_dB_dT(gauss_weight,opac_y,ny,opac_interwave[x],opac_interwave[x+1],T_lay[i]) / opac_band_lay[x+nbin*i];

            // calculates Planck mean opacity with stellar blackbody function
            num_planck_T_star += opac_band_lay[x+nbin*i] * planckband_lay[nlayer+x*(nlayer+2)]*opac_deltawave[x];
            denom_planck_T_star += planckband_lay[nlayer+x*(nlayer+2)]*opac_deltawave[x];

            // calculates Rosseland mean opacity with stellar blackbody function
            num_ross_T_star += integrated_dB_dT(gauss_weight, opac_y, ny, opac_interwave[x],opac_interwave[x+1],T_star);
            
            denom_ross_T_star += integrated_dB_dT(gauss_weight, opac_y, ny, opac_interwave[x],opac_interwave[x+1],T_star) / opac_band_lay[x+nbin*i];
        }

        planck_opac_T_pl[i] = num_planck_T_pl / denom_planck_T_pl;
        ross_opac_T_pl[i] = num_ross_T_pl / denom_ross_T_pl;
        
        // for T < 70, K dB_dT is too small to be calculated numerically at short wavelengths
        if(T_lay[i] < 70){
            ross_opac_T_pl[i] = -3;      
        }
        
        planck_opac_T_star[i] = num_planck_T_star / denom_planck_T_star;
        ross_opac_T_star[i] = num_ross_T_star / denom_ross_T_star;
        
        // same as above for star with 70 K. Does not make any sense anyway, such a cool star, 
        // hence also planck calculation prohibited.
        if (T_star < 70){
            planck_opac_T_star[i] = -3;
            ross_opac_T_star[i] = -3;         
        }
    }
}


// integrate the direct beam flux
__global__ void integrate_beamflux(
        utype* F_dir_tot, 
        utype* F_dir_band,
        utype* deltalambda, 
        utype* gauss_weight, 
        int 	nbin, 
        int 	numinterfaces
){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if(i < numinterfaces){

        F_dir_tot[i] = 0;

        for (int x = 0; x < nbin; x++) {

            // sum the bin contributions to obtain the integrated flux
            F_dir_tot[i] += F_dir_band[x + nbin * i] * deltalambda[x];
        }
    }
}
