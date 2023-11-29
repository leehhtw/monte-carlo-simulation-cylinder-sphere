//
//  main.cpp
//  diffusion_cylinder_exchange
//
//  Update Journal:
//  -- 6/26/2017: massive job version
//  -- 7/20/2017: do not divide DW signal & cumulants by b0 signal, record particle number in each compartment
//  -- 4/27/2019: diffusion in spheres with permeable membrane
//  -- 5/1/2019: implement cuda
//  -- 6/18/2019: re-write sphere code to cylinder code for cuda
//  -- 2/20/2020: re-write coaxial cylinder code to single-layer cylinder code for cuda
//  -- 3/6/2020: fix the bug for permeation step along z-axis
//  -- 4/23/2020: replace pow with pow2, turn on translateFlag
//
//  Created by Hong-Hsi Lee in February, 2017.
//


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <iomanip>
#include <time.h>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <complex>
#include <string>

#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

using namespace std;
    
#define Pi 3.14159265
#define timepoints 1000
#define nbin 200
#define nite 100

// ********** diffusion library **********

__device__ double pow2 (const double &x) {
    return (x*x);
}

__device__ void pixPosition ( const double x_in[], const unsigned int &NPix, int xPix[] ) {
    double x[2]={0}; x[0]=x_in[0]; x[1]=x_in[1]; //x[2]=x_in[2];
    
    if ( x[0]<0 ) { x[0]+=1; }
    if ( x[0]>1 ) { x[0]-=1; }
    
    if ( x[1]<0 ) { x[1]+=1; }
    if ( x[1]>1 ) { x[1]-=1; }
    
//    if ( x[2]<0 ) { x[2]+=1; }
//    if ( x[2]>1 ) { x[2]-=1; }
    
    xPix[0]=floor(x[0]*NPix);
    xPix[1]=floor(x[1]*NPix);
//    xPix[2]=floor(x[2]*NPix);
}

__device__ void translateXc ( const double x[], double xc[] ) {
    // Translate circle center xc to make it as close to the position x as possible
    double ti=0, tj=0;
    double d2 = pow2(x[0]-xc[0])+pow2(x[1]-xc[1]), d2Tmp=0;
    int ii[2]={0}, jj[2]={0};
    ii[1]=2*(xc[0]<0.5)-1;
    jj[1]=2*(xc[1]<0.5)-1;
    for (int i=0; i<2; i++) {
        for (int j=0; j<2; j++) {
            if ( i==0 & j==0 ){ continue; }
            d2Tmp=pow2(x[0]-xc[0]-ii[i])+pow2(x[1]-xc[1]-jj[j]);
            if (d2Tmp<d2) {
                d2=d2Tmp;
                ti=ii[i];
                tj=jj[j];
            }
        }
    }
    xc[0]+=ti;
    xc[1]+=tj;
}

__device__ bool inCyl ( const double x[], const double xc_in[], const double &rc, const bool &translateFlag ) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // If the point x is in the circle (xc,rc), return 1; if not, return 0.
    
    // Translate circle center xc to make it as close to the position xt as possible
    if ( translateFlag ) { translateXc(x,xc); }
    
    return ( ( pow2(x[0]-xc[0])+pow2(x[1]-xc[1]) ) <= rc*rc );
}

__device__ bool stepE2A (const double xi[], const double xt[], const double xc_in[], const double &rc, double t[], const double &dx, const bool &translateFlag) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // If segment(xi,xt) overlaps circle (xc,rc), return 1; if not, return 0.
    
    // Translate circle center xc to make it as close to the position xt as possible
    if ( translateFlag ) { translateXc(xt,xc); }
    
    t[0]=-( (xi[0]-xc[0])*(xt[0]-xi[0])+(xi[1]-xc[1])*(xt[1]-xi[1]) )/dx/dx;
    
    // If xt is in the cell, segment overlaps the circle.
    if ( ( pow2(xt[0]-xc[0])+pow2(xt[1]-xc[1]) ) <= rc*rc ) {
        return true;
    } else {
        // L: a line connecting xi and xt
        // xl: a point on L closest to xc, xl = xi + (xt-xi)*t
        // d: distance of xc to L (or xl)
        // Reference: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
        double xl[2]={0};
        xl[0]=xi[0]+(xt[0]-xi[0])*t[0];
        xl[1]=xi[1]+(xt[1]-xi[1])*t[0];
        double d2=pow2(xl[0]-xc[0])+pow2(xl[1]-xc[1]);
        
        // If d>rc, segment does not overlap the circle.
        if (d2>rc*rc) {
            return false;
        } else {
            // xl is in ICS, but xi and xt are both in ECS.
            return ( ( (xi[0]-xl[0])*(xt[0]-xl[0])+(xi[1]-xl[1])*(xt[1]-xl[1]) ) <= 0 );
        }
    }
}

__device__ void elasticECS (const double x[], const double v[], const double &dx, const double &dz, const double xc_in[], const double &rc, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Elastic collision from x in ECS onto a cell membrane (xc,rc) with a direction v and a step size dx.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dx*v[0];
    xTmp[1]=x[1]+dx*v[1];
    xTmp[2]=x[2]+dz*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;
    
    
    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0}, n[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                    // Does not encounter the cell membrane
        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=min(t1,t2);
        if ( (t>=dx) | (t<0) ) {        // Does encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        } else {                          // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // n parallel to (xm-xc), outward unit normal vector
            t1=sqrt( pow2(xc[0]-xm[0])+pow2(xc[1]-xm[1]) );
            n[0]=(xm[0]-xc[0])/t1;
            n[1]=(xm[1]-xc[1])/t1;
            
            // v' = v - 2*dot(v,n)*n
            t1=v[0]*n[0]+v[1]*n[1];
            n[0]=v[0]-2*t1*n[0];
            n[1]=v[1]-2*t1*n[1];
            
            // xt = xm + (dx-t)*v'
            xt[0]=xm[0]+(dx-t)*n[0];
            xt[1]=xm[1]+(dx-t)*n[1];
            xt[2]=xTmp[2];
        }
    }
}

__device__ void permeateE2I (const double x[], const double v[], const double &dxEX, const double &dzEX, const double xc_in[], const double &rc, const double &dxIN, const double &dzIN, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Permeation from x in ECS into a cell (xc,rc) with a direction v and a step size dxEX.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dxEX*v[0];
    xTmp[1]=x[1]+dxEX*v[1];
    xTmp[2]=x[2]+dzEX*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;
    
    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0}, n[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                    // Does not encounter the cell membrane
        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=min(t1,t2);
        if ( (t>=dxEX) | (t<0) ) {      // Does encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        } else {                          // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // n parallel to (xc-xm), inward unit normal vector
            t1=sqrt( pow2(xc[0]-xm[0])+pow2(xc[1]-xm[1]) );
            n[0]=(xc[0]-xm[0])/t1;
            n[1]=(xc[1]-xm[1])/t1;
            
            // Diffuse transverse to the cell membrane after permeation to make the result converge faster.
            // xt = xm + n*(1-t/dxEX)*dxIN*dot(v,n)
            t1=fabs(v[0]*n[0]+v[1]*n[1]);
            xt[0]=xm[0]+n[0]*(1-t/dxEX)*dxIN*t1;
            xt[1]=xm[1]+n[1]*(1-t/dxEX)*dxIN*t1;
            // xt = t/dxEX*dzEX*vz + (1-t/dxEX)*dzIN*vz
            xt[2]=x[2] + t/dxEX*dzEX*v[2] + (1-t/dxEX)*dzIN*v[2];
        }
    }
}

__device__ void elasticICS (const double x[], const double v[], const double &dx, const double &dz, const double xc_in[], const double &rc, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Elastic collision from x in ICS onto a cell membrane (xc,rc) with a direction v and a step size dx.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dx*v[0];
    xTmp[1]=x[1]+dx*v[1];
    xTmp[2]=x[2]+dz*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;

    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0}, n[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                     // Walker is right on the surface and diffuses tangent to the surface
//        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        xt[0]=x[0]; xt[1]=x[1]; xt[2]=xTmp[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=max(t1,t2);
        if ( t>=dx ) {                  // Does not encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        }
        else {                          // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // n parallel to (xm-xc), outward unit normal vector
            t1=sqrt( pow2(xc[0]-xm[0])+pow2(xc[1]-xm[1]) );
            n[0]=(xm[0]-xc[0])/t1;
            n[1]=(xm[1]-xc[1])/t1;
            
            // v' = v - 2*dot(v,n)*n
            t1=v[0]*n[0]+v[1]*n[1];
            n[0]=v[0]-2*t1*n[0];
            n[1]=v[1]-2*t1*n[1];
            
            // xt = xm + (dx-t)*v'
            xt[0]=xm[0]+(dx-t)*n[0];
            xt[1]=xm[1]+(dx-t)*n[1];
            xt[2]=xTmp[2];
        }
    }
}

__device__ void permeateI2E (const double x[], const double v[], const double &dxIN, const double &dzIN, const double xc_in[], const double &rc, const double &dxEX, const double &dzEX, const bool &translateFlag, double xt[]) {
    double xc[2]={0}; xc[0]=xc_in[0]; xc[1]=xc_in[1];
    // Permeation from x in ICS out of the cell (xc,rc) with a direction v and a step size dxIN.
    
    // Translate circle center xc to make it as close to the position (x + dx*v) as possible
    double xTmp[3]={0};
    xTmp[0]=x[0]+dxIN*v[0];
    xTmp[1]=x[1]+dxIN*v[1];
    xTmp[2]=x[2]+dzIN*v[2];
    if ( translateFlag ) { translateXc(xTmp,xc); }
    
    // distance( x+t*v, xc )==rc, solve t
    double a=0,b=0,c=0,t1=0,t2=0,t=0;
    a=v[0]*v[0] + v[1]*v[1];
    b=2*(x[0]-xc[0])*v[0] + 2*(x[1]-xc[1])*v[1];
    c=pow2(x[0]-xc[0]) + pow2(x[1]-xc[1]) - rc*rc;
    
    // xt: final position, xm: contact point on cell membrane, n: unit normal vector
    // discri: discriminant
    double xm[2]={0}, n[2]={0};
    double discri=b*b-4*a*c;
    if (discri<=0) {                     // Walker is right on the surface and diffuses tangent to the surface
        xt[0]=x[0]; xt[1]=x[1]; xt[2]=x[2];
    } else {
        discri=sqrt(discri);
        t1=0.5/a*( -b+discri );
        t2=0.5/a*( -b-discri );
        t=max(t1,t2);
        if ( t>=dxIN ) {                 // Does not encounter the cell membrane
            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
        } else {                         // Encounter the cell membrane
            // xm = x + t*v;
            xm[0]=x[0]+t*v[0];
            xm[1]=x[1]+t*v[1];
            
            // n parallel to (xm-xc), outward unit normal vector
            t1=sqrt( pow2(xc[0]-xm[0])+pow2(xc[1]-xm[1]) );
            n[0]=(xm[0]-xc[0])/t1;
            n[1]=(xm[1]-xc[1])/t1;
            
            // Diffuse perpendicular to the cell membrane after permeation to make the result converge faster.
            // xt = xm + n*(1-t/dxIN)*dxEX*dot(v,n)
            t1=fabs(v[0]*n[0]+v[1]*n[1]);
            xt[0]=xm[0]+n[0]*(1-t/dxIN)*dxEX*t1;
            xt[1]=xm[1]+n[1]*(1-t/dxIN)*dxEX*t1;
            // xt = t/dxIN*dzIN*vz + (1-t/dxIN)*dzEX*vz
            xt[2]=x[2] + t/dxIN*dzIN*v[2] + (1-t/dxIN)*dzEX*v[2];
        }
    }
}

// ********** cuda kernel **********
__device__ double atomAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
        
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

__global__ void setup_kernel(curandStatePhilox4_32_10_t *state, unsigned long seed){
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void propagate(curandStatePhilox4_32_10_t *state, double *dx2, double *dx4, double *NParICS, double *NParBin, double *sig, const int TN, const int NPar, const int Nbvec, const double res, const double stepIN, const double stepEX, const double stepINz, const double stepEXz, const double probI, const double probE, const unsigned int NPix, const unsigned int Nmax, const int initFlag, const double *xCir, const double *yCir, const double *rCir, const bool *translateFlag, const unsigned int *APix, const double *btab, const double *TD){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandStatePhilox4_32_10_t localstate=state[idx];
    
    int Tstep=TN/timepoints;
    
    for (int k=idx; k<NPar; k+=stride){
        double xPar[3]={0}, xCirTmp[3]={0};
        int xParG[2]={0};                                   // Particle position on a grid
        
        bool instruction1=false, instruction2=false;
        unsigned int a=0, aTmp=0;                           // Element of APix matrix
        int a1=0, a2=0;                                     // Label of cells close to the particle
        int acell[4]={0}; bool instruction[4]={0};          // Cell label
        double tcell[4]={0};                                // xi+(xt-xi)*tcell is a point on segment(xi,xt) closest to the cell center
        double tcellTmp[1]={0};
        double tcellMin=0;
        
        double xi[3]={0}, xt[3]={0}, xTmp[3]={0};           // Particle position
        double xCollision[3]={0};                           // Position after collision
        int xtG[2]={0}, xTmpG[2]={0};                       // Position on grid after diffusion
        double vrand=0;                                     // Random number
        int tidx=0, bidx=0;
        
        double vp[3]={0};                                   // Nomalized diffusion velocity
        int acell_hit=0;                                    // Label of the cell encountered by the walker
        bool iterateFlag=false;                             // true: choose another direction and leap again, false: finish the iteration
        int ite=0;                                          // number of iterations
        bool ICSFlag=false;                                 // true: in ICS, false: not in ICS
        
        int xjmp=0, yjmp=0, zjmp=0;
        double dx=0, dy=0, dz=0;
        
        double qx=0;
        
        //********** Initialize Walker Positions *********
        while (1){
            xPar[0]=curand_uniform_double(&localstate);
            xPar[1]=curand_uniform_double(&localstate);
            xPar[2]=curand_uniform_double(&localstate);
            
            if ( initFlag==1 ) { // 1. Initial positon: ICS
                // Identify the cells close to the walker
                pixPosition(xPar,NPix,xParG);
                a=APix[ NPix*xParG[0]+xParG[1] ];
                a1=a%Nmax; a2=a/Nmax;
                
                // If the walker is in ICS, take the initial position
                instruction1=false; instruction2=false;
                if ( a1 ){
                    xCirTmp[0]=xCir[a1-1]; xCirTmp[1]=yCir[a1-1];
                    instruction1=inCyl(xPar,xCirTmp,rCir[a1-1],translateFlag[a1-1]);
                }
                
                if ( a2 ){
                    xCirTmp[0]=xCir[a2-1]; xCirTmp[1]=yCir[a2-1];
                    instruction2=inCyl(xPar,xCirTmp,rCir[a2-1],translateFlag[a2-1]);
                }
                
                if ( instruction1 || instruction2 ){
                    xi[0]=xPar[0]; xi[1]=xPar[1]; xi[2]=xPar[2];
                    break;
                }
            } else if ( initFlag==2 ) { // 2. Initial positon: ECS
                // Identify the cells close to the walker
                pixPosition(xPar,NPix,xParG);
                a=APix[ NPix*xParG[0]+xParG[1] ];
                a1=a%Nmax; a2=a/Nmax;
                
                // If the walker is not in Axon (ICS+myelin), take the initial position
                instruction1=false; instruction2=false;
                if ( a1 ){
                    xCirTmp[0]=xCir[a1-1]; xCirTmp[1]=yCir[a1-1];
                    instruction1=inCyl(xPar,xCirTmp,rCir[a1-1],translateFlag[a1-1]);
                }
                
                if ( a2 ){
                    xCirTmp[0]=xCir[a2-1]; xCirTmp[1]=yCir[a2-1];
                    instruction2=inCyl(xPar,xCirTmp,rCir[a2-1],translateFlag[a2-1]);
                }
                
                if ( instruction1==false && instruction2==false ){
                    xi[0]=xPar[0]; xi[1]=xPar[1]; xi[2]=xPar[2];
                    break;
                }
            } else if ( initFlag==3 ) { // 3. Initial position: ICS+ECS
                xi[0]=xPar[0]; xi[1]=xPar[1]; xi[2]=xPar[2];
                break;
            } else if ( initFlag==4 ) { // 4. Initial position: center
                xi[0]=0.5; xi[1]=0.5; xi[2]=0.5;
                break;
            }
        }
        
        // ********** Simulate diffusion **********
        xt[0]=xi[0]; xt[1]=xi[1]; xt[2]=xi[2];
        pixPosition(xt,NPix,xtG);                             // Position on grid
        for (int i=0; i<TN; i++){
            // The cells close to the walker in the previous step
            a=APix[ NPix*xtG[0]+xtG[1] ];
            a1=a%Nmax, a2=a/Nmax;
            
            // Check if the particle is in axon
            instruction1=false; instruction2=false;
            if ( a1 ) {
                xCirTmp[0]=xCir[a1-1]; xCirTmp[1]=yCir[a1-1];
                instruction1=inCyl(xt,xCirTmp,rCir[a1-1],translateFlag[a1-1]);
            }
            
            if ( a2 ) {
                xCirTmp[0]=xCir[a2-1]; xCirTmp[1]=yCir[a2-1];
                instruction2=inCyl(xt,xCirTmp,rCir[a2-1],translateFlag[a2-1]);
            }
            
            ICSFlag=false;
            if (instruction1 | instruction2) { ICSFlag=true; }
            
            iterateFlag=true; ite=0;
            
            // ********** One step **********
            while (iterateFlag & (ite<nite)) {
                if ( (instruction1==false) & (instruction2==false) ) {
                    // Case 1 Diffusion In ECS
                    
                    acell[0]=a1; acell[1]=a2;
                    
                    // Primitive position after diffusion
                    vrand=curand_uniform_double(&localstate);
                    vp[0]=cos(2*Pi*vrand);
                    vp[1]=sin(2*Pi*vrand);
                    vrand=curand_uniform_double(&localstate);
                    vp[2]=2.0*static_cast<double>(vrand<0.5)-1.0;
                    xTmp[0]=xt[0]+stepEX*vp[0];
                    xTmp[1]=xt[1]+stepEX*vp[1];
                    xTmp[2]=xt[2]+stepEXz*vp[2];
                    
                    pixPosition(xTmp,NPix,xTmpG);
                    aTmp=APix[ NPix*xTmpG[0]+xTmpG[1] ];
                    acell[2]=aTmp%Nmax; acell[3]=aTmp/Nmax;
                    
                    // Check if the segment(xt,xTmp) overlaps with any cell
                    for (int j=0; j<4; j++) {
                        instruction[j]=false; tcell[j]=-1;
                        if ( acell[j] ) {
                            xCirTmp[0]=xCir[acell[j]-1];
                            xCirTmp[1]=yCir[acell[j]-1];
                            instruction[j]=stepE2A(xt, xTmp, xCirTmp, rCir[acell[j]-1], tcellTmp, stepEX, translateFlag[acell[j]-1]);
                            tcell[j]=tcellTmp[0];
                        }
                    }
                    if ( (instruction[0]==false) & (instruction[1]==false) & (instruction[2]==false) & (instruction[3]==false) ) {
                        // Case 1.1 Walker diffuses in ECS and does not encounter any cell membrane.
                        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                        iterateFlag=false; ICSFlag=false;
                    } else {
                        // Case 1.2 Walker diffuses in ECS and encounters the cell membrane.
                        
                        // Determine the cell to collide with.
                        tcellMin=-1;
                        for (int j=0; j<4; j++) {
                            if ( instruction[j] & (tcell[j]>=0) ) {
                                tcellMin=tcell[j];
                            }
                        }
                        if ( tcellMin<0 ){
                            printf("error: Walker in ECS does not encounter the cell membrane.\n");
                            printf("%d %d %d %d\n",acell[0],acell[1],acell[2],acell[3]);
                            printf("%s %s %s %s\n",instruction[0] ? "true":"false",instruction[1] ? "true":"false",instruction[2] ? "true":"false",instruction[3] ? "true":"false");
                            printf("%.4f %.4f %.4f %.4f\n",tcell[0],tcell[1],tcell[2],tcell[3]);
                        }
                        
                        for (int j=0; j<4; j++) {
                            if ( instruction[j] & (tcell[j]>=0) ) {
                                if (tcell[j]<=tcellMin){
                                    tcellMin=tcell[j];
                                    acell_hit=acell[j];
                                }
                            }
                        }
                        
                        xCirTmp[0]=xCir[acell_hit-1];
                        xCirTmp[1]=yCir[acell_hit-1];
                        
                        vrand=curand_uniform_double(&localstate);
                        if (vrand<probE) {
                            // Case 1.2.1 Permeation from ECS to ICS
                            permeateE2I(xt, vp, stepEX, stepEXz, xCirTmp, rCir[acell_hit-1], stepIN, stepINz, translateFlag[acell_hit-1], xTmp);
                            xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                            iterateFlag=false; ICSFlag=true;
                        }
                        else {
                            // Case 1.2.2 Elastic collision in ECS
                            elasticECS(xt, vp, stepEX, stepEXz, xCirTmp, rCir[acell_hit-1], translateFlag[acell_hit-1], xCollision);
                            
                            // Use xTmp to save the present position
                            xTmp[0]=xt[0]; xTmp[1]=xt[1]; xTmp[2]=xt[2];
                            
                            // Case 1.2.2.1 Renew the step for the elastic collision
                            xt[0]=xCollision[0]; xt[1]=xCollision[1]; xt[2]=xCollision[2];
                            iterateFlag=false;
                            
                            // Case 1.2.2.2 Cancel this step and choose another direction if bouncing twice
                            pixPosition(xCollision, NPix, xTmpG);
                            aTmp=APix[ NPix*xTmpG[0]+xTmpG[1] ];
                            acell[2]=aTmp%Nmax, acell[3]=aTmp/Nmax;
                            
                            if ( acell[2] ) {
                                xCirTmp[0]=xCir[acell[2]-1];
                                xCirTmp[1]=yCir[acell[2]-1];
                                if ( inCyl(xCollision, xCirTmp, rCir[acell[2]-1], translateFlag[acell[2]-1]) ) {
                                    xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                                    iterateFlag=true; ite++;
                                }
                            }
                            
                            if ( acell[3] ) {
                                xCirTmp[0]=xCir[acell[3]-1];
                                xCirTmp[1]=yCir[acell[3]-1];
                                if ( inCyl(xCollision, xCirTmp, rCir[acell[3]-1], translateFlag[acell[3]-1]) ) {
                                    xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                                    iterateFlag=true; ite++;
                                }
                            }
                            
                            if ( iterateFlag==false ) {
                                ICSFlag=false;
                            }
                        }
                    }
                } else {
                    // Case 2 Diffusion in ICS
                    if ( instruction1 ){
                        acell[0]=a1;
                    } else if ( instruction2 ){
                        acell[0]=a2;
                    } else {
                        acell[0]=0;
                        printf("error: Walker in ICS has no cell label.\n");
                    }
                    
                    // Primitive position after diffusion
                    vrand=curand_uniform_double(&localstate);
                    vp[0]=cos(2*Pi*vrand);
                    vp[1]=sin(2*Pi*vrand);
                    vrand=curand_uniform_double(&localstate);
                    vp[2]=2.0*static_cast<double>(vrand<0.5)-1;
                    xTmp[0]=xt[0]+stepIN*vp[0];
                    xTmp[1]=xt[1]+stepIN*vp[1];
                    xTmp[2]=xt[2]+stepINz*vp[2];
                    
//                     pixPosition(xTmp,NPix,xTmpG);
//                     aTmp=APix[ NPix*xTmpG[0]+xTmpG[1] ];
//                     acell[2]=aTmp%Nmax; acell[3]=aTmp/Nmax;
                    
                    // Check if the segment(xt,xTmp) overlaps with the cell membrane
                    xCirTmp[0]=xCir[acell[0]-1];
                    xCirTmp[1]=yCir[acell[0]-1];
                    instruction[0]=inCyl(xTmp,xCirTmp,rCir[acell[0]-1],translateFlag[acell[0]-1]);
                    
                    if ( instruction[0] ) {
                        // Case 2.1 Walker diffuses in ICS and does not encounter the cell membrane
                        xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                        iterateFlag=false; ICSFlag=true;
                    } else {
                        // Case 2.2 Walker diffuses in ICS and encounters the cell membrane
                        acell_hit=acell[0];
                        xCirTmp[0]=xCir[acell_hit-1];
                        xCirTmp[1]=yCir[acell_hit-1];
                        
                        vrand=curand_uniform_double(&localstate);
                        if (vrand<probI) {
                            // Case 2.2.1 Permeation from ICS to ECS
                            permeateI2E(xt, vp, stepIN, stepINz, xCirTmp, rCir[acell_hit-1], stepEX, stepEXz, translateFlag[acell_hit-1], xCollision);
                            
                            // Use xTmp to save the present position
                            xTmp[0]=xt[0]; xTmp[1]=xt[1]; xTmp[2]=xt[2];
                            
                            // Case 2.2.1.1 Renew the step for the permeation
                            xt[0]=xCollision[0]; xt[1]=xCollision[1]; xt[2]=xCollision[2];
                            iterateFlag=false;
                            
                            // Case 2.2.1.2 Cancel this step and choose another direction if steping into another axon
                            pixPosition(xCollision, NPix, xTmpG);
                            aTmp=APix[ NPix*xTmpG[0]+xTmpG[1] ];
                            acell[2]=aTmp%Nmax; acell[3]=aTmp/Nmax;

                            if ( acell[2] ) {
                                xCirTmp[0]=xCir[acell[2]-1];
                                xCirTmp[1]=yCir[acell[2]-1];
                                if ( inCyl(xCollision, xCirTmp, rCir[acell[2]-1], translateFlag[acell[2]-1]) ) {
                                    xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                                    iterateFlag=true; ite++;
                                }
                            }
                            
                            if ( acell[3] ) {
                                xCirTmp[0]=xCir[acell[3]-1];
                                xCirTmp[1]=yCir[acell[3]-1];
                                if ( inCyl(xCollision, xCirTmp, rCir[acell[3]-1], translateFlag[acell[3]-1]) ) {
                                    xt[0]=xTmp[0]; xt[1]=xTmp[1]; xt[2]=xTmp[2];
                                    iterateFlag=true; ite++;
                                }
                            }
                            
                            if ( iterateFlag==false ){
                                ICSFlag=false;
                            }
                        }
                        else {
                            // Case 2.2.2 Elastic collision in ICS
                            elasticICS(xt, vp, stepIN, stepINz, xCirTmp, rCir[acell_hit-1], translateFlag[acell_hit-1], xCollision);
                            
                            if ( inCyl(xCollision, xCirTmp, rCir[acell_hit-1], translateFlag[acell_hit-1]) ) {
                                // Case 2.2.2.1 Renew the step for the elastic collision
                                xt[0]=xCollision[0]; xt[1]=xCollision[1]; xt[2]=xCollision[2];
                                iterateFlag=false; ICSFlag=true;
                            } else {
                                // Case 2.2.2.2 Cancel this step and choose another direction if bouncing twice
                                iterateFlag=true; ite++;
                            }
                        }
                    }
                }
            }
            pixPosition(xt, NPix, xtG);                // Position on grid after diffusion

            if (ite==nite) {
                printf("Run out of iterations.\n");
            }
            
            // Periodic boundary condition
            if (xt[0]>1) {
                xt[0]-=1;
                xjmp+=1;
            }
            else if (xt[0]<0) {
                xt[0]+=1;
                xjmp-=1;
            }
            
            if (xt[1]>1) {
                xt[1]-=1;
                yjmp+=1;
            }
            else if (xt[1]<0) {
                xt[1]+=1;
                yjmp-=1;
            }
            
            if (xt[2]>1) {
                xt[2]-=1;
                zjmp+=1;
            }
            else if (xt[2]<0) {
                xt[2]+=1;
                zjmp-=1;
            }
            
            // ********** End one step **********
            
            if ( (i%Tstep) ==0 ) { // Save moment tensor for dx^2 and dx^4, and signal for the b-table
                tidx=i/Tstep;
                
                if ( ICSFlag ) { atomAdd(&NParICS[tidx],1); }
                if ( initFlag==4 ) {
                    bidx=floor( sqrt( pow2(xt[0]-0.5) + pow2(xt[1]-0.5) )*nbin*2 );
                    if (bidx<nbin) {
                        atomAdd(&NParBin[nbin*tidx+bidx],1);
                    }
                }
                
                
                dx=(xt[0]+xjmp-xi[0])*res;
                dy=(xt[1]+yjmp-xi[1])*res;
                dz=(xt[2]+zjmp-xi[2])*res;
                
                atomAdd(&dx2[6*tidx+0],dx*dx);
                atomAdd(&dx2[6*tidx+1],dx*dy);
                atomAdd(&dx2[6*tidx+2],dx*dz);
                atomAdd(&dx2[6*tidx+3],dy*dy);
                atomAdd(&dx2[6*tidx+4],dy*dz);
                atomAdd(&dx2[6*tidx+5],dz*dz);
                
                atomAdd(&dx4[15*tidx+0],dx*dx*dx*dx);
                atomAdd(&dx4[15*tidx+1],dx*dx*dx*dy);
                atomAdd(&dx4[15*tidx+2],dx*dx*dx*dz);
                atomAdd(&dx4[15*tidx+3],dx*dx*dy*dy);
                atomAdd(&dx4[15*tidx+4],dx*dx*dy*dz);
                atomAdd(&dx4[15*tidx+5],dx*dx*dz*dz);
                atomAdd(&dx4[15*tidx+6],dx*dy*dy*dy);
                atomAdd(&dx4[15*tidx+7],dx*dy*dy*dz);
                atomAdd(&dx4[15*tidx+8],dx*dy*dz*dz);
                atomAdd(&dx4[15*tidx+9],dx*dz*dz*dz);
                atomAdd(&dx4[15*tidx+10],dy*dy*dy*dy);
                atomAdd(&dx4[15*tidx+11],dy*dy*dy*dz);
                atomAdd(&dx4[15*tidx+12],dy*dy*dz*dz);
                atomAdd(&dx4[15*tidx+13],dy*dz*dz*dz);
                atomAdd(&dx4[15*tidx+14],dz*dz*dz*dz);
                
                for (int j=0; j<Nbvec; j++) {
                    qx=sqrt(btab[4*j]/TD[tidx])*( btab[4*j+1]*dx + btab[4*j+2]*dy + btab[4*j+3]*dz );
                    atomAdd(&sig[Nbvec*tidx+j],cos(qx));
                }
            }
            
        }
    }
    state[idx]=localstate;
}

    
//********** Define tissue parameters **********

int main(int argc, char *argv[]) {
    
    clock_t begin=clock();
    clock_t end=clock();
    
    // Define index number
    int i=0, j=0;
    
    //********** Load mictostructure **********
    
    double dt=0;                // Time step in ms
    int TN=0;                   // Number of time steps
    int NPar=0;                 // Number of time points to record
    int Nbvec=0;                // Number of gradient directions
    
    double Din=0;               // Diffusion coefficient inside the axon in µm^2/ms
    double Dex=0;               // Diffusion coefficient outside the axon in µm^2/ms
    double kappa=0;             // Permeability of a lipid bi-layer in µm/ms
    int initFlag=1;             // Initial position: 1=ICS, 2=ECS, 3=ICS+ECS+myelin, 4=center
    int thread_per_block=0;     // Number of threads per block
    
    unsigned int NPix=0, NAx=0;
    double res=0;
    
    // simulation parameter
    ifstream myfile0 ("simParamInput.txt", ios::in);
    myfile0>>dt; myfile0>>TN; myfile0>>NPar; myfile0>>Nbvec;
    myfile0>>Din; myfile0>>Dex; myfile0>>kappa;
    myfile0>>initFlag;
    myfile0>>thread_per_block;
    myfile0.close();
    
    double stepIN=sqrt(4.0*dt*Din);     // Step size in ICS in µm
    double stepEX=sqrt(4.0*dt*Dex);     // Step size in ECS in µm
    
    double stepINz=sqrt(2.0*dt*Din);
    double stepEXz=sqrt(2.0*dt*Dex);
    
    // resolution
    ifstream myfile1 ("phantom_res.txt", ios::in);
    myfile1>>res;
    myfile1.close();
    
    // Pixel # along each side
    ifstream myfile2 ("phantom_NPix.txt", ios::in);
    myfile2>>NPix;
    myfile2.close();
    
    // Pixelized matrix A indicating axon labels
    thrust::host_vector<unsigned int> APix(NPix*NPix);
    ifstream myfile3 ("phantom_APix.txt", ios::in);
    for (i=0; i<NPix*NPix; i++){
        myfile3>>APix[i];
    }
    myfile3.close();
    
    // Number of axons
    ifstream myfile4 ("phantom_NAx.txt", ios::in);
    myfile4>>NAx;
    myfile4.close();
    
    // Circle center of x-coordinate
    thrust::host_vector<double> xCir(NAx);
    ifstream myfile5 ("phantom_xCir.txt", ios::in);
    for (i=0; i<NAx; i++){
        myfile5>>xCir[i];
    }
    myfile5.close();
    
    // Circle center of y-coordinate
    thrust::host_vector<double> yCir(NAx);
    ifstream myfile6 ("phantom_yCir.txt", ios::in);
    for (i=0; i<NAx; i++){
        myfile6>>yCir[i];
    }
    myfile6.close();
    
    // Circle outer radius
    thrust::host_vector<double> rCir(NAx);
    ifstream myfile7 ("phantom_rCir.txt", ios::in);
    for (i=0; i<NAx; i++){
        myfile7>>rCir[i];
    }
    myfile7.close();
    
    // The smallest number, which is > NAx, in the base 10
    unsigned int Nmax=0;
    ifstream myfile8 ("phantom_Nmax.txt", ios::in);
    myfile8>>Nmax;
    myfile8.close();
    
    // btable: [bval gx gy gz]
    thrust::host_vector<double> btab(Nbvec*4);
    ifstream myfile9 ("btable.txt", ios::in);
    for (i=0; i<Nbvec*4; i++) {
        myfile9>>btab[i];
    }
    myfile9.close();
    
    // Diffusion time
    thrust::host_vector<double> TD(timepoints);
    for (i=0; i<timepoints; i++){
        TD[i]=(i*(TN/timepoints)+1)*dt;
    }
    
    //********** Initialize Particle Positions in IAS *********
    const double probE=Pi/4.0*stepEX*kappa/Dex;          // Probability constant from ECS to myelin
    const double probI=Pi/4.0*stepIN*kappa/Din;          // Probability constant from ICS to myelin
    stepEX/=res; stepIN/=res;                            // Normalize the step size with the voxel size
    stepEXz/=res; stepINz/=res;
    
    // Create translate flag to speed up the code
    double Lpix = 1.0/static_cast<double>(NPix);
    thrust::host_vector<bool> translateFlag(NAx);
    for (i=0; i<NAx; i++) {
        if ( ((xCir[i]+rCir[i]+2*Lpix)>=1) | ((xCir[i]-rCir[i]-2*Lpix)<=0) | ((yCir[i]+rCir[i]+2*Lpix)>=1) | ((yCir[i]-rCir[i]-2*Lpix)<=0) ) {
            translateFlag[i]=true;
        } else {
            translateFlag[i]=false;
        }
    }
    
    // ********** Simulate diffusion **********
    
    // Initialize seed
    unsigned long seed=0;
    FILE *urandom;
    urandom = fopen("/dev/random", "r");
    fread(&seed, sizeof (seed), 1, urandom);
    fclose(urandom);
    
    // Initialize state of RNG
    int blockSize = thread_per_block;
    int numBlocks = (NPar + blockSize - 1) / blockSize;
    cout<<numBlocks<<endl<<blockSize<<endl;
    
    thrust::device_vector<curandStatePhilox4_32_10_t> devState(numBlocks*blockSize);
    setup_kernel<<<numBlocks, blockSize>>>(devState.data().get(),seed);
    
    // Initialize output
    thrust::host_vector<double> dx2(timepoints*6);
    thrust::host_vector<double> dx4(timepoints*15);
    thrust::host_vector<double> NParICS(timepoints);
    thrust::host_vector<double> NParBin(timepoints*nbin);
    thrust::host_vector<double> sig(timepoints*Nbvec);
    for (i=0;i<timepoints*6;i++){ dx2[i]=0; }
    for (i=0;i<timepoints*15;i++){ dx4[i]=0; }
    for (i=0;i<timepoints;i++){ NParICS[i]=0; }
    for (i=0;i<timepoints*nbin;i++){ NParBin[i]=0; }
    for (i=0;i<timepoints*Nbvec;i++) { sig[i]=0; }
    
    // Move data from host to device
    thrust::device_vector<double> d_dx2=dx2;
    thrust::device_vector<double> d_dx4=dx4;
    thrust::device_vector<double> d_NParICS=NParICS;
    thrust::device_vector<double> d_NParBin=NParBin;
    thrust::device_vector<double> d_sig=sig;
    thrust::device_vector<double> d_xCir=xCir;
    thrust::device_vector<double> d_yCir=yCir;
    thrust::device_vector<double> d_rCir=rCir;
    thrust::device_vector<bool> d_translateFlag=translateFlag;
    thrust::device_vector<unsigned int> d_APix=APix;
    thrust::device_vector<double> d_btab=btab;
    thrust::device_vector<double> d_TD=TD;
    
    // Parallel computation
    begin=clock();
    propagate<<<numBlocks, blockSize>>>(devState.data().get(), d_dx2.data().get(), d_dx4.data().get(), d_NParICS.data().get(), d_NParBin.data().get(), d_sig.data().get(), TN, NPar, Nbvec, res, stepIN, stepEX, stepINz, stepEXz, probI, probE, NPix, Nmax, initFlag, d_xCir.data().get(), d_yCir.data().get(), d_rCir.data().get(), d_translateFlag.data().get(), d_APix.data().get(), d_btab.data().get(), d_TD.data().get());
    cudaDeviceSynchronize();
    end=clock();
    cout << "Done! Elpased time "<<double((end-begin)/CLOCKS_PER_SEC) << " s"<< endl;
    
    thrust::copy(d_dx2.begin(), d_dx2.end(), dx2.begin());
    thrust::copy(d_dx4.begin(), d_dx4.end(), dx4.begin());
    thrust::copy(d_NParICS.begin(), d_NParICS.end(), NParICS.begin());
    thrust::copy(d_NParBin.begin(), d_NParBin.end(), NParBin.begin());
    thrust::copy(d_sig.begin(), d_sig.end(), sig.begin());
    
    ofstream fdx2out("dx2_diffusion.txt");
    ofstream fdx4out("dx4_diffusion.txt");
    ofstream fNParICSout("NParICS.txt");
    ofstream fNParBinout("NParBin.txt");
    ofstream fsigout("sig_diffusion.txt");
    fdx2out.precision(15);
    fdx4out.precision(15);
    fNParICSout.precision(15);
    fNParBinout.precision(15);
    fsigout.precision(15);
    double dr = 0.5*res/nbin;
    for (i=0; i<timepoints; i++) {
        for (j=0; j<6; j++) {
            if (j==5) {
                fdx2out<<dx2[i*6+j]<<endl;
            } else {
                fdx2out<<dx2[i*6+j]<<"\t";
            }
        }
        for (j=0; j<15; j++) {
            if (j==14) {
                fdx4out<<dx4[i*15+j]<<endl;
            } else {
                fdx4out<<dx4[i*15+j]<<"\t";
            }
        }
        fNParICSout<<NParICS[i]<<endl;
        for (j=0; j<nbin; j++) {
            if (j==nbin-1){
                fNParBinout<<NParBin[i*nbin+j]/(Pi*dr*dr*(2*j+1))<<endl;
            } else {
                fNParBinout<<NParBin[i*nbin+j]/(Pi*dr*dr*(2*j+1))<<"\t";
            }
        }
        for(j=0; j<Nbvec; j++) {
            if (j==Nbvec-1){
                fsigout<<sig[i*Nbvec+j]<<endl;
            } else {
                fsigout<<sig[i*Nbvec+j]<<"\t";
            }
        }
    }
    fdx2out.close();
    fdx4out.close();
    fNParICSout.close();
    fNParBinout.close();
    fsigout.close();
    
    ofstream paraout ("sim_para.txt");
    paraout<<dt<<endl<<TN<<endl<<NPar<<endl<<Nbvec<<endl;
    paraout<<Din<<endl<<Dex<<endl;
    paraout<<kappa<<endl<<initFlag<<endl<<res<<endl;
    paraout.close();
    
    ofstream TDout("diff_time.txt");
    for (i=0; i<timepoints; i++){
        TDout<<(i*(TN/timepoints)+1)*dt<<endl;
    }
    TDout.close();
}

