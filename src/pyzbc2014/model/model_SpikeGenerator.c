/*
  This file is part of a variant of version 5.2 of the Zilany, Bruce, and Carney (2014)
  auditory-nerve model. Licensing information can be found below. See README.md for a list
  of changes made versus the original version, which is available at:
  https://www.urmc.rochester.edu/labs/carney/publications-code/auditory-models.aspx.


  Papers related to this code are cited below:

  Zilany, M. S., & Bruce, I. C. (2006). Modeling auditory-nerve responses for high sound 
  pressure levels in the normal and impaired auditory periphery. The Journal of the 
  Acoustical Society of America, 120(3), 1446-1466.

  * Zilany, M.S.A., Bruce, I.C., Nelson, P.C., and Carney, L.H. (2009). "A
  Phenomenological model of the synapse between the inner hair cell and auditory
  nerve : Long-term adaptation with power-law dynamics," Journal of the
  Acoustical Society of America 126(5): 2390-2412.
 
  Ibrahim, R. A., and Bruce, I. C. (2010). "Effects of peripheral tuning
  on the auditory nerve's representation of speech envelope and temporal fine
  structure cues," in The Neurophysiological Bases of Auditory Perception, eds.
  E. A. Lopez-Poveda and A. R. Palmer and R. Meddis, Springer, NY, pp. 429-438.

  Zilany, M.S.A., Bruce, I.C., Ibrahim, R.A., and Carney, L.H. (2013).
  "Improved parameters and expanded simulation options for a model of the
  auditory periphery," in Abstracts of the 36th ARO Midwinter Research Meeting.

  * Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded 
  simulation options for a model of the auditory periphery. The Journal of the Acoustical 
  Society of America, 135(1), 283-286.

  Please cite these papers marked with asterisks (*) if you publish any research results 
  obtained with this code or any modified versions of this code.


  Zilany, Bruce, Carney (2014) auditory-nerve model source code, Copyright (C) 2024 
    M. S. A. Zilany <msazilany@gmail.com> 
    Ian C. Bruce <ibruce@ieee.org> 
    Laurel H. Carney <laurel_carney@urmc.rochester.edu>
    Daniel R. Guest <daniel_guest@urmc.rochester.edu>

  This program is free software: you can redistribute it and/or modify it under the terms of
  the GNU Affero General Public License as published by the Free Software Foundation, either
  version 3 of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
  See the GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License along with this
  program. If not, see <http://www.gnu.org/licenses/>.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "complex.h"

#define MAXSPIKES 1000000
#ifndef TWOPI
#define TWOPI 6.28318530717959
#endif

#ifndef __max
#define __max(a,b) (((a) > (b))? (a): (b))
#endif

#ifndef __min
#define __min(a,b) (((a) < (b))? (a): (b))
#endif



/* ------------------------------------------------------------------------------------ */
/* Pass the output of Synapse model through the Spike Generator */

/* The spike generator now uses a method coded up by B. Scott Jackson (bsj22@cornell.edu)
   Scott's original code is available from Laurel Carney's web site at:
   http://www.urmc.rochester.edu/smd/Nanat/faculty-research/lab-pages/LaurelCarney/auditory-models.cfm
   That link is broken but you can find the code at this link:
   https://www.urmc.rochester.edu/labs/carney/publications-code/auditory-models.aspx
   Scroll down to Jackson and Carney, JARO 2005. 
   )
*/

int SpikeGenerator(double *synouttmp, double *randNums, double tdres, int totalstim, int nrep, double *sptime)
{
    double c0, s0, c1, s1, dead;
    int nspikes, k, NoutMax, Nout, deadtimeIndex, randBufIndex;
    double deadtimeRnd, endOfLastDeadtime, refracMult0, refracMult1, refracValue0, refracValue1;
    double Xsum, unitRateIntrvl, countTime, DT;

    // The random numbers are now calculated outside this function and passed
    // as a pointer to the array.
    // double *randNums;

    c0 = 0.5;
    s0 = 0.001;
    c1 = 0.5;
    s1 = 0.0125;
    dead = 0.00075;

    DT = totalstim * tdres * nrep; /* Total duration of the rate function */

    NoutMax = (long)ceil(totalstim * nrep * tdres / dead);

    // randNums = generate_random_numbers(NoutMax + 1);
    Nout = 0;
    randBufIndex = 0;

    /* Calculate useful constants */
    deadtimeIndex = (long)floor(dead / tdres); /* Integer number of discrete time bins within deadtime */
    deadtimeRnd = deadtimeIndex * tdres;       /* Deadtime rounded down to length of an integer number of discrete time bins */

    refracMult0 = 1 - tdres / s0; /* If y0(t) = c0*exp(-t/s0), then y0(t+tdres) = y0(t)*refracMult0 */
    refracMult1 = 1 - tdres / s1; /* If y1(t) = c1*exp(-t/s1), then y1(t+tdres) = y1(t)*refracMult1 */

    /* Calculate effects of a random spike before t=0 on refractoriness and the time-warping sum at t=0 */
    if (synouttmp[0] == 0)
    {
        endOfLastDeadtime = 0.0;
    }
    else
    {
        endOfLastDeadtime = log(randNums[randBufIndex++]) / synouttmp[0]; /* End of last deadtime before t=0 */
    }
    refracValue0 = c0 * exp(endOfLastDeadtime / s0); /* Value of first exponential in refractory function */
    refracValue1 = c1 * exp(endOfLastDeadtime / s1); /* Value of second exponential in refractory function */
    Xsum = synouttmp[0] * (-endOfLastDeadtime + c0 * s0 * (exp(endOfLastDeadtime / s0) - 1) + c1 * s1 * (exp(endOfLastDeadtime / s1) - 1));
    /* Value of time-warping sum */
    /*  ^^^^ This is the "integral" of the refractory function ^^^^ (normalized by 'tdres') */

    /* Calculate first interspike interval in a homogeneous, unit-rate Poisson process (normalized by 'tdres') */
    unitRateIntrvl = -log(randNums[randBufIndex++]) / tdres;
    /* NOTE: Both 'unitRateInterval' and 'Xsum' are divided (or normalized) by 'tdres' in order to reduce calculation time.
        This way we only need to divide by 'tdres' once per spike (when calculating 'unitRateInterval'), instead of
        multiplying by 'tdres' once per time bin (when calculating the new value of 'Xsum').                         */

    countTime = tdres;
    for (k = 0; (k < totalstim * nrep) && (countTime < DT); ++k, countTime += tdres, refracValue0 *= refracMult0, refracValue1 *= refracMult1) /* Loop through rate vector */
    {
        if (synouttmp[k] > 0) /* Nothing to do for non-positive rates, i.e. Xsum += 0 for non-positive rates. */
        {
            Xsum += synouttmp[k] * (1 - refracValue0 - refracValue1); /* Add synout*(refractory value) to time-warping sum */

            if (Xsum >= unitRateIntrvl) /* Spike occurs when time-warping sum exceeds interspike "time" in unit-rate process */
            {
                sptime[Nout] = countTime;
                Nout = Nout + 1;
                unitRateIntrvl = -log(randNums[randBufIndex++]) / tdres;
                Xsum = 0;

                /* Increase index and time to the last time bin in the deadtime, and reset (relative) refractory function */
                k += deadtimeIndex;
                countTime += deadtimeRnd;
                refracValue0 = c0;
                refracValue1 = c1;
            }
        }
    } /* End of rate vector loop */

    // free(randNums);  # array came from outside, so nothing to free. 
    nspikes = Nout; /* Number of spikes that occurred. */
    return (sptime);
}


