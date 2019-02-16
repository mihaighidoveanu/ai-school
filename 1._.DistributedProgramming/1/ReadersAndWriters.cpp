//
//  ReadersAndWriters.cpp
//  Lab
//
//  Created by Ciprian on 10/10/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#include "MySemphore.hpp"
#include "MyFairSemaphore2.hpp"
#include "SimulationSettings.h"
#include "Simulation.hpp"
#include "ReadersAndWriters_Fair.hpp"
#include "ReadersAndWriters_WritersPriority.hpp"
#include "ReadersAndWriters_ReadersPriority.hpp"
#include <iostream>
#include <stdarg.h>
#include <sstream>
#include <algorithm>
#include <array>
#include <random>
#include "Utils.hpp"



int main()
{
    Simulation<ReadersAndWriters_Fair> sim; // Set the desired policy as template parameter
    sim.run_simulation();
    return 0;
}
