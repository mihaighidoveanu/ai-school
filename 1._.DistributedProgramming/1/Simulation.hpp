//
//  Simulation.hpp
//  Lab
//
//  Created by Ciprian on 10/13/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef Simulation_hpp
#define Simulation_hpp

#include "Simulation.hpp"
#include "Utils.hpp"
#include <random>


template <class Policy>
class Simulation
{
private:
    
    Policy m_policy ;
    
    void runReader(const int id)
    {
        //std::cout<<"Reader " << id <<" is alive !\n";
        for (int i = 0; i < NUM_SIM_CYCLES_FOR_READERS; i++)
        {
            m_policy.Read(id);
            Utils::SleepMs(100);
        }
        
        //std::cout<<"Reader " << id <<" is dead !\n";
    }
    
    void runWriter(const int id)
    {
        //std::cout<<"Writer " << id <<" is alive !\n";
        for (int i = 0; i < NUM_SIM_CYCLES_FOR_WRITERS; i++)
        {
            m_policy.Write(id);
            Utils::SleepMs(100);
        }
        
        //std::cout<<"Writer " << id <<" is dead !\n";
    }
    
public:
    void run_simulation()
    {
        std::thread entities[NUM_READERS + NUM_WRITERS];
        
        // Create the threads ; don't start with readers then writers since this will bias more to readers execution
        // Create a pool then shuffle.
        enum ENTITY_TYPE : bool { ENTITY_READER = false, ENTITY_WRITER = true};
        std::array<ENTITY_TYPE, NUM_WRITERS + NUM_READERS> entityToType;
        
        for (int i = 0; i < NUM_READERS; i++)
        {
            entityToType[i] = ENTITY_READER;
        }
        
        for (int i = NUM_WRITERS; i < NUM_WRITERS + NUM_READERS; i++)
        {
            entityToType[i] = ENTITY_WRITER;
        }
        
        // Shuffle the array
        const unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(entityToType.begin(), entityToType.end(), std::default_random_engine(seed));
        
        for (int i = 0; i < NUM_READERS + NUM_WRITERS; i++)
        {
            auto func = entityToType[i] == ENTITY_READER ? &Simulation::runReader : &Simulation::runWriter;
            entities[i] = std::thread(func, this, i);
        }
        
        // Wait everybody
        for (int i = 0; i < NUM_READERS + NUM_WRITERS; i++)
        {
            entities[i].join();
        }
    }
};


#endif /* Simulation_hpp */
