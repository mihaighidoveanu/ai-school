//
//  ReadersAndWriters_ReadersPriority.hpp
//  Lab
//
//  Created by Ciprian on 10/13/18.
//  Copyright © 2018 Ciprian. All rights reserved.
//

#ifndef ReadersAndWriters_ReadersPriority_hpp
#define ReadersAndWriters_ReadersPriority_hpp

#include <mutex>
#include "MySemphore.hpp"
#include "Utils.hpp"
#include "SimulationSettings.h"

class ReadersAndWriters_ReadPriority
{
public:
    
    void Write(const int id)
    {
        // Try to enter in safe zone
        m_resourcesLock.wait(); // We have to wait every readers and writers in the system
        
        // We are in safe zone...do WRITING
        std::ostringstream msgin; msgin <<"ID " << id <<  " +++++ I'm WRITING.\n";
        Utils::writeConsoleMessageSynchronized(msgin);
        
        Utils::SleepRandomTime(100, 500);
        
        std::ostringstream msgout; msgout << "ID " << id << " ----- Finished WRITING\n";
        Utils::writeConsoleMessageSynchronized(msgout);
        //------------
        
        
        // Leaving safe zone
        m_resourcesLock.signal(); // Add a signal, maybe someone else will continue
    }
    
    void Read(const int id)
    {
        // Ensure mutual exclusion for readers. Semaphore used as classic mutex, modern way is to use lock guards as used in the semaphore
        
        // Try to enter in safe zone
        m_readersLock.wait();
        m_numReaders++;
        if (m_numReaders == 1) // What would happen without the initial lock ?
        {
            // If first reader, block writers.
            m_resourcesLock.wait();
        }
        m_readersLock.signal(); // Release the semaphore entry
        
        // We are in safe zone...do READING
        std::ostringstream msgin; msgin << "ID: " << id << " +++++ I'm READING.\n";
        Utils::writeConsoleMessageSynchronized(msgin);
        Utils::SleepRandomTime(100, 500);
        std::ostringstream msgout; msgout << "ID: " << id << " ----- Finished READING\n";
        Utils::writeConsoleMessageSynchronized(msgout);
        //----
        
        // Exiting from safe zone
        m_readersLock.wait();
        m_numReaders--;
        if (m_numReaders == 0)
            m_resourcesLock.signal(); // RElease the resource, give a chance to writer
        m_readersLock.signal();
    }
    
    ReadersAndWriters_ReadPriority()
    {
        m_numReaders = 0;
        m_readersLock.init(1);
        m_resourcesLock.init(1);
    }
    
private:
    int m_numReaders;
    MySemaphore m_readersLock; // Controls access to readers counters and logic
    MySemaphore m_resourcesLock; // Control access between readers versus readers to the shared resource
};
#endif /* ReadersAndWriters_ReadersPriority_hpp */
